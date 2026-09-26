// water/box3d.js — two-way coupling between the fluid and a Box3D world.
//
//   const coupling = createBox3DCoupling({ sim, b3, world });
//   // every frame, before stepping physics:
//   coupling.update();            // colliders ← bodies, bodies ← fluid impulses
//   b3.b3World_Step(world, dt, 4);
//   sim.update(dt);
//
// Each frame the Box3D shapes near the water become solver colliders:
//   sphere → sphere, capsule → capsule, hull → its oriented bounding box in
//   the body frame (exact for boxes; conservative for other convex hulls).
// Mesh / heightfield / compound shapes are skipped (register terrain with
// sim.addHeightfield instead). Static bodies are fixed colliders, kinematic
// bodies push water with their velocity, and dynamic bodies additionally
// receive the fluid's reaction: the impulses the solver accumulated on their
// colliders are applied as a linear impulse at the collider origin plus an
// angular impulse — buoyancy, drag and splashes all come out of that one
// momentum exchange (particle mass = waterDensity · spacing³, so a body of
// density ρ floats with ρ/1000 of its volume submerged).

import { vec3, quat } from 'math';

const TYPE_SPHERE = 5, TYPE_CAPSULE = 0, TYPE_HULL = 3;
const UP = /* @__PURE__ */ vec3.fromValues(0, 1, 0);

// module scratch (one coupling update runs at a time on the main thread)
const _shape_a = /* @__PURE__ */ vec3.create();
const _shape_b = /* @__PURE__ */ vec3.create();
const _shape_axis = /* @__PURE__ */ vec3.create();
const _shape_q = /* @__PURE__ */ quat.create();
const _vb = [0, 0, 0];

// Solid volume of a collider description (Archimedes bound).
function colliderVolume(desc) {
  if (desc.type === 'sphere') return (4 / 3) * Math.PI * desc.radius ** 3;
  if (desc.type === 'capsule') return Math.PI * desc.radius ** 2 * (2 * desc.halfHeight + (4 / 3) * desc.radius);
  if (desc.type === 'box') return 8 * desc.size[0] * desc.size[1] * desc.size[2];
  return 0;
}

const idKey = (id) => `${id.index1}:${id.world0}:${id.generation}`;

/**
 * @param {object} o
 *   sim          createSimulation() result
 *   b3, world    Box3D module + world id
 *   colliders    extra solver colliders (planes, containers, heightfields…)
 *   region       'auto' (water AABB + margin) | {min,max} | null (whole world)
 *   margin       AABB growth for 'auto' (m, default 1)
 *   filter       (bodyId, shapeId) => boolean — include this shape?
 *   impulseScale scale on the fluid→body reaction (1 = physical)
 */
export function createBox3DCoupling({
  sim, b3, world, colliders = [], region = 'auto', margin = 1, filter = null, impulseScale = 1,
}) {
  const TYPE = {
    static: b3.b3BodyType.b3_staticBody.value,
    kinematic: b3.b3BodyType.b3_kinematicBody.value,
    dynamic: b3.b3BodyType.b3_dynamicBody.value,
  };
  const queryFilter = b3.b3DefaultQueryFilter();
  const _p = [0, 0, 0], _q = [0, 0, 0, 1], _v = [0, 0, 0], _w = [0, 0, 0], _com = [0, 0, 0];
  // Impulse slots are STABLE per shape: with the threaded solver, impulses
  // arrive a frame or two after the colliders that produced them, so they
  // must not depend on this frame's collider order.
  const slotOfShape = new Map();   // shape key → slot
  const slotInfo = [];             // slot → { body, origin, lastSeen }
  const freeSlots = [];
  let frame = 0;
  let extra = colliders;
  const stats = { bodies: 0, colliders: 0, skipped: 0 };

  function waterAABB() {
    if (region && region !== 'auto') return [...region.min, ...region.max];
    if (region === null) return [-1e9, -1e9, -1e9, 1e9, 1e9, 1e9];
    const n = sim.count, p = sim.positions;
    if (n === 0) return null;
    let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
      if (z < z0) z0 = z; if (z > z1) z1 = z;
    }
    return [x0 - margin, y0 - margin, z0 - margin, x1 + margin, y1 + margin, z1 + margin];
  }

  function slotFor(shapeId, body, origin, volume) {
    const key = idKey(shapeId);
    let slot = slotOfShape.get(key);
    if (slot === undefined) {
      slot = freeSlots.length ? freeSlots.pop() : slotInfo.length;
      if (slot >= sim.maxColliders) return -1;
      slotOfShape.set(key, slot);
    }
    const prev = slotInfo[slot];
    slotInfo[slot] = {
      key, body, origin, volume, lastSeen: frame,
      bankJ: prev?.key === key ? prev.bankJ : [0, 0, 0], bankL: prev?.key === key ? prev.bankL : [0, 0, 0],
    };
    return slot;
  }

  function recycleSlots() {
    // free slots unseen for a few frames (late impulses for them are dropped)
    for (let slot = 0; slot < slotInfo.length; slot++) {
      const s = slotInfo[slot];
      if (s && frame - s.lastSeen > 3) { slotOfShape.delete(s.key); slotInfo[slot] = null; freeSlots.push(slot); }
    }
  }

  // Physical bound on what the fluid can hand a body in one frame: the
  // Archimedes force of the fully submerged shape over the frame, plus an
  // inelastic exchange with the fluid mass it touched,
  //   |J| ≤ ρ_w·g·V·T + M·m_c/(M + m_c) · max(0, (v_fluid − v_body)·Ĵ).
  // Within a step the solver treats bodies as immovable, so at a fast entry
  // every touched particle is pushed to the body's velocity and the full
  // reaction would rebound a light body faster than it arrived (a 10 kg
  // crate hitting water at 2.9 m/s left at +2.8 m/s). Resting floaters and
  // sinking bodies are below the bound and unaffected. Clipped impulse is
  // banked and offered again next frame (halved each frame), so ordinary
  // frame-to-frame pressure noise keeps its mean while an impact spike is
  // spread out and partly dissipated.
  function capImpulse(s, J, L, contacts, slot, time) {
    for (let i = 0; i < 3; i++) { J[i] += s.bankJ[i]; L[i] += s.bankL[i]; s.bankJ[i] = 0; s.bankL[i] = 0; }
    const jn = Math.hypot(J[0], J[1], J[2]);
    if (jn === 0) return;
    const dp = sim.params;
    const g = Math.hypot(dp.gravity[0], dp.gravity[1], dp.gravity[2]);
    const M = b3.b3Body_GetMass(s.body);
    const N = contacts ? contacts[slot * 4] : 0;
    const mc = N * dp.particleMass;
    let rel = 0;
    if (N > 0) {
      b3.b3Body_GetWorldPointVelocity(_vb, s.body, s.origin);
      const c = slot * 4;
      rel = ((contacts[c + 1] - _vb[0]) * J[0] + (contacts[c + 2] - _vb[1]) * J[1] + (contacts[c + 3] - _vb[2]) * J[2]) / jn;
      if (rel < 0) rel = 0;
    }
    const cap = dp.waterDensity * g * s.volume * time + (M > 0 ? (M * mc) / (M + mc) : mc) * rel;
    if (jn > cap) {
      const k = cap / jn;
      for (let i = 0; i < 3; i++) {
        s.bankJ[i] = J[i] * (1 - k) * 0.5; s.bankL[i] = L[i] * (1 - k) * 0.5;
        J[i] *= k; L[i] *= k;
      }
    }
  }

  function applyImpulses() {
    const { impulses, time, contacts } = sim.takeImpulses();
    for (let slot = 0; slot < slotInfo.length; slot++) {
      const s = slotInfo[slot];
      if (!s || slot * 6 + 5 >= impulses.length) continue;
      const o = slot * 6;
      const J = [impulses[o] * impulseScale, impulses[o + 1] * impulseScale, impulses[o + 2] * impulseScale];
      const L = [impulses[o + 3] * impulseScale, impulses[o + 4] * impulseScale, impulses[o + 5] * impulseScale];
      if (!b3.b3Body_IsValid(s.body)) continue;
      capImpulse(s, J, L, contacts, slot, time);
      if (J[0] === 0 && J[1] === 0 && J[2] === 0 && L[0] === 0 && L[1] === 0 && L[2] === 0) continue;
      // J applied at the collider origin + L (moments were taken about that origin)
      b3.b3Body_ApplyLinearImpulse(s.body, J, s.origin, true);
      b3.b3Body_ApplyAngularImpulse(s.body, L, true);
    }
  }

  function shapeCollider(shapeId, body, type) {
    b3.b3Body_GetPosition(_p, body);
    b3.b3Body_GetRotation(_q, body);
    const st = b3.b3Shape_GetType(shapeId).value;
    let desc;
    if (st === TYPE_SPHERE) {
      const s = b3.b3Shape_GetSphere(shapeId);
      const c = vec3.transformQuat(_shape_a, s.center, _q);
      desc = { type: 'sphere', position: vec3.add(vec3.create(), _p, c), radius: s.radius };
    } else if (st === TYPE_CAPSULE) {
      const s = b3.b3Shape_GetCapsule(shapeId);
      const a = vec3.transformQuat(_shape_a, s.center1, _q), b = vec3.transformQuat(_shape_b, s.center2, _q);
      const axis = vec3.subtract(_shape_axis, b, a);
      const len = vec3.length(axis);
      // shortest arc +Y → axis (the capsule collider's local axis is +Y)
      if (len > 1e-9) quat.rotationTo(_shape_q, UP, vec3.scale(axis, axis, 1 / len));
      else quat.identity(_shape_q);
      const mid = vec3.lerp(vec3.create(), a, b, 0.5);
      desc = {
        type: 'capsule', position: vec3.add(mid, mid, _p), rotation: quat.clone(_shape_q),
        radius: s.radius, halfHeight: len / 2,
      };
    } else if (st === TYPE_HULL) {
      const v = b3.b3Shape_GetHullVertices(shapeId); // body-local
      let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
      for (let i = 0; i < v.length; i += 3) {
        if (v[i] < x0) x0 = v[i]; if (v[i] > x1) x1 = v[i];
        if (v[i + 1] < y0) y0 = v[i + 1]; if (v[i + 1] > y1) y1 = v[i + 1];
        if (v[i + 2] < z0) z0 = v[i + 2]; if (v[i + 2] > z1) z1 = v[i + 2];
      }
      const c = vec3.transformQuat(_shape_a, vec3.set(_shape_a, (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), _q);
      desc = {
        type: 'box', position: vec3.add(vec3.create(), _p, c), rotation: quat.clone(_q),
        size: [(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2],
      };
    } else {
      stats.skipped++;
      return null;
    }
    if (type !== TYPE.static) {
      // surface velocity at the collider origin: v + ω × (origin − com)
      b3.b3Body_GetLinearVelocity(_v, body);
      b3.b3Body_GetAngularVelocity(_w, body);
      b3.b3Body_GetWorldCenterOfMass(_com, body);
      const r = vec3.subtract(_shape_a, desc.position, _com);
      desc.velocity = vec3.add(vec3.create(), _v, vec3.cross(_shape_b, _w, r));
      desc.angularVelocity = vec3.clone(_w);
    }
    desc.dynamic = type === TYPE.dynamic;
    if (desc.dynamic) desc.mass = b3.b3Body_GetMass(body);
    return desc;
  }

  function buildColliders() {
    frame++;
    const list = extra.map((c) => ({ ...c, slot: c.slot ?? -1 }));
    const aabb = waterAABB();
    const bodies = new Set();
    if (aabb) {
      const shapes = [];
      b3.b3World_OverlapAABB(world, aabb, queryFilter, (shapeId) => { shapes.push(shapeId); return true; });
      for (const shapeId of shapes) {
        if (list.length >= sim.maxColliders) break;
        const body = b3.b3Shape_GetBody(shapeId);
        if (filter && !filter(body, shapeId)) continue;
        const type = b3.b3Body_GetType(body).value;
        const desc = shapeCollider(shapeId, body, type);
        if (!desc) continue;
        bodies.add(idKey(body));
        desc.slot = desc.dynamic ? slotFor(shapeId, body, desc.position, colliderVolume(desc)) : -1;
        if (desc.slot < 0) desc.dynamic = false;
        list.push(desc);
      }
    }
    stats.bodies = bodies.size;
    stats.colliders = list.length;
    recycleSlots();
    sim.setColliders(list);
  }

  return {
    stats,
    /** Replace the extra (non-Box3D) colliders. */
    setColliders(list) { extra = list; },
    /** Push fluid impulses into bodies, then mirror bodies into colliders. */
    update() {
      applyImpulses();
      buildColliders();
    },
  };
}
