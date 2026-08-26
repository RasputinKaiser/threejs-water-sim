// box3d-debug.js — sync + debug visualization layer between a Box3D world and three.js.
//
// - PhysicsSync: one mesh per body (geometry derived from shape type), updated each frame.
// - DebugDraw: toggles for AABB wireframes, contact points, velocity vectors, body axes,
//   and world bounds. All zero-alloc-ish: pooled buffers, reused scratch arrays.

import * as THREE from 'three';
import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js';

const _p = [0, 0, 0];
const _q = [0, 0, 0, 1];

export function createPhysicsSync(b3, world) {
  const group = new THREE.Group();
  const meshes = new Map(); // bodyKey -> mesh
  const seen = new Set();

  const shapeKey = (s) => `${s.index1}:${s.world0}:${s.generation}`;

  function geometryFor(shapeId) {
    const type = b3.b3Shape_GetType(shapeId).value;
    if (type === b3.b3ShapeType.b3_sphereShape.value) {
      const s = b3.b3Shape_GetSphere(shapeId);
      const g = new THREE.SphereGeometry(s.radius, 24, 16);
      g.translate(s.center[0], s.center[1], s.center[2]);
      return g;
    }
    if (type === b3.b3ShapeType.b3_capsuleShape.value) {
      const c = b3.b3Shape_GetCapsule(shapeId);
      const axis = new THREE.Vector3(c.center2[0] - c.center1[0], c.center2[1] - c.center1[1], c.center2[2] - c.center1[2]);
      const g = new THREE.CapsuleGeometry(c.radius, Math.max(axis.length(), 1e-4), 6, 16);
      g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis.clone().normalize()));
      g.translate((c.center1[0] + c.center2[0]) / 2, (c.center1[1] + c.center2[1]) / 2, (c.center1[2] + c.center2[2]) / 2);
      return g;
    }
    if (type === b3.b3ShapeType.b3_hullShape.value) {
      const flat = b3.b3Shape_GetHullVertices(shapeId);
      const pts = [];
      for (let i = 0; i < flat.length; i += 3) pts.push(new THREE.Vector3(flat[i], flat[i + 1], flat[i + 2]));
      return new ConvexGeometry(pts);
    }
    return null; // box handled via scale path; meshes/heightfields unsupported here
  }

  function ensureMesh(shapeId) {
    const key = shapeKey(shapeId);
    let m = meshes.get(key);
    if (m !== undefined) return m;
    const body = b3.b3Shape_GetBody(shapeId);
    const isStatic = b3.b3Body_GetType(body).value === b3.b3BodyType.b3_staticBody.value;

    // Note: Box3D has no box shape type — boxes are convex hulls (b3_hullShape),
    // so everything flows through geometryFor() below.
    {
      const geo = geometryFor(shapeId);
      if (!geo) return null;
      const color = isStatic ? 0x4a4f58 : PALETTE[(colorIdx++) % PALETTE.length];
      m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.08 }));
    }
    m.castShadow = true; m.receiveShadow = true;
    m.userData.body = body;
    meshes.set(key, m);
    group.add(m);
    return m;
  }
  let colorIdx = 0;
  const PALETTE = [0x7fb2ff, 0xffd93d, 0x6bcb77, 0xff8b6b, 0xc78bff, 0x22d3ee];

  const filter = b3.b3DefaultQueryFilter();
  const HUGE = [-1e9, -1e9, -1e9, 1e9, 1e9, 1e9];

  function update() {
    seen.clear();
    b3.b3World_OverlapAABB(world, HUGE, filter, (shapeId) => {
      const m = ensureMesh(shapeId);
      if (!m) return true;
      const key = shapeKey(shapeId);
      seen.add(key);
      const body = b3.b3Shape_GetBody(shapeId);
      b3.b3Body_GetPosition(_p, body);
      b3.b3Body_GetRotation(_q, body);
      m.position.set(_p[0], _p[1], _p[2]);
      m.quaternion.set(_q[0], _q[1], _q[2], _q[3]);
      return true;
    });
    for (const [key, m] of meshes) {
      if (!seen.has(key)) { group.remove(m); meshes.delete(key); m.geometry.dispose?.(); }
    }
  }

  return { object3d: group, update };
}

// ---------------------------------------------------------------- debug draw
export function createDebugDraw(b3, world, scene) {
  const state = {
    showAABBs: false, showContacts: false, showVelocities: false,
    showAxes: false, velScale: 0.15, maxVectors: 300,
  };

  // AABB wireframes
  const aabbGroup = new THREE.Group(); scene.add(aabbGroup);
  const aabbPool = [];
  const aabbGeo = new THREE.BoxGeometry(1, 1, 1);
  const aabbMat = new THREE.LineBasicMaterial({ color: 0xffd93d, transparent: true, opacity: 0.35 });
  function getAABBHelper(i) {
    while (aabbPool.length <= i) {
      const h = new THREE.Box3Helper(new THREE.Box3(), aabbMat.color);
      h.material = aabbMat;
      aabbGroup.add(h); aabbPool.push(h);
    }
    return aabbPool[i];
  }

  // contact points
  const contactGroup = new THREE.Group(); scene.add(contactGroup);
  const contactMat = new THREE.PointsMaterial({ color: 0xff5c5c, size: 0.09, sizeAttenuation: true });
  const MAX_CONTACT_PTS = 2048;
  const contactPos = new Float32Array(MAX_CONTACT_PTS * 3);
  const contactGeo = new THREE.BufferGeometry();
  contactGeo.setAttribute('position', new THREE.BufferAttribute(contactPos, 3));
  const contactPoints = new THREE.Points(contactGeo, contactMat);
  contactPoints.frustumCulled = false;
  contactGroup.add(contactPoints);

  // velocity arrows
  const velGroup = new THREE.Group(); scene.add(velGroup);

  // axes per dynamic body
  const axesGroup = new THREE.Group(); scene.add(axesGroup);
  const axesPool = [];
  function getAxis(i) {
    while (axesPool.length <= i) {
      const a = new THREE.AxesHelper(0.45);
      axesGroup.add(a); axesPool.push(a);
    }
    return axesPool[i];
  }

  // contacts buffer plumbing (per-shape gather, dedupe by contactId.index1 —
  // Box3D has no world-level contact enumeration; verified against .d.ts)
  const contactsBuffer = b3.createContactsBuffer();
  const contactOut = b3.createContact();
  const manifoldOut = b3.createManifold();
  const pointOut = b3.createPoint();
  const seenContacts = new Set();

  function gatherContacts() {
    let fn = 0;
    seenContacts.clear();
    b3.b3World_OverlapAABB(world, [-1e9, -1e9, -1e9, 1e9, 1e9, 1e9], filter, (shapeId) => {
      const buf = b3.getShapeContactData(contactsBuffer, shapeId);
      const n = b3.getNumContacts(buf);
      for (let c = 0; c < n; c++) {
        const contact = b3.getContactAt(contactOut, buf, c);
        const dedupe = String(contact.contactId.index1 ?? c) + ':' + String(contact.shapeIdA.index1);
        if (seenContacts.has(dedupe)) continue;
        seenContacts.add(dedupe);
        for (let m = 0; m < contact.manifoldCount; m++) {
          const manifold = b3.getManifoldAt(manifoldOut, contact, m);
          for (let pt = 0; pt < manifold.pointCount && fn < MAX_CONTACT_PTS; pt++) {
            const mp = manifold.points ? manifold.points[pt] : b3.getManifoldPointAt?.(pointOut, manifold, pt);
            const anchor = mp.anchorA ?? mp.point;
            contactPos[fn * 3] = anchor[0]; contactPos[fn * 3 + 1] = anchor[1]; contactPos[fn * 3 + 2] = anchor[2];
            fn++;
          }
        }
      }
      return true;
    });
    contactGeo.setDrawRange(0, fn);
    contactGeo.attributes.position.needsUpdate = true;
    contactTriCount = fn;
  }
  let contactTriCount = 0;

  const filter = b3.b3DefaultQueryFilter();

  function update() {
    // ---- AABBs ----
    aabbGroup.visible = state.showAABBs;
    if (state.showAABBs) {
      let i = 0;
      b3.b3World_OverlapAABB(world, [-1e9, -1e9, -1e9, 1e9, 1e9, 1e9], filter, (shapeId) => {
        const aabb = b3.b3Shape_GetAABB([0, 0, 0, 0, 0, 0], shapeId);
        const helper = getAABBHelper(i++);
        const box = helper.box;
        box.min.set(aabb[0], aabb[1], aabb[2]);
        box.max.set(aabb[3], aabb[4], aabb[5]);
        helper.updateMatrixWorld(true);
        return true;
      });
      for (let j = i; j < aabbPool.length; j++) aabbPool[j].visible = false;
      for (let j = 0; j < Math.min(i, aabbPool.length); j++) aabbPool[j].visible = true;
    }

    // ---- Contacts ----
    contactGroup.visible = state.showContacts;
    if (state.showContacts) gatherContacts();

    // ---- Velocities + Axes (iterate awake bodies via overlap of whole world) ----
    velGroup.visible = state.showVelocities || false;
    axesGroup.visible = state.showAxes;
    let vi = 0;
    if (state.showVelocities || state.showAxes) {
      // clear old arrows lazily
      while (velGroup.children.length) { const c = velGroup.children.pop(); velGroup.remove(c); }
      b3.b3World_OverlapAABB(world, [-1e9, -1e9, -1e9, 1e9, 1e9, 1e9], filter, (shapeId) => {
        const body = b3.b3Shape_GetBody(shapeId);
        if (b3.b3Body_GetType(body).value !== b3.b3BodyType.b3_dynamicBody.value) return true;
        if (!b3.b3Body_IsAwake(body)) return true;
        if (vi >= state.maxVectors) return false;
        b3.b3Body_GetPosition(_p, body);
        const v = b3.b3Body_GetLinearVelocity([0, 0, 0], body);
        if (state.showVelocities) {
          const len = Math.hypot(v[0], v[1], v[2]) * state.velScale;
          if (len > 0.02) {
            const dir = new THREE.Vector3(v[0], v[1], v[2]).normalize();
            const arrow = new THREE.ArrowHelper(dir, new THREE.Vector3(_p[0], _p[1], _p[2]), Math.min(len, 4), 0x56d364, 0.12, 0.07);
            velGroup.add(arrow);
          }
        }
        if (state.showAxes) {
          b3.b3Body_GetRotation(_q, body);
          const ax = getAxis(vi);
          ax.position.set(_p[0], _p[1], _p[2]);
          ax.quaternion.set(_q[0], _q[1], _q[2], _q[3]);
          ax.visible = true;
        }
        vi++;
        return true;
      });
      for (let j = vi; j < axesPool.length; j++) axesPool[j].visible = false;
    }
  }

  function addGui(folder) {
    const f = folder.addFolder('Debug Draw');
    f.add(state, 'showAABBs').name('AABB wireframes');
    f.add(state, 'showContacts').name('contact points');
    f.add(state, 'showVelocities').name('velocity vectors');
    f.add(state, 'showAxes').name('body local axes');
    f.add(state, 'velScale', 0.05, 1).name('vel vector scale');
    f.close();
    return f;
  }

  return { state, update, addGui };
}
