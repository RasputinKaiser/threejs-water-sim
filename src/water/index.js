// water/index.js — drop-in physical water for three.js (+ optional Box3D).
//
//   import { createWater } from './water/index.js';
//
//   const water = await createWater({ renderer, scene, b3, world });
//   water.addCollider({ type: 'container', position: [0, 1, 0], size: [2, 1, 2] });
//   water.fillBox([-2, 0, -2], [2, 1, 2]);
//   water.addSource({ position: [0, 3, 0], direction: [0, -1, 0], radius: 0.2, speed: 3 });
//
//   // every frame
//   const simDt = water.update(dt);   // fluid time advanced (fixed steps; 0 while the
//   if (simDt > 0) b3.b3World_Step(world, simDt, 4);  // worker is still busy) — keep Box3D in lockstep
//   water.render(camera);             // instead of renderer.render(scene, camera)
//
// Physics: Position Based Fluids on a spatial hash grid (see core/), run on
// worker threads when the page is cross-origin isolated (COOP/COEP headers),
// otherwise on the main thread. Box3D bodies near the water become colliders
// every frame and receive buoyancy/drag back (see box3d.js).

import * as THREE from 'three';
import { vec3 } from 'math';
import { createSimulation, latticeBox } from './sim.js';
import { createBox3DCoupling } from './box3d.js';
import { createScreenSpaceRenderer } from './render/screen-space.js';

export { createSimulation } from './sim.js';
export { createBox3DCoupling } from './box3d.js';
export { createScreenSpaceRenderer } from './render/screen-space.js';
export { DEFAULTS as SOLVER_DEFAULTS } from './core/params.js';

// Resolution/cost presets. Particle count for a given volume scales with
// 1/spacing³; cost per particle rises with the solver accuracy (DFSPH: the
// density tolerance; PBF: iterations).
export const QUALITY = {
  low: { spacing: 0.15, maxParticles: 16384, densityTolerance: 0.005, iterations: 3 },
  medium: { spacing: 0.1, maxParticles: 32768, densityTolerance: 0.002, iterations: 4 },
  high: { spacing: 0.08, maxParticles: 65536, densityTolerance: 0.001, iterations: 6 },
};

/**
 * @param {object} o
 *   renderer, scene   three.js renderer and scene (renderer optional for headless use)
 *   b3, world         Box3D module + world id (optional; enables two-way coupling)
 *   quality           'low' | 'medium' | 'high' (default 'medium')
 *   params            solver params (core/params.js) overriding the preset
 *   colliders         static solver colliders (planes, containers, heightfields…)
 *   threads           'auto' | n | 0 (main thread)
 *   backend           'cpu' (default) | 'gpu' | 'auto' — WebGPU compute for the
 *                     DFSPH solver ('auto': WebGPU when available); no
 *                     whitewater on the GPU yet
 *   render            'screen' (default) | 'points' | false; look overrides in `look`
 *   workerFactory     custom Worker constructor (bundlers without module workers, tests)
 */
export async function createWater({
  renderer = null, scene = null, b3 = null, world = null,
  quality = 'medium', params = {}, colliders = [], threads = 'auto',
  render = 'screen', look = {}, workerFactory, fixedDt, maxStepsPerFrame, maxColliders, backend = 'cpu',
} = {}) {
  const preset = QUALITY[quality] ?? QUALITY.medium;
  const sim = await createSimulation({ ...preset, ...params }, {
    threads, workerFactory, fixedDt, maxStepsPerFrame, maxColliders, backend,
  });
  const dp = sim.params;
  let staticColliders = [...colliders];
  let collidersDirty = true;
  // colliders take effect immediately (spawns right after addCollider already
  // see them); with Box3D the coupling rebuilds the list every update
  function syncColliders() {
    if (coupling) coupling.setColliders(staticColliders);
    else { sim.setColliders(staticColliders); collidersDirty = false; }
  }
  const coupling = b3 && world ? createBox3DCoupling({ sim, b3, world, colliders: staticColliders }) : null;

  // ---- rendering ---------------------------------------------------------
  const interp = new Float32Array(dp.maxParticles * 3);
  let screen = null, points = null;
  if (renderer && render === 'screen') {
    screen = createScreenSpaceRenderer({
      renderer, capacity: dp.maxParticles, spacing: dp.spacing,
      particleRadius: dp.spacing * 0.62 * (dp.solver === 'dfsph' ? 1.8 : dp.kernelScale),
      diffuseCapacity: dp.solver === 'dfsph' ? dp.maxDiffuse : 0, look,
    });
  } else if (scene && render === 'points') {
    const geo = new THREE.BufferGeometry();
    const attr = new THREE.BufferAttribute(interp, 3);
    attr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('position', attr);
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Infinity);
    points = new THREE.Points(geo, new THREE.PointsMaterial({ color: 0x3aa0ff, size: dp.spacing, sizeAttenuation: true }));
    points.frustumCulled = false;
    scene.add(points);
  }

  // ---- sources & drains --------------------------------------------------
  const sources = new Set();
  const drains = new Set();
  const _u = vec3.create(), _v = vec3.create();

  function emitLayer(src, offset) {
    const s = dp.spacing, r = src.radius, dir = src.dir;
    vec3.perpendicular(_u, dir);
    vec3.cross(_v, dir, _u);
    const n = Math.ceil(r / s);
    const out = [];
    for (let a = -n; a <= n; a++) for (let b = -n; b <= n; b++) {
      const ua = (a + ((src.layer & 1) ? 0.5 : 0)) * s, vb = (b + ((src.layer & 2) ? 0.5 : 0)) * s;
      if (ua * ua + vb * vb > r * r) continue;
      out.push([
        src.position[0] + _u[0] * ua + _v[0] * vb + dir[0] * offset,
        src.position[1] + _u[1] * ua + _v[1] * vb + dir[1] * offset,
        src.position[2] + _u[2] * ua + _v[2] * vb + dir[2] * offset,
        dir[0] * src.speed, dir[1] * src.speed, dir[2] * src.speed,
      ]);
    }
    src.layer++;
    return out;
  }

  // Lattice points of a source that are already taken by water are skipped:
  // a source that is backed up (its outlet submerged in slow water) then
  // injects less instead of stacking particles inside the pressurised column.
  const occupied = new Map();
  const OCC = dp.spacing, OCC_R2 = (0.75 * dp.spacing) ** 2;
  const occKey = (x, y, z) => ((Math.floor(x / OCC) * 73856093) ^ (Math.floor(y / OCC) * 19349663) ^ (Math.floor(z / OCC) * 83492791)) | 0;
  function buildOccupancy(src) {
    occupied.clear();
    const p = sim.positions, n = sim.count, R = src.radius + 3 * dp.spacing;
    const R2 = R * R, [sx, sy, sz] = src.position;
    for (let i = 0; i < n; i++) {
      const x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
      if ((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2 > R2) continue;
      const k = occKey(x, y, z);
      let list = occupied.get(k);
      if (!list) occupied.set(k, list = []);
      list.push(x, y, z);
    }
  }
  function isOccupied(x, y, z) {
    const cx = Math.floor(x / OCC), cy = Math.floor(y / OCC), cz = Math.floor(z / OCC);
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
      const list = occupied.get(((cx + a) * 73856093 ^ (cy + b) * 19349663 ^ (cz + c) * 83492791) | 0);
      if (!list) continue;
      for (let k = 0; k < list.length; k += 3) {
        if ((list[k] - x) ** 2 + (list[k + 1] - y) ** 2 + (list[k + 2] - z) ** 2 < OCC_R2) return true;
      }
    }
    return false;
  }

  let spawnedTotal = 0;
  function runSources(dt) {
    const batch = [];
    for (const src of sources) {
      if (!src.enabled || src.speed <= 0) continue;
      src.travel += src.speed * dt;
      // one disk layer per `spacing` of travel: particles enter on the rest lattice
      let guard = 0, built = false;
      while (src.travel >= dp.spacing && guard++ < 8) {
        src.travel -= dp.spacing;
        if (!built) { buildOccupancy(src); built = true; }
        for (const p of emitLayer(src, src.travel)) if (!isOccupied(p[0], p[1], p[2])) batch.push(p);
      }
      if (guard >= 8) src.travel = 0;
    }
    if (batch.length) spawnedTotal += sim.spawn(batch);
  }

  // ---- surface height query (gameplay: floating logic, wading, audio) ------
  const COLUMN = 2 * dp.spacing;
  let heightMap = null, heightFrame = -1, frame = 0;
  function buildHeights() {
    heightMap = new Map();
    const p = sim.positions, n = sim.count;
    for (let i = 0; i < n; i++) {
      const k = `${Math.floor(p[i * 3] / COLUMN)},${Math.floor(p[i * 3 + 2] / COLUMN)}`;
      const y = p[i * 3 + 1];
      const h = heightMap.get(k);
      if (h === undefined || y > h) heightMap.set(k, y);
    }
    heightFrame = frame;
  }

  const water = {
    sim,
    coupling,
    get params() { return sim.params; },
    get mode() { return sim.mode; },
    get count() { return sim.count; },
    get stats() {
      return {
        ...sim.stats, stepMs: sim.stepMs, threads: sim.threads, colliders: coupling?.stats.colliders,
        spawned: spawnedTotal, whitewater: sim.diffuse?.count ?? 0,
      };
    },
    /** The debug THREE.Points (render: 'points'), else null. */
    points,

    setParams(patch) { sim.setParams(patch); },
    /** Fill an axis-aligned box with water at rest density. Returns particles added. */
    fillBox(min, max, opts) { return sim.fillBox(min, max, opts); },
    /** Spawn explicit particles: Float32Array [x,y,z,vx,vy,vz]* or [[x,y,z,vx?,vy?,vz?], …]. */
    spawn(particles) { return sim.spawn(particles); },
    removeInBox(min, max) { return sim.removeInBox(min, max); },
    reset() { sim.reset(); },

    /** Static solver collider (plane, box, sphere, capsule, container, heightfield). */
    addCollider(desc) {
      staticColliders.push(desc);
      syncColliders();
      return desc;
    },
    removeCollider(desc) {
      staticColliders = staticColliders.filter((c) => c !== desc);
      syncColliders();
    },
    /** Register terrain; returns a collider you can pass to addCollider. */
    addHeightfield({ minX, minZ, dx, dz, nx, nz, heights, friction }) {
      const index = sim.addHeightfield({ minX, minZ, dx, dz, nx, nz, heights });
      return water.addCollider({ type: 'heightfield', heightfield: index, friction });
    },
    /**
     * A nozzle: disks of radius `radius` leave `position` along `direction` at
     * `speed` m/s (flow = π·radius²·speed m³/s). Returns a handle; set
     * `enabled`, `speed`, `position`, `direction` on it at any time.
     */
    addSource({ position, direction = [0, -1, 0], radius = 0.2, speed = 2, enabled = true }) {
      const dir = vec3.normalize(vec3.create(), direction);
      const src = { position: [...position], dir, radius, speed, enabled, travel: 0, layer: 0 };
      const handle = {
        get enabled() { return src.enabled; }, set enabled(v) { src.enabled = !!v; },
        get speed() { return src.speed; }, set speed(v) { src.speed = v; },
        get radius() { return src.radius; }, set radius(v) { src.radius = v; },
        set position(p) { src.position = [...p]; },
        set direction(d) { vec3.normalize(src.dir, d); },
        remove() { sources.delete(src); },
      };
      sources.add(src);
      return handle;
    },
    /** Remove water entering an axis-aligned box every frame (outflows, plugholes). */
    addDrain({ min, max }) {
      const d = { min: [...min], max: [...max], enabled: true };
      drains.add(d);
      return { get enabled() { return d.enabled; }, set enabled(v) { d.enabled = !!v; }, remove() { drains.delete(d); } };
    },
    /** Highest water particle top near (x, z), or -Infinity where there is none. */
    surfaceHeight(x, z) {
      if (heightFrame !== frame) buildHeights();
      const h = heightMap.get(`${Math.floor(x / COLUMN)},${Math.floor(z / COLUMN)}`);
      return h === undefined ? -Infinity : h + 0.5 * dp.spacing;
    },

    /**
     * Advance the water by dt seconds of real time. Returns the fluid time
     * actually simulated (a multiple of the fixed step; 0 when the worker is
     * still busy with the previous batch or dt is below one step). Step
     * coupled rigid bodies by exactly this amount: when the fluid cannot keep
     * up with real time it runs in slow motion, and bodies advancing in real
     * time would jump into water that has not moved yet.
     */
    update(dt) {
      frame++;
      if (coupling) coupling.update();
      else if (collidersDirty) { sim.setColliders(staticColliders); collidersDirty = false; }
      runSources(dt);
      for (const d of drains) if (d.enabled) sim.removeInBox(d.min, d.max);
      const steps = sim.update(dt);
      if (screen || points) {
        const n = sim.count;
        sim.interpolate(interp, sim.alpha);
        if (screen) {
          screen.setParticles(interp, n);
          const ww = sim.diffuse;
          if (ww) screen.setDiffuse(ww.data, ww.count);
        }
        if (points) {
          points.geometry.attributes.position.needsUpdate = true;
          points.geometry.setDrawRange(0, n);
        }
      }
      return steps * sim.fixedDt;
    },
    /** Render the scene with water (replaces renderer.render(scene, camera)). */
    render(camera, target = null) {
      if (screen) screen.render(scene, camera, target);
      else if (renderer) { renderer.setRenderTarget(target); renderer.render(scene, camera); }
    },
    /** Rendering look (absorption, scatter, roughness…); see render/screen-space.js. */
    get look() { return screen?.look ?? null; },
    setLook(patch) { screen?.setLook(patch); },
    dispose() {
      sim.dispose();
      screen?.dispose();
      if (points) { scene.remove(points); points.geometry.dispose(); points.material.dispose(); }
    },
  };
  return water;
}

export { latticeBox };
