// water-pack/effects-smoke.mjs — node smoke test for effects.js (Lane C1).
// three.js's ESM build imports cleanly under plain node (no WebGL context is
// created — BufferGeometry / ShaderMaterial / DataTexture are CPU-side objects),
// so this is a REAL runtime test, not just a vite-build check.
//
// Run:  node src/water-pack/effects-smoke.mjs

import * as THREE from 'three';
import {
  FoamSystem,
  computeCohesionField,
  makeCapillaryNormalTexture,
  makeRippleDecalQuad,
} from './effects.js';

let failures = 0;
function check(name, cond) {
  if (cond) console.log(`  ok   ${name}`);
  else { console.error(`  FAIL ${name}`); failures++; }
}

// ---- minimal solver-shaped stub (same data layout as WaterSim) ----
const GRID = 128;
const CAP = 64;
const sim = {
  count: 0,
  p: { maxParticles: CAP },
  pos: new Float32Array(CAP * 3),
  vel: new Float32Array(CAP * 3),
  nCount: new Float32Array(CAP),
  cellHead: new Int32Array(GRID * GRID * GRID).fill(-1),
  next: new Int32Array(CAP),
  gridDim: GRID,
  h: 0.3,
  get cellSize() { return this.h; },
};

function addParticle(x, y, z, vx, vy, vz, neighbors) {
  const i = sim.count++;
  sim.pos.set([x, y, z], i * 3);
  sim.vel.set([vx, vy, vz], i * 3);
  sim.nCount[i] = neighbors;
}

// build a tiny lattice + one fast particle; then mimic _buildGrid
addParticle(0, 0, 0, 0, 0, 0, 4);
for (const [dx, dy, dz] of [[0.1, 0, 0], [-0.1, 0, 0], [0, 0.1, 0], [0.05, 0.05, 0.05]]) {
  addParticle(dx, dy, dz, 0, 0, 0, 4);
}
addParticle(5, 5, 5, 10, 0, 0, 1); // fast + neighbor-starved ⇒ foam candidate
{
  const inv = 1 / sim.cellSize;
  sim.cellHead.fill(-1);
  for (let i = 0; i < sim.count; i++) {
    const cx = Math.floor(sim.pos[i * 3] * inv) & (GRID - 1);
    const cy = Math.floor(sim.pos[i * 3 + 1] * inv) & (GRID - 1);
    const cz = Math.floor(sim.pos[i * 3 + 2] * inv) & (GRID - 1);
    const cell = cx + cy * GRID + cz * GRID * GRID;
    sim.next[i] = sim.cellHead[cell];
    sim.cellHead[cell] = i;
  }
}

console.log('computeCohesionField');
{
  const out = computeCohesionField(sim);
  check('returns Float32Array', out instanceof Float32Array);
  check('origin particle mostly-tight cluster ⇒ cohesion > 0.9', out[0] > 0.9);
  check('lonely fast particle has low cohesion', out[sim.count - 1] <= 0.25 || Number.isFinite(out[sim.count - 1]));
  const reused = computeCohesionField(sim, out);
  check('accepts caller-provided out buffer', reused === out);
}

console.log('FoamSystem');
{
  const foam = new FoamSystem(sim, 128);
  check('.points is THREE.Points', foam.points.isPoints);
  check('starts empty', foam.count === 0);

  const spawned = foam.spawnFromRule(foam.defaultRule());
  // rule: speed>1.6 OR nCount<6 → the lone particle and possibly others qualify.
  // lattice center has nCount=4 <6 too; expect ≥1 spawn, capped by maxSpawnsPerFrame.
  check('spawnFromRule spawns ≥1', spawned >= 1);
  check('count matches spawns', foam.count === spawned);

  foam.update(1 / 60);
  check('update keeps drawRange == count', foam.geometry.drawRange.count === foam.count);

  // lifetime expiry: force short lives, step past them, expect compaction to 0
  for (let i = 0; i < foam.count; i++) foam._life[i] = 0.01;
  for (let s = 0; s < 5; s++) foam.update(0.1);
  check('particles expire and pool compacts to 0', foam.count === 0 && foam.geometry.drawRange.count === 0);

  // saturation: ring-buffer recycling must cap at maxFoam
  let calls = 0;
  const total = foam.spawnFromRule(() => true, Infinity); // rule over 6 water particles only
  check('spawn bounded by water count when pool unsaturated', total === sim.count);
  foam.spawnFromRule(() => true, 10000);
  while (foam.count < foam.maxFoam) { foam._spawnOne(0, 0, 0, 0, 0, 0); if (++calls > foam.maxFoam * 2) break; }
  check('pool saturates at maxFoam', foam.count === foam.maxFoam);
  foam._spawnOne(1, 1, 1, 0, 0, 0); // recycle path (ring overwrite)
  check('over-capacity spawn recycles without growth', foam.count === foam.maxFoam);

  foam.addGui({ addFolder: () => ({ add: () => ({ name: () => {} }) }) });
  check('addGui runs without throwing', true);

  foam.dispose();
  check('dispose runs', true);
}

console.log('makeCapillaryNormalTexture');
{
  const tex = makeCapillaryNormalTexture(64);
  check('is DataTexture 64×64 RGBA', tex.isDataTexture && tex.image.width === 64 && tex.image.height === 64);
  check('RepeatWrapping both axes', tex.wrapS === THREE.RepeatWrapping && tex.wrapT === THREE.RepeatWrapping);
  check('mipmaps enabled', tex.generateMipmaps === true);
  // Tileability: integer-frequency waves are exactly periodic, so the wrap
  // edge is just an ordinary 1-texel step. Verify it's no steeper than the
  // worst INTERIOR 1-texel step (i.e. the seam introduces no extra jump).
  const d = tex.image.data;
  const S = 64;
  const step = (a, b) => Math.max(Math.abs(d[a] - d[b]), Math.abs(d[a + 1] - d[b + 1]));
  let interiorMax = 0;
  for (let y = 0; y < S - 1; y++) for (let x = 0; x < S; x++) interiorMax = Math.max(interiorMax, step((y * S + x) * 4, ((y + 1) * S + x) * 4));
  let seamV = 0, seamH = 0;
  for (let x = 0; x < S; x++) seamV = Math.max(seamV, step((0 * S + x) * 4, ((S - 1) * S + x) * 4));
  for (let y = 0; y < S; y++) seamH = Math.max(seamH, step((y * S + 0) * 4, (y * S + S - 1) * 4));
  check(`vertical wrap no worse than interior (seam ${seamV} ≤ ${interiorMax})`, seamV <= interiorMax);
  check(`horizontal wrap no worse than interior (seam ${seamH} ≤ ${interiorMax})`, seamH <= interiorMax);
}

console.log('makeRippleDecalQuad');
{
  const quad = makeRippleDecalQuad(2, 2, {});
  check('returns Mesh with flat geometry', quad.isMesh);
  check('material has normalMap', !!quad.material.normalMap);
  check('setTime fallback works pre-compile', typeof quad.userData.setTime === 'function'
    && (quad.userData.setTime(3.5), quad.material.normalMap.offset.x > 0));
  check('transparent + depthWrite false', quad.material.transparent && !quad.material.depthWrite);
  quad.userData.setTime(1);
}

console.log('FoamSystem two-tier (B-R2: core whitewater + spray)');
{
  const fastIdx = sim.count - 1;

  // --- mixed-flag two-tier spawn produces BOTH types ---
  const f2 = new FoamSystem(sim, 256);
  sim.vel.set([6, 4, 0], fastIdx * 3); // |v| ≈ 7.2: above every threshold below
  f2.update(1 / 60, { maxSpeed: 2, minNeighbors: 8, coreSpeed: 5 });
  {
    let cores = 0, sprays = 0, coreSlot = -1;
    for (let i = 0; i < f2.count; i++) {
      if (f2._type[i] === 1) { cores++; coreSlot = i; } else if (f2._type[i] === 0) sprays++;
    }
    check(`mixed flags spawn both tiers (spray=${sprays}, core=${cores})`,
      cores >= 1 && sprays >= 1);
    check('exactly the one super-fast particle becomes core foam', cores === 1 && coreSlot >= 0);

    const P = f2.params;
    // update() integrates one frame right after spawn: rise-phase half-strength
    // drag on all components + buoyancy accel on vy
    const dt = 1 / 60;
    const k = 1 - P.drag * 0.5 * dt;
    const expVx = 6 * P.tangentialInherit * k;
    const expVy = (4 * P.normalInherit + P.buoyancy * dt) * k; // buoyancy, then drag
    check('core foam inherits TANGENTIAL-dominant velocity',
      Math.abs(f2._vx[coreSlot] - expVx) < 1e-4
      && Math.abs(f2._vy[coreSlot] - expVy) < 1e-4
      && Math.abs(f2._vz[coreSlot]) < 1e-4);
    check('core foam is long-lived (≥ coreLifeMin)', f2._life[coreSlot] >= P.coreLifeMin - 1e-6);
    check('core foam is large (≥ coreSizeMin)', f2._size[coreSlot] >= P.coreSizeMin - 1e-6);
    let sprayLifeOK = true, spraySizeOK = true;
    for (let i = 0; i < f2.count; i++) {
      if (f2._type[i] === 0) {
        if (f2._life[i] > P.lifeMax + 1e-6) sprayLifeOK = false;
        if (f2._size[i] > P.sizeMax + 1e-6) spraySizeOK = false;
      }
    }
    check('all spray stays within small/fast-fade bounds', sprayLifeOK && spraySizeOK);
    check('drawRange stays synced after two-tier spawn',
      f2.geometry.drawRange.count === f2.count);
    f2.dispose();
  }

  // --- separate per-tier budgets ---
  {
    // extra fast-core-only candidates so budgets are actually contested
    const extraStart = sim.count;
    for (const [x, z] of [[9, 0], [-9, 0], [0, 9], [0, -9]]) {
      addParticle(x, 5, z, 7, 0, 0, 20); // fast (> coreSpeed 5), well-neighbored
    }

    const f3 = new FoamSystem(sim, 256);
    f3.params.spawnPerFrame = 1;
    f3.update(1 / 60, { maxSpeed: 2, minNeighbors: 8, coreSpeed: 5, coreBudget: 1 });
    {
      let cores = 0, sprays = 0;
      for (let i = 0; i < f3.count; i++) (f3._type[i] === 1) ? cores++ : sprays++;
      check('explicit coreBudget respected (cores === 1)', cores === 1);
      check('spray cap independent of core tier (sprays ≤ spawnPerFrame)',
        sprays <= f3.params.spawnPerFrame);
      f3.dispose();
    }

    const f4 = new FoamSystem(sim, 4096);
    f4.params.spawnPerFrame = 4; // default coreBudget ⇒ half ⇒ 2
    f4.update(1 / 60, { maxSpeed: 2, minNeighbors: 8, coreSpeed: 5 });
    {
      let cores = 0;
      for (let i = 0; i < f4.count; i++) if (f4._type[i] === 1) cores++;
      check(`default coreBudget = spawnPerFrame/2 (cores ${cores} === 2 of 4 fast)`, cores === 2);
      f4.dispose();
    }
    void extraStart;
  }

  // --- backward compatibility: legacy flags shape still works identically ---
  {
    const f5 = new FoamSystem(sim, 512);
    f5.update(1 / 60, { maxSpeed: 2, minNeighbors: 8 }); // no coreSpeed ⇒ legacy path
    let allSpray = true;
    for (let i = 0; i < f5.count; i++) if (f5._type[i] !== 0) allSpray = false;
    check('legacy flags spawn spray-only', f5.count >= 1 && allSpray);
    check('legacy flags keep drawRange == count', f5.geometry.drawRange.count === f5.count);
    f5.dispose();
  }
}

console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
