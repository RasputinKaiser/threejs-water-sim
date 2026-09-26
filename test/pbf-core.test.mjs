// test/pbf-core.test.mjs — physical and structural checks for the PBF core
// (src/water/core). Each test pins a property the pack promises.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PBFSolver, H, U } from '../src/water/core/solver.js';
import { writeCollider, colliderSDF, packHeightfield, SHAPE } from '../src/water/core/colliders.js';
import { deriveParams, wallFraction } from '../src/water/core/params.js';

const DT = 1 / 60;

function block(s, min, max, v = [0, 0, 0]) {
  const sp = s.p.spacing;
  for (let x = min[0] + sp / 2; x < max[0]; x += sp)
    for (let y = min[1] + sp / 2; y < max[1]; y += sp)
      for (let z = min[2] + sp / 2; z < max[2]; z += sp) s.addParticle(x, y, z, ...v);
}

function setColliders(s, list) {
  list.forEach((c, i) => writeCollider(s.colliders, i, c));
  s.header[H.colliders] = list.length;
}

test('derived constants: rest density matches the emitter lattice; wall fraction is 1/2 at contact', () => {
  const dp = deriveParams({ spacing: 0.1 });
  // continuous limit of a lattice density is 1/s³ (poly6 integrates to 1)
  assert.ok(Math.abs(dp.rho0 * 0.001 - 1) < 0.08, `rho0·s³ = ${dp.rho0 * 0.001}`);
  assert.ok(Math.abs(wallFraction(0, dp.h) - 0.5) < 1e-12);
  assert.equal(wallFraction(dp.h, dp.h), 0);
  assert.equal(wallFraction(-dp.h, dp.h), 1);
});

test('neighbor lists equal the brute-force set, including hash-bucket collisions', () => {
  // tiny table (capacity 64 → 128 buckets) with particles spread over many
  // cells guarantees different cells share buckets
  const s = new PBFSolver({ spacing: 0.1, maxParticles: 400, gravity: [0, 0, 0] });
  let seed = 3;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  for (let i = 0; i < 400; i++) s.addParticle(rnd() * 1.2 - 0.6, rnd() * 1.2, rnd() * 1.2 - 0.6);
  s.u[U.dt] = DT;
  const n = s.count;
  s.phasePredict(0, n); s.phaseSort(); s.phaseGather(0, n); s.phaseScatterBack(0, n);
  s.phaseNeighbors(0, n, 0);
  const h2 = s.p.h * s.p.h;
  for (let i = 0; i < n; i++) {
    const want = new Set();
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dx = s.pos[i * 3] - s.pos[j * 3], dy = s.pos[i * 3 + 1] - s.pos[j * 3 + 1], dz = s.pos[i * 3 + 2] - s.pos[j * 3 + 2];
      if (dx * dx + dy * dy + dz * dz < h2) want.add(j);
    }
    const got = Array.from(s.nbr.subarray(i * s.M, i * s.M + s.nbrCount[i]));
    assert.equal(new Set(got).size, got.length, `duplicates for ${i}`);
    assert.deepEqual(got.sort((a, b) => a - b), [...want].sort((a, b) => a - b), `particle ${i}`);
  }
});

test('incompressible: a settled column keeps its volume and rests flat', () => {
  const s = new PBFSolver({ spacing: 0.1, maxParticles: 4000 });
  setColliders(s, [{ type: 'container', position: [0, 1, 0], size: [0.6, 1, 0.6] }]);
  block(s, [-0.6, 0, -0.6], [0.6, 1.2, 0.6]); // 12×12×12 = 1728 particles
  const n0 = s.count;
  for (let f = 0; f < 240; f++) s.step(DT);
  assert.equal(s.count, n0);
  // volume n·s³ over a 1.2×1.2 floor → 1.2 m column; mean height = depth/2
  let mean = 0, top = -Infinity;
  for (let i = 0; i < n0; i++) { mean += s.pos[i * 3 + 1]; top = Math.max(top, s.pos[i * 3 + 1]); }
  mean /= n0;
  assert.ok(Math.abs(mean - 0.6) / 0.6 < 0.04, `mean height ${mean.toFixed(3)} (ideal 0.6)`);
  assert.ok(top < 1.2, `top ${top.toFixed(3)}`);
  const rms = Math.sqrt(2 * s.u[U.kineticEnergy] / s.p.particleMass / n0);
  assert.ok(rms < 0.05, `resting rms speed ${rms.toFixed(3)} m/s`);
  // 99th-percentile density error (the 4 container-corner particles see only
  // their nearest wall's density and read ~12%; everything else is < 10%)
  const errs = Array.from(s.density.subarray(0, n0), (d) => d * s.p.invRho0 - 1).sort((x, y) => y - x);
  assert.ok(errs[Math.floor(n0 * 0.01)] < 0.1, `p99 density error ${errs[Math.floor(n0 * 0.01)]}`);
});

test('pressure forces conserve linear momentum', () => {
  const s = new PBFSolver({ spacing: 0.1, maxParticles: 2000, gravity: [0, 0, 0], vorticity: 0, viscosity: 0 });
  let seed = 11;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };
  for (let x = 0; x < 10; x++) for (let y = 0; y < 10; y++) for (let z = 0; z < 10; z++) {
    s.addParticle(x * 0.09, y * 0.09, z * 0.09, rnd(), rnd(), rnd()); // compressed → pressure acts
  }
  const mom = () => {
    const m = [0, 0, 0];
    for (let i = 0; i < s.count; i++) for (let k = 0; k < 3; k++) m[k] += s.vel[i * 3 + k];
    return m;
  };
  const before = mom();
  for (let f = 0; f < 30; f++) s.step(DT);
  const after = mom();
  for (let k = 0; k < 3; k++) assert.ok(Math.abs(after[k] - before[k]) < 1e-3 * s.count, `axis ${k}: ${before[k]} → ${after[k]}`);
});

test('no particle ends up inside a solid (box, rotated box, sphere, capsule)', () => {
  const s = new PBFSolver({ spacing: 0.1, maxParticles: 4000 });
  const cols = [
    { type: 'plane', position: [0, 0, 0] },
    { type: 'box', position: [0.6, 0.3, 0], size: [0.3, 0.3, 0.3], rotation: [0, 0.3826834, 0, 0.9238795] },
    { type: 'sphere', position: [-0.5, 0.4, 0.3], radius: 0.3 },
    { type: 'capsule', position: [0, 0.3, -0.6], radius: 0.15, halfHeight: 0.3, rotation: [0.7071068, 0, 0, 0.7071068] },
  ];
  setColliders(s, cols);
  block(s, [-1, 1.2, -1], [1, 2.2, 1]);
  for (let f = 0; f < 150; f++) s.step(DT);
  const n = new Float64Array(3);
  for (let i = 0; i < s.count; i++) {
    for (let c = 0; c < cols.length; c++) {
      const d = colliderSDF(s.colliders, c, s.heightfields, s.pos[i * 3], s.pos[i * 3 + 1], s.pos[i * 3 + 2], n);
      assert.ok(d > -0.01, `particle ${i} is ${(-d).toFixed(3)} m inside collider ${c}`);
    }
  }
});

test('water spawned into solids is discarded; overlaps resolve without launching it', () => {
  // no gravity: water at rest density stays put unless overlap resolution kicks it
  const s = new PBFSolver({ spacing: 0.1, maxParticles: 4000, gravity: [0, 0, 0] });
  const cols = [
    { type: 'plane', position: [0, 0, 0] },
    { type: 'box', position: [0, 0.3, 0], size: [0.3, 0.3, 0.3] },
  ];
  setColliders(s, cols);
  // a block straddling the box, plus a sheet 5 mm above the floor (inside the contact skin)
  block(s, [-0.6, 0, -0.6], [0.6, 0.8, 0.6]);
  const n = new Float64Array(3);
  for (let i = 0; i < s.count; i++) {
    const d = colliderSDF(s.colliders, 1, s.heightfields, s.pos[i * 3], s.pos[i * 3 + 1], s.pos[i * 3 + 2], n);
    assert.ok(d >= 0, `spawned ${(-d).toFixed(3)} m inside the box`);
  }
  for (let x = -0.95; x < 0.95; x += 0.1) for (let z = -0.95; z < 0.95; z += 0.1) {
    if (Math.max(Math.abs(x), Math.abs(z)) > 0.65) s.addParticle(x, 0.005, z);
  }
  let vmax = 0;
  for (let f = 0; f < 30; f++) {
    s.step(DT);
    for (let i = 0; i < s.count; i++) vmax = Math.max(vmax, Math.hypot(s.vel[i * 3], s.vel[i * 3 + 1], s.vel[i * 3 + 2]));
  }
  // pushing the sheet out of the skin as a collision would give Δx/dt ≈ 2.7 m/s;
  // depenetration leaves at most slop/dt (0.75 m/s) for the contact response
  assert.ok(vmax < 1, `max speed ${vmax.toFixed(2)} m/s`);
});

test('a body created inside water pushes it aside without an explosion', () => {
  const s = new PBFSolver({ spacing: 0.1, maxParticles: 4000, gravity: [0, 0, 0] });
  block(s, [-0.6, 0, -0.6], [0.6, 1, 0.6]);
  // the sphere appears after the fill (no spawn filtering), moving down at 1 m/s
  setColliders(s, [
    { type: 'plane', position: [0, 0, 0] },
    { type: 'sphere', position: [0, 0.5, 0], radius: 0.2, velocity: [0, -1, 0], dynamic: true, slot: 0, mass: 50 },
  ]);
  let vmax = 0;
  for (let f = 0; f < 10; f++) {
    s.step(DT);
    for (let i = 0; i < s.count; i++) vmax = Math.max(vmax, Math.hypot(s.vel[i * 3], s.vel[i * 3 + 1], s.vel[i * 3 + 2]));
  }
  // its 34 L of displaced water must flow aside (≈ 3.4 m/s here); resolving the
  // overlap as a collision and as wall pressure threw particles at ~9 m/s
  assert.ok(vmax < 5, `max speed ${vmax.toFixed(2)} m/s`);
});

test('rotated box SDF: distance and normal agree with sampling', () => {
  const rec = new Float32Array(24);
  writeCollider(rec, 0, { type: 'box', position: [1, 2, 3], size: [0.5, 0.2, 0.3], rotation: [0.1, 0.3, 0.2, 0.9273618] });
  const n = new Float64Array(3), m = new Float64Array(3);
  let seed = 5;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };
  for (let k = 0; k < 200; k++) {
    const x = 1 + rnd() * 2, y = 2 + rnd() * 2, z = 3 + rnd() * 2;
    const d = colliderSDF(rec, 0, [], x, y, z, n);
    const e = 1e-4; // normal = gradient of the SDF
    const gx = (colliderSDF(rec, 0, [], x + e, y, z, m) - colliderSDF(rec, 0, [], x - e, y, z, m)) / (2 * e);
    const gy = (colliderSDF(rec, 0, [], x, y + e, z, m) - colliderSDF(rec, 0, [], x, y - e, z, m)) / (2 * e);
    const gz = (colliderSDF(rec, 0, [], x, y, z + e, m) - colliderSDF(rec, 0, [], x, y, z - e, m)) / (2 * e);
    if (Math.abs(Math.hypot(gx, gy, gz) - 1) > 0.01) continue; // skip medial-axis kinks
    assert.ok(Math.hypot(gx - n[0], gy - n[1], gz - n[2]) < 0.02, `normal at sample ${k}`);
    assert.ok(Number.isFinite(d));
  }
});

test('water weight is transmitted to a dynamic container (coupling impulse = m·g·dt)', () => {
  const s = new PBFSolver({ spacing: 0.1, maxParticles: 2000 });
  setColliders(s, [{ type: 'container', position: [0, 0.5, 0], size: [0.4, 0.5, 0.4], dynamic: true }]);
  block(s, [-0.4, 0, -0.4], [0.4, 0.6, 0.4]); // 8×6×8 = 384 particles
  for (let f = 0; f < 180; f++) { s.step(DT); s.reduceImpulses(new Float64Array(s.maxColliders * 6)); }
  const out = new Float64Array(s.maxColliders * 6);
  let jy = 0;
  const F = 30;
  for (let f = 0; f < F; f++) { s.step(DT); s.reduceImpulses(out); jy += out[1]; }
  const weight = s.count * s.p.particleMass * 9.81;           // N
  const force = jy / (F * DT);                                 // N, on the container
  assert.ok(Math.abs(-force - weight) / weight < 0.1, `container load ${(-force).toFixed(2)} N vs weight ${weight.toFixed(2)} N`);
});

test('heightfield terrain holds water above the ground', () => {
  const nx = 21, nz = 21, heights = new Float32Array(nx * nz);
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) heights[z * nx + x] = 0.02 * ((x - 10) ** 2 + (z - 10) ** 2) * 0.1;
  const hf = packHeightfield({ minX: -1, minZ: -1, dx: 0.1, dz: 0.1, nx, nz, heights });
  const s = new PBFSolver({ spacing: 0.1, maxParticles: 2000 }, null, { heightfields: [hf] });
  setColliders(s, [{ type: SHAPE.heightfield, heightfield: 0 }]);
  block(s, [-0.4, 0.5, -0.4], [0.4, 1.0, 0.4]);
  const n0 = s.count;
  for (let f = 0; f < 150; f++) s.step(DT);
  assert.equal(s.count, n0);
  const n = new Float64Array(3);
  for (let i = 0; i < s.count; i++) {
    const d = colliderSDF(s.colliders, 0, s.heightfields, s.pos[i * 3], s.pos[i * 3 + 1], s.pos[i * 3 + 2], n);
    assert.ok(d > -0.01, `particle ${i} below terrain by ${-d}`);
  }
});

test('non-finite particles are quarantined; leaving the bounds removes particles', () => {
  const s = new PBFSolver({ spacing: 0.1, maxParticles: 2000, bounds: { min: [-2, -1, -2], max: [2, 5, 2] } });
  setColliders(s, [{ type: 'plane', position: [0, 0, 0] }]);
  block(s, [-0.3, 0, -0.3], [0.3, 0.6, 0.3]);
  const n0 = s.count;
  s.step(DT);
  s.pos[10 * 3] = NaN;
  s.vel[20 * 3 + 1] = Infinity;
  s.addParticle(1.9, 0.5, 0, 50, 0, 0); // flies out of the bounds
  s.step(DT); s.step(DT);
  assert.equal(s.header[H.quarantined], 2);
  assert.equal(s.header[H.leaked], 1);
  assert.equal(s.count, n0 - 2);
  for (let k = 0; k < s.count * 3; k++) assert.ok(Number.isFinite(s.pos[k]));
});

test('deterministic; ids travel with particles through the sort', () => {
  const run = () => {
    const s = new PBFSolver({ spacing: 0.1, maxParticles: 2000 });
    setColliders(s, [{ type: 'container', position: [0, 1, 0], size: [0.5, 1, 0.5] }]);
    block(s, [-0.5, 0, -0.5], [0, 0.8, 0.5]);
    for (let f = 0; f < 60; f++) s.step(DT);
    return s;
  };
  const a = run(), b = run();
  assert.deepEqual(a.pos.subarray(0, a.count * 3), b.pos.subarray(0, b.count * 3));
  const ids = new Set(a.id.subarray(0, a.count));
  assert.equal(ids.size, a.count, 'ids unique');
  // framePrev (interpolation origin) is permuted with the particles: it must
  // stay within one frame's travel of the current position
  a.markFrame();
  a.step(DT);
  for (let i = 0; i < a.count; i++) {
    const d = Math.hypot(a.pos[i * 3] - a.framePrev[i * 3], a.pos[i * 3 + 1] - a.framePrev[i * 3 + 1], a.pos[i * 3 + 2] - a.framePrev[i * 3 + 2]);
    assert.ok(d <= a.p.maxSpeed * DT + 1e-4, `particle ${i} framePrev off by ${d}`);
  }
});
