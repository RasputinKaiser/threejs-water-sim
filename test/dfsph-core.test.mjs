// test/dfsph-core.test.mjs — physical checks for the DFSPH solver
// (src/water/core/dfsph.js) and its lattice-calibrated volume-map boundaries.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DFSPHSolver } from '../src/water/core/dfsph.js';
import { H, U } from '../src/water/core/fluid-core.js';
import { writeCollider, colliderSDF } from '../src/water/core/colliders.js';
import { deriveParams, cubicW } from '../src/water/core/params.js';
import { createSimulation } from '../src/water/sim.js';

const DT = 1 / 60;

function lattice(s, min, max, v = [0, 0, 0]) {
  const sp = s.p.spacing;
  for (let x = min[0] + sp / 2; x < max[0]; x += sp)
    for (let y = min[1] + sp / 2; y < max[1]; y += sp)
      for (let z = min[2] + sp / 2; z < max[2]; z += sp) s.addParticle(x, y, z, ...v);
}
function setColliders(s, list) {
  list.forEach((c, i) => writeCollider(s.colliders, i, c));
  s.header[H.colliders] = list.length;
}
const rms = (s) => Math.sqrt(2 * s.u[U.kineticEnergy] / s.p.particleMass / s.count);

test('lattice calibration: interior and wall particles read exactly ρ0', () => {
  const dp = deriveParams({ solver: 'dfsph', spacing: 0.1 });
  const s = dp.spacing, h = dp.h;
  // a particle half a spacing from a wall: the fluid half-lattice plus ρ0Ψ(s/2)
  let fluid = 0;
  for (let a = -3; a <= 3; a++) for (let b = -3; b <= 3; b++) for (let c = 0; c <= 3; c++) fluid += cubicW(s * Math.hypot(a, b, c), h);
  const f = 0.5 * s * dp.bndInv, i = Math.floor(f), t = f - i;
  const psi = dp.bndF[i] + (dp.bndF[i + 1] - dp.bndF[i]) * t;
  assert.ok(Math.abs((fluid + dp.rho0 * psi) / dp.rho0 - 1) < 1e-4, `wall particle density ${(fluid + dp.rho0 * psi) / dp.rho0}`);
  assert.ok(Math.abs(dp.bndF[0] - 0.5) < 0.01, `Ψ(0) = ${dp.bndF[0]} (half the kernel)`);
});

test('a resting column settles calm, keeps its volume and holds the density tolerance', () => {
  const s = new DFSPHSolver({ spacing: 0.1, maxParticles: 4000 });
  setColliders(s, [{ type: 'container', position: [0, 1, 0], size: [0.6, 1, 0.6] }]);
  lattice(s, [-0.6, 0, -0.6], [0.6, 1.2, 0.6]);
  const n0 = s.count;
  for (let f = 0; f < 240; f++) s.step(DT);
  assert.equal(s.count, n0);
  let mean = 0;
  for (let i = 0; i < n0; i++) mean += s.pos[i * 3 + 1];
  mean /= n0;
  // the emission lattice repacks by a few % as it settles (as with PBF)
  assert.ok(Math.abs(mean - 0.6) / 0.6 < 0.045, `mean height ${mean.toFixed(3)} (lattice 0.6)`);
  assert.ok(rms(s) < 0.12, `resting rms speed ${rms(s).toFixed(3)} m/s`);
  assert.ok(s.u[U.avgDensityError] < 0.003, `mean density error ${(s.u[U.avgDensityError] * 100).toFixed(2)}%`);
});

test('pressure conserves momentum and dissipates energy (no gravity, no walls)', () => {
  const s = new DFSPHSolver({ spacing: 0.1, maxParticles: 3000, gravity: [0, 0, 0], viscosity: 0, vorticity: 0 });
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5;
  for (let x = -0.55; x < 0.6; x += 0.1) for (let y = -0.55; y < 0.6; y += 0.1) for (let z = -0.55; z < 0.6; z += 0.1) {
    s.addParticle(x, y, z, rnd() * 0.3, rnd() * 0.3, rnd() * 0.3);
  }
  const mom = () => { const m = [0, 0, 0]; for (let i = 0; i < s.count; i++) for (let k = 0; k < 3; k++) m[k] += s.vel[i * 3 + k]; return m; };
  const m0 = mom();
  s.step(DT);
  const e1 = s.u[U.kineticEnergy];
  for (let f = 0; f < 60; f++) s.step(DT);
  const m1 = mom();
  for (let k = 0; k < 3; k++) assert.ok(Math.abs(m1[k] - m0[k]) < 1e-3, `axis ${k}: ${m0[k]} → ${m1[k]}`);
  assert.ok(s.u[U.kineticEnergy] <= e1 * 1.001, `kinetic energy ${e1} → ${s.u[U.kineticEnergy]}`);
});

test('Archimedes: a fixed submerged sphere and box feel ρ·g·V', async () => {
  for (const [desc, V] of [
    [{ type: 'sphere', position: [0, 0.6, 0], radius: 0.25 }, (4 / 3) * Math.PI * 0.25 ** 3],
    [{ type: 'box', position: [0, 0.6, 0], size: [0.2, 0.2, 0.2] }, 0.4 ** 3],
  ]) {
    const sim = await createSimulation({ solver: 'dfsph', spacing: 0.1, maxParticles: 8000 }, { threads: 0 });
    sim.setColliders([{ type: 'container', position: [0, 0.75, 0], size: [0.75, 0.75, 0.75] }, { ...desc, dynamic: true, slot: 0, mass: 1e9 }]);
    sim.fillBox([-0.75, 0, -0.75], [0.75, 1.2, 0.75], { jitter: 0 });
    sim.stepNow(90);
    sim.takeImpulses();
    sim.stepNow(90);
    const { impulses, time } = sim.takeImpulses();
    const ratio = impulses[1] / time / (1000 * 9.81 * V);
    assert.ok(Math.abs(ratio - 1) < 0.1, `${desc.type}: buoyancy ${ratio.toFixed(3)} × ρgV`);
  }
});

test('no particle ends up inside a solid (box, rotated box, sphere, capsule, heightfield)', () => {
  const s = new DFSPHSolver({ spacing: 0.1, maxParticles: 4000 });
  const cols = [
    { type: 'plane', position: [0, 0, 0] },
    { type: 'box', position: [0.6, 0.3, 0], size: [0.3, 0.3, 0.3], rotation: [0, 0.3826834, 0, 0.9238795] },
    { type: 'sphere', position: [-0.5, 0.4, 0.3], radius: 0.3 },
    { type: 'capsule', position: [0, 0.3, -0.6], radius: 0.15, halfHeight: 0.3, rotation: [0.7071068, 0, 0, 0.7071068] },
  ];
  setColliders(s, cols);
  lattice(s, [-1, 1.2, -1], [1, 2.2, 1]);
  for (let f = 0; f < 150; f++) s.step(DT);
  const n = new Float64Array(3);
  for (let i = 0; i < s.count; i++) {
    for (let c = 0; c < cols.length; c++) {
      const d = colliderSDF(s.colliders, c, s.heightfields, s.pos[i * 3], s.pos[i * 3 + 1], s.pos[i * 3 + 2], n);
      assert.ok(d > -0.01, `particle ${i} is ${(-d).toFixed(3)} m inside collider ${c}`);
    }
  }
});

test('wall drag: a sheet sliding on a floor slows with the drag coefficient', () => {
  const speed = (cf) => {
    const s = new DFSPHSolver({ spacing: 0.1, maxParticles: 3000, friction: cf, viscosity: 0.03 });
    setColliders(s, [{ type: 'plane', position: [0, 0, 0] }]);
    lattice(s, [-1.5, 0, -1.5], [1.5, 0.2, 1.5], [1, 0, 0]);
    for (let f = 0; f < 60; f++) s.step(DT);
    let u = 0;
    for (let i = 0; i < s.count; i++) u += s.vel[i * 3];
    return u / s.count;
  };
  const u0 = speed(0), u1 = speed(0.01), u2 = speed(0.05);
  assert.ok(u0 > 0.95, `frictionless sheet keeps its speed (${u0.toFixed(3)} m/s)`);
  assert.ok(u0 > u1 && u1 > u2, `speeds after 1 s: C_f 0 → ${u0.toFixed(3)}, 0.01 → ${u1.toFixed(3)}, 0.05 → ${u2.toFixed(3)}`);
  // two layers, bottom one in contact: u(1 s) ≈ u0 / (1 + C_f·u0·t/(2s)) roughly
  assert.ok(u1 > 0.7 && u1 < 0.99, `C_f 0.01 after 1 s: ${u1.toFixed(3)} m/s`);
});

test('water spawned on top of water decompresses without exploding', () => {
  const s = new DFSPHSolver({ spacing: 0.1, maxParticles: 4000 });
  setColliders(s, [{ type: 'container', position: [0, 1, 0], size: [0.6, 1, 0.6] }]);
  lattice(s, [-0.6, 0, -0.6], [0.6, 0.6, 0.6]);
  lattice(s, [-0.6, 0, -0.6], [0.6, 0.6, 0.6]); // the same block again
  let vmax = 0;
  for (let f = 0; f < 60; f++) {
    s.step(DT);
    vmax = Math.max(vmax, s.u[U.maxVel]);
  }
  assert.ok(vmax < 8, `max speed while decompressing ${vmax.toFixed(2)} m/s`);
});
