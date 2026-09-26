// test/solver.test.mjs — WaterSim regression tests (node --test).
//
// Each test pins a property that has actually broken before: the pair list
// (stale cell tables produced duplicate/dead pairs), viscosity (impulses were
// discarded by the v = Δx/dt derive), NaN containment (one NaN particle
// poisoned its neighbors), and the external particle-order contract that the
// renderers, foam and worker interpolation rely on.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { WaterSim, PHASES } from '../src/water-pack/solver.js';

const DT = 1 / 60;
const FLOOR = [{ type: 'plane', o: [0, 0, 0], n: [0, 1, 0] }];

function rng(seed) {
  return () => {
    seed = (seed * 16807) % 2147483647;
    return seed / 2147483647;
  };
}

function withSeed(seed, fn) {
  const orig = Math.random;
  Math.random = rng(seed);
  try { return fn(); } finally { Math.random = orig; }
}

// All unordered pairs within h, by brute force over the working copy.
function brutePairs(sim) {
  const sp = sim._sp, n = sim.count, h2 = sim.h * sim.h;
  const set = new Set();
  for (let a = 0; a < n; a++) {
    for (let b = a + 1; b < n; b++) {
      const dx = sp[b * 3] - sp[a * 3], dy = sp[b * 3 + 1] - sp[a * 3 + 1], dz = sp[b * 3 + 2] - sp[a * 3 + 2];
      const r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < h2 && r2 > 1e-12) set.add(a * 1e6 + b);
    }
  }
  return set;
}

function solverPairs(sim) {
  const list = [];
  for (let a = 0; a < sim.count; a++) {
    for (let t = sim._pairStart[a]; t < sim._pairStart[a + 1]; t++) {
      const b = sim._pairB[t];
      list.push(Math.min(a, b) * 1e6 + Math.max(a, b));
    }
  }
  return list;
}

test('pair list equals the brute-force set (no duplicates, no stale cells)', () => {
  withSeed(5, () => {
    const sim = new WaterSim({ h: 0.3, maxParticles: 4000 });
    sim.bounds = { min: [-20, -1, -20], max: [20, 20, 20] };
    // straddle the origin so negative cell coordinates (mask wrap) are covered
    sim.spawnBlock(0, 0.5, 0, 12, 10, 12, 0.05, 0);
    for (let s = 0; s < 90; s++) {
      // vacate cells between steps: stale per-cell tables used to be walked
      if (s % 15 === 7) sim.drain({ min: [-20, -1, -20], max: [20, 20, -1 + s / 60] });
      sim.step(DT, FLOOR);
      if (s % 10 !== 9) continue;
      // rebuild pairs on the current positions, then compare
      sim._sortByCell();
      sim._buildPairs(DT);
      const got = solverPairs(sim);
      const want = brutePairs(sim);
      assert.equal(new Set(got).size, got.length, `step ${s}: duplicate pairs`);
      assert.equal(got.length, want.size, `step ${s}: pair count`);
      for (const k of got) assert.ok(want.has(k), `step ${s}: spurious pair ${k}`);
    }
  });
});

test('densities and neighbor counts match the pair list', () => {
  withSeed(9, () => {
    const sim = new WaterSim({ h: 0.3, maxParticles: 3000 });
    sim.spawnBlock(0, 0.3, 0, 10, 10, 10);
    for (let s = 0; s < 20; s++) sim.step(DT, FLOOR);
    sim._sortByCell();
    sim._buildPairs(DT);
    const n = sim.count;
    const rho = new Float64Array(n), cnt = new Int32Array(n);
    for (let a = 0; a < n; a++) {
      for (let t = sim._pairStart[a]; t < sim._pairStart[a + 1]; t++) {
        const b = sim._pairB[t], q = sim._pairData[t * 4];
        rho[a] += q * q; rho[b] += q * q; cnt[a]++; cnt[b]++;
      }
    }
    for (let a = 0; a < n; a++) {
      assert.ok(Math.abs(rho[a] - sim.rho[a]) < 1e-4, `rho[${a}]`);
      assert.equal(cnt[a], sim._nc[a], `nCount[${a}]`);
    }
  });
});

test('nCount is reported in particle order', () => {
  // cubic lattice at 0.9h: only the 6 axis neighbors are within h, so each
  // particle's count is known from its lattice position (interior 6, face 5,
  // edge 4, corner 3); spawn order differs from the solver's cell order
  const sim = new WaterSim({ h: 0.3, maxParticles: 1000, gravity: 0 });
  const N = 7, s = 0.27;
  for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) for (let z = 0; z < N; z++) sim.spawn(x * s, y * s, z * s);
  sim.step(DT, []);
  let i = 0;
  for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) for (let z = 0; z < N; z++, i++) {
    const inner = (c) => (c > 0 ? 1 : 0) + (c < N - 1 ? 1 : 0);
    assert.equal(sim.nCount[i], inner(x) + inner(y) + inner(z), `particle ${i} (${x},${y},${z})`);
  }
});

// Free block (no gravity, no colliders) with random velocities.
function freeBlockKE(params) {
  return withSeed(7, () => {
    const sim = new WaterSim({ h: 0.3, maxParticles: 2000, gravity: 0, ...params });
    const r = Math.random, sp = 0.3 * 0.55;
    for (let x = 0; x < 10; x++) for (let y = 0; y < 10; y++) for (let z = 0; z < 10; z++) {
      sim.spawn(x * sp, y * sp, z * sp, (r() - 0.5) * 2, (r() - 0.5) * 2, (r() - 0.5) * 2);
    }
    for (let s = 0; s < 60; s++) sim.step(DT, []);
    return sim;
  });
}

test('viscosity damps relative motion (impulses survive the Δx/dt derive)', () => {
  const inviscid = freeBlockKE({ viscositySigma: 0, viscosityBeta: 0 }).kineticEnergy;
  const viscous = freeBlockKE({ viscositySigma: 4, viscosityBeta: 1 }).kineticEnergy;
  assert.ok(viscous < inviscid * 0.9, `KE viscous ${viscous.toFixed(1)} vs inviscid ${inviscid.toFixed(1)}`);
});

test('pair forces conserve linear momentum', () => {
  const sim = freeBlockKE({});
  const n = sim.count;
  const mom = () => {
    const m = [0, 0, 0];
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) m[k] += sim.vel[i * 3 + k];
    return m;
  };
  const before = mom();
  for (let s = 0; s < 30; s++) sim.step(DT, []);
  const after = mom();
  for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(after[k] - before[k]) < 1e-2 * n, `axis ${k}: ${before[k]} → ${after[k]}`);
  }
});

test('a NaN particle is quarantined without infecting its neighbors', () => {
  withSeed(3, () => {
    const sim = new WaterSim({ h: 0.3, maxParticles: 2000 });
    sim.spawnBlock(0, 0.3, 0, 10, 10, 10);
    for (let s = 0; s < 10; s++) sim.step(DT, FLOOR);
    const n0 = sim.count;
    sim.pos[500 * 3 + 1] = NaN;
    sim.vel[600 * 3] = Infinity;
    for (let s = 0; s < 5; s++) sim.step(DT, FLOOR);
    assert.equal(sim.count, n0 - 2);
    for (let k = 0; k < sim.count * 3; k++) {
      assert.ok(Number.isFinite(sim.pos[k]) && Number.isFinite(sim.vel[k]), `entry ${k}`);
    }
  });
});

test('particle order is stable across steps (renderers index by particle)', () => {
  withSeed(4, () => {
    const sim = new WaterSim({ h: 0.3, maxParticles: 3000 });
    sim.spawnBlock(0, 1, 0, 12, 12, 12, 0.02, 2);
    const prev = new Float32Array(sim.count * 3);
    for (let s = 0; s < 40; s++) {
      prev.set(sim.pos.subarray(0, sim.count * 3));
      sim.step(DT, FLOOR);
      const maxStep = sim.p.maxSpeed * DT + 1e-4;
      for (let i = 0; i < sim.count; i++) {
        const d = Math.hypot(sim.pos[i * 3] - prev[i * 3], sim.pos[i * 3 + 1] - prev[i * 3 + 1], sim.pos[i * 3 + 2] - prev[i * 3 + 2]);
        assert.ok(d <= maxStep, `step ${s}: particle ${i} jumped ${d.toFixed(3)} m`);
      }
    }
  });
});

test('spawn overlap guard also sees same-frame spawns after stepping', () => {
  const sim = new WaterSim({ h: 0.3, maxParticles: 100 });
  assert.equal(sim.spawn(0, 1, 0), true);
  sim.step(DT, FLOOR);
  const x = sim.pos[0], y = sim.pos[1], z = sim.pos[2];
  assert.equal(sim.spawn(x + 0.01, y, z), false, 'duplicate of stepped particle');
  assert.equal(sim.spawn(3, 1, 3), true);
  assert.equal(sim.spawn(3.01, 1, 3), false, 'duplicate of same-frame spawn');
  assert.equal(sim.spawn(NaN, 1, 3), false, 'non-finite spawn rejected');
  assert.equal(sim.count, 2);
});

test('drain returns the number removed; reset empties the sim', () => {
  const sim = new WaterSim({ h: 0.3, maxParticles: 1000 });
  const added = sim.spawnBlock(0, 0.2, 0, 6, 6, 6);
  assert.equal(added, 216);
  const removed = sim.drain({ min: [-10, -1, -10], max: [0, 10, 10] });
  assert.ok(removed > 0 && removed < 216);
  assert.equal(sim.count, 216 - removed);
  sim.reset();
  assert.equal(sim.count, 0);
  assert.equal(sim.spawn(0, 1, 0), true, 'spawn after reset');
});

test('deterministic for a fixed seed; phase timings are reported', () => {
  const run = () => withSeed(21, () => {
    const sim = new WaterSim({ h: 0.3, maxParticles: 3000 });
    sim.spawnBlock(0, 1, 0, 10, 10, 10, 0.02, 1);
    for (let s = 0; s < 30; s++) sim.step(DT, FLOOR);
    return sim;
  });
  const a = run(), b = run();
  assert.equal(a.count, b.count);
  assert.deepEqual(a.pos.subarray(0, a.count * 3), b.pos.subarray(0, b.count * 3));
  for (const k of PHASES) assert.ok(a.phaseMs[k] >= 0, `phase ${k}`);
  const sum = PHASES.reduce((s, k) => s + a.phaseMs[k], 0);
  assert.ok(Math.abs(sum - a.simMs) < 1, `phases sum ${sum} vs simMs ${a.simMs}`);
});

test('a poured column settles: stays in the tank and comes to rest', () => {
  withSeed(2, () => {
    const H = 1.5, T = 0.125;
    const tank = [
      ...FLOOR,
      { type: 'box', c: [-(H + T), 1, 0], e: [T, 1, H + 2 * T] }, { type: 'box', c: [H + T, 1, 0], e: [T, 1, H + 2 * T] },
      { type: 'box', c: [0, 1, -(H + T)], e: [H + 2 * T, 1, T] }, { type: 'box', c: [0, 1, H + T], e: [H + 2 * T, 1, T] },
    ];
    const sim = new WaterSim({ h: 0.3, maxParticles: 3000 });
    sim.bounds = { min: [-5, -1, -5], max: [5, 10, 5] };
    sim.spawnBlock(-0.5, 0.4, 0, 8, 10, 8);
    const n0 = sim.count;
    for (let s = 0; s < 60 * 6; s++) sim.step(DT, tank);
    assert.equal(sim.count, n0, 'no particles lost');
    let inside = 0;
    for (let i = 0; i < sim.count; i++) {
      if (Math.abs(sim.pos[i * 3]) < H + 0.1 && Math.abs(sim.pos[i * 3 + 2]) < H + 0.1) inside++;
    }
    assert.equal(inside, sim.count, 'all water inside the tank');
    const rms = Math.sqrt((2 * sim.kineticEnergy) / sim.count);
    assert.ok(rms < 0.15, `rms speed at rest ${rms.toFixed(3)} m/s`);
  });
});
