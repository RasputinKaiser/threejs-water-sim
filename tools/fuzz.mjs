#!/usr/bin/env node
// tools/fuzz.mjs — adversarial fuzz suite for the water solver.
//
// Each scenario runs a fresh inline simulation (spacing 0.15, 24k particle
// cap, bounds [-20,-3,-12]..[20,6,12], plane floor) and abuses one input path
// while invariants are checked every few steps:
//   1. every position/velocity entry is finite
//   2. 0 ≤ count ≤ maxParticles
//   3. no speed above maxSpeed · 1.01
//   4. kinetic energy never jumps by more than 100× in one check interval
//
// Run:  node tools/fuzz.mjs [--seed N] [--steps N]
// Output: one line per scenario with PASS / VIOLATIONS and a detail line.
// Exit code 0 when all scenarios pass, else 1.

import { createSimulation } from '../src/water/sim.js';

const args = process.argv.slice(2);
const argOf = (name, dflt) => {
  const i = args.indexOf(name);
  return i !== -1 && args[i + 1] !== undefined ? Number(args[i + 1]) : dflt;
};
const SEED = argOf('--seed', 1234);
const STEPS = argOf('--steps', 300);
const DT = 1 / 60;
const BOUNDS = { min: [-20, -3, -12], max: [20, 6, 12] };
const FLOOR = [{ type: 'plane', position: [0, 0, 0] }];

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
let rand = mulberry32(SEED);

async function freshSim(extra = {}) {
  const sim = await createSimulation({ spacing: 0.15, maxParticles: 24000, bounds: BOUNDS, ...extra }, { threads: 0 });
  sim.setColliders(FLOOR);
  return sim;
}
const block = (sim, c, half, o) => sim.fillBox([c[0] - half[0], c[1] - half[1], c[2] - half[2]], [c[0] + half[0], c[1] + half[1], c[2] + half[2]],
  { seed: Math.floor(rand() * 2 ** 31), ...o });
const randInBounds = () => [
  BOUNDS.min[0] + rand() * (BOUNDS.max[0] - BOUNDS.min[0]),
  rand() * BOUNDS.max[1],
  BOUNDS.min[2] + rand() * (BOUNDS.max[2] - BOUNDS.min[2]),
];

function checkInvariants(sim, ctx) {
  const out = [];
  const n = sim.count, pos = sim.positions, vel = sim.velocities;
  let bad = 0;
  for (let i = 0; i < n * 3; i++) {
    if (!Number.isFinite(pos[i]) || !Number.isFinite(vel[i])) { if (++bad <= 3) out.push(`non-finite state at entry ${i}`); }
  }
  if (!Number.isInteger(n) || n < 0 || n > sim.params.maxParticles) out.push(`count ${n} outside [0, ${sim.params.maxParticles}]`);
  const lim = sim.params.maxSpeed * 1.01, lim2 = lim * lim;
  let fast = 0;
  for (let i = 0; i < n; i++) {
    const vx = vel[i * 3], vy = vel[i * 3 + 1], vz = vel[i * 3 + 2];
    if (vx * vx + vy * vy + vz * vz > lim2 && ++fast <= 3) out.push(`particle ${i} faster than ${lim.toFixed(1)} m/s`);
  }
  const ke = sim.stats.kineticEnergy;
  if (!Number.isFinite(ke)) out.push(`kinetic energy ${ke}`);
  else if (ctx.lastKE > 1 && ke > ctx.lastKE * 100) out.push(`KE explosion ${ke.toFixed(1)} J > 100 × ${ctx.lastKE.toFixed(1)} J`);
  if (Number.isFinite(ke)) ctx.lastKE = ke;
  return out;
}

function run(sim, steps, every, before) {
  const violations = [];
  const ctx = { lastKE: 0 };
  for (let s = 0; s < steps; s++) {
    before?.(s, sim);
    sim.stepNow(1);
    if ((s + 1) % every === 0 || s === steps - 1) for (const v of checkInvariants(sim, ctx)) violations.push(`step ${s}: ${v}`);
  }
  return violations;
}

/* S1: NaN written into one particle every 30 steps. Quarantine must cost only
 * the poisoned particles — a NaN reaching the density sums would spread. */
async function s1NanInjection(steps) {
  const sim = await freshSim();
  const n0 = block(sim, [0, 1, 0], [0.75, 0.75, 0.75]);
  let injected = 0;
  const v = run(sim, steps, 10, (s) => {
    if (s >= 50 && s % 30 === 0) { sim.positions[Math.floor(rand() * sim.count) * 3] = NaN; injected++; }
  });
  const lost = n0 - sim.count - sim.stats.leaked;
  if (lost > injected) v.push(`NaN spread: lost ${lost} particles to ${injected} injections`);
  return { v, detail: `start=${n0} injections=${injected} quarantined=${sim.stats.quarantined} final=${sim.count}` };
}

/* S2: 10 random particles teleported anywhere in bounds every step. */
async function s2TeleportStorm(steps) {
  const sim = await freshSim();
  block(sim, [0, 0.6, 0], [1, 0.6, 1]);
  const v = run(sim, steps, 10, () => {
    for (let t = 0; t < 10 && sim.count; t++) {
      const i = Math.floor(rand() * sim.count), p = randInBounds();
      sim.positions[i * 3] = p[0]; sim.positions[i * 3 + 1] = p[1]; sim.positions[i * 3 + 2] = p[2];
    }
  });
  return { v, detail: `final=${sim.count} leaked=${sim.stats.leaked}` };
}

/* S3: the same block spawned twice on exactly the same lattice. Coincident
 * pairs have a zero kernel gradient, and the block starts at twice rest
 * density, so it bursts (upward — the floor must hold every particle; the
 * ones thrown past y = 6 leave the bounds and are counted as leaked). */
async function s3OverlapBomb(steps) {
  const sim = await freshSim();
  const a = block(sim, [0, 1, 0], [1, 0.7, 1], { jitter: 0 });
  const b = block(sim, [0, 1, 0], [1, 0.7, 1], { jitter: 0 });
  let minY = Infinity;
  const v = run(sim, steps, 10, () => {
    for (let i = 0; i < sim.count; i++) minY = Math.min(minY, sim.positions[i * 3 + 1]);
  });
  if (minY < -0.01) v.push(`particle below the floor (y ${minY.toFixed(3)})`);
  return { v, detail: `particles=${a}+${b} final=${sim.count} minY=${minY.toFixed(3)} leaked=${sim.stats.leaked}` };
}

/* S4: tunables thrown around every 30 steps within hostile ranges. */
async function s4ParamChaos(steps) {
  const sim = await freshSim();
  block(sim, [0, 0.6, 0], [1, 0.6, 1]);
  const RANGES = { viscosity: [0, 1], vorticity: [0, 2], cohesion: [0, 5], friction: [0, 1], iterations: [1, 12], sor: [0.2, 1.2] };
  const keys = Object.keys(RANGES), log = [];
  const v = run(sim, steps, 10, (s) => {
    if (s > 0 && s % 30 === 0) {
      const k = keys[Math.floor(rand() * keys.length)], [lo, hi] = RANGES[k];
      const val = k === 'iterations' ? Math.round(lo + rand() * (hi - lo)) : lo + rand() * (hi - lo);
      sim.setParams({ [k]: val });
      log.push(`${k}=${val.toFixed(2)}`);
    }
  });
  return { v, detail: log.join(' ') };
}

/* S5: 50 random spawns into a region that is drained the same frame. */
async function s5DrainSpawnRace(steps) {
  const sim = await freshSim();
  // the untouched water sits in its own tank so it cannot flow into the region
  // (four wall boxes: a container would make everything outside it solid)
  sim.setColliders([...FLOOR,
    { type: 'box', position: [6.9, 1, 0], size: [0.1, 1, 1.2] }, { type: 'box', position: [9.1, 1, 0], size: [0.1, 1, 1.2] },
    { type: 'box', position: [8, 1, -1.1], size: [1.2, 1, 0.1] }, { type: 'box', position: [8, 1, 1.1], size: [1.2, 1, 0.1] }]);
  const n0 = block(sim, [8, 0.6, 0], [1, 0.6, 1]);
  const R = { min: [-3, 0, -3], max: [3, 4, 3] };
  let spawned = 0, drained = 0;
  const v = run(sim, steps, 10, () => {
    const batch = new Float32Array(50 * 6);
    for (let t = 0; t < 50; t++) {
      batch[t * 6] = R.min[0] + rand() * 6; batch[t * 6 + 1] = rand() * 4; batch[t * 6 + 2] = R.min[2] + rand() * 6;
    }
    spawned += sim.spawn(batch);
    drained += sim.removeInBox(R.min, R.max);
  });
  if (drained !== spawned) v.push(`drained ${drained} of ${spawned} spawned into the region`);
  if (sim.count !== n0 - sim.stats.leaked) v.push(`the block outside the region lost particles (${sim.count} of ${n0})`);
  return { v, detail: `spawned=${spawned} drained=${drained} final=${sim.count}` };
}

/* S6: gravity flips between -9.81, 0, +9.81 and -98 every 40 steps. */
async function s6GravityFlips(steps) {
  const sim = await freshSim();
  block(sim, [0, 0.6, 0], [1, 0.6, 1]);
  const G = [-9.81, 0, 9.81, -98];
  const v = run(sim, steps, 10, (s) => { if (s % 40 === 0) sim.setParams({ gravity: [0, G[(s / 40) % G.length], 0] }); });
  return { v, detail: `final=${sim.count} leaked=${sim.stats.leaked}` };
}

/* S7: filled to capacity, then 1000 more spawns must all be rejected. */
async function s7MaxPressure(steps) {
  const sim = await freshSim({ maxParticles: 8000 });
  const cap = sim.params.maxParticles;
  for (let k = 0; k < 40 && sim.count < cap; k++) block(sim, [-15 + (k % 8) * 4, 0.8 + Math.floor(k / 8) * 1.6, 0], [1, 0.75, 1]);
  const filled = sim.count;
  let accepted = 0;
  for (let t = 0; t < 1000; t++) accepted += sim.spawn([randInBounds()]);
  const v = run(sim, Math.min(steps, 120), 10);
  if (sim.count > cap || accepted > cap - filled) v.unshift(`count exceeded maxParticles (accepted ${accepted} past ${filled}/${cap})`);
  return { v, detail: `filled=${filled}/${cap} acceptedOf1000=${accepted}` };
}

const SCENARIOS = [
  ['S1 nan-injection ', s1NanInjection],
  ['S2 teleport-storm', s2TeleportStorm],
  ['S3 overlap-bomb  ', s3OverlapBomb],
  ['S4 param-chaos   ', s4ParamChaos],
  ['S5 drain-spawn   ', s5DrainSpawnRace],
  ['S6 gravity-flips ', s6GravityFlips],
  ['S7 max-pressure  ', s7MaxPressure],
];

console.log(`=== water fuzz suite: seed=${SEED} steps=${STEPS}/scenario ===\n`);
let failed = 0;
for (const [name, fn] of SCENARIOS) {
  rand = mulberry32(SEED);
  const t0 = performance.now();
  let res;
  try { res = await fn(STEPS); } catch (err) {
    console.log(`${name} | CRASH           | ${String(err?.stack ?? err).split('\n').slice(0, 2).join(' ')}`);
    failed++;
    continue;
  }
  const status = res.v.length ? `VIOLATIONS(${res.v.length})` : 'PASS';
  if (res.v.length) failed++;
  console.log(`${name} | ${status.padEnd(15)} | ${res.v[0] ?? '-'}`);
  console.log(`${''.padEnd(18)}   ${res.detail} (${(performance.now() - t0).toFixed(0)} ms)`);
}
console.log(failed ? `\nFAIL: ${failed} scenario(s) with violations` : '\nALL SCENARIOS PASS');
process.exit(failed ? 1 : 0);
