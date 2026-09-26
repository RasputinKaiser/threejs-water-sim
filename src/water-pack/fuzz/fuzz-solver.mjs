#!/usr/bin/env node
// water-pack/fuzz/fuzz-solver.mjs — property-based fuzz suite for WaterSim
// (Lane A). Seven adversarial scenarios, each on a fresh sim
// {h:0.35, maxParticles:26000}, bounds [-20,-3,-12]..[20,6,12], plane floor.
//
// Run:  node src/water-pack/fuzz/fuzz-solver.mjs [--seed N] [--steps N]
// Default seed 1234, default 300 steps/scenario. Math.random is seeded for
// full reproducibility (spawnBlock jitter + scenario randomness).
//
// Output: table of scenario | status | first violation | repro seed.
// Exit code 0 if all scenarios pass, else 1.

import { WaterSim } from '../solver.js';
import { checkInvariants, runScenario, FUZZ_DT, FLOOR } from './fuzz-harness.mjs';

// ---------- CLI ----------
const args = process.argv.slice(2);
function argOf(name, dflt) {
  const i = args.indexOf(name);
  return i !== -1 && args[i + 1] !== undefined ? Number(args[i + 1]) : dflt;
}
const SEED = argOf('--seed', 1234);
const STEPS = argOf('--steps', 300);

// ---------- deterministic RNG: seed Math.random itself so spawnBlock's
// jitter and every scenario's random picks reproduce exactly ----------
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
Math.random = mulberry32(SEED);

const BOUNDS = { min: [-20, -3, -12], max: [20, 6, 12] };

function freshSim() {
  const sim = new WaterSim({ h: 0.35, maxParticles: 26000 });
  sim.bounds = BOUNDS;
  return sim;
}

// random point uniformly inside bounds (y kept above the floor plane y=0 so
// teleports/spawns land in physically reachable space unless a scenario
// deliberately goes below)
function randInBounds(belowFloor = false) {
  const mn = BOUNDS.min, mx = BOUNDS.max;
  return [
    mn[0] + Math.random() * (mx[0] - mn[0]),
    belowFloor ? -1 + Math.random() * 3 : Math.random() * (mx[1] - mn[1]),
    mn[2] + Math.random() * (mx[2] - mn[2]),
  ];
}

// ================= S1: NaN INJECTION =================
// Spawn a 10x10x10 block at center, run 50 clean steps, then every 30th step
// write NaN into one random particle's position component BEFORE stepping.
// Question: does the solver recover/quarantine, or does NaN persist/propagate?
function s1NanInjection(steps) {
  const sim = freshSim();
  sim.spawnBlock(0, 1, 0, 10, 10, 10); // 1000 particles
  const finiteCounts = [];
  const injectAt = [];
  const viols = runScenario(sim, steps, 10, (s, sim2) => {
    if (s >= 50 && s % 30 === 0) {
      const i = Math.floor(Math.random() * sim2.count);
      sim2.pos[i * 3] = NaN; // one component of one particle
      injectAt.push({ step: s, particle: i });
    }
  });
  // count finite particles over time is sampled by the checker; here we take
  // a final census plus per-injection survival
  let finitePos = 0;
  for (let k = 0; k < sim.count * 3; k++) if (Number.isFinite(sim.pos[k])) finitePos++;
  const detail = `injections=${injectAt.length} finalCount=${sim.count} nonFinitePosEntries=${sim.count * 3 - finitePos}`;
  // Containment: quarantine must cost only the injected particles. A NaN
  // that reaches the pair walk spreads through densities to its neighbors
  // (this once wiped out ~95% of the block while still reading as "finite").
  const lost = 1000 - sim.count - (sim.leakedTotal ?? 0);
  if (lost > injectAt.length) viols.push(`NaN spread: lost ${lost} particles to ${injectAt.length} injections`);
  return { violations: viols, detail };
}

// ================= S2: TELEPORT STORM =================
// Each frame teleport 10 random particles to random positions in bounds via
// direct pos writes before step.
function s2TeleportStorm(steps) {
  const sim = freshSim();
  sim.spawnBlock(0, 1, 0, 14, 8, 14);
  const viols = runScenario(sim, steps, 10, (s, sim2) => {
    for (let t = 0; t < 10 && sim2.count > 0; t++) {
      const i = Math.floor(Math.random() * sim2.count);
      const [x, y, z] = randInBounds();
      sim2.pos[i * 3] = x; sim2.pos[i * 3 + 1] = y; sim2.pos[i * 3 + 2] = z;
    }
  });
  return { violations: viols, detail: `finalCount=${sim.count} leakedTotal=${sim.leakedTotal ?? 0}` };
}

// ================= S3: OVERLAP BOMB =================
// spawnBlock twice at IDENTICAL coordinates — exercises the overlap-guard
// rejection path in spawn(). A short warm-up first ensures cellHead is
// populated (pre-first-step the guard's grid walk is empty — that hole is
// covered by a dedicated probe below).
function s3OverlapBomb(steps) {
  const sim = freshSim();
  sim.spawnBlock(0, 1, 0, 16, 10, 16);
  for (let s = 0; s < 5; s++) sim.step(FUZZ_DT, FLOOR); // settle + build grid
  const afterFirst = sim.count;
  sim.spawnBlock(0, 1, 0, 16, 10, 16);
  const afterSecond = sim.count;
  const viols = runScenario(sim, steps, 25);
  return {
    violations: viols,
    detail: `block=${afterFirst} secondPassAdded=${afterSecond - afterFirst} (guard rejected ${100 - Math.round((afterSecond - afterFirst) / afterFirst * 100)}% of duplicates)`,
  };
}

// regression probe for the pre-first-step overlap-guard hole (fixed via lazy
// _buildGrid() in spawn()): a duplicate block before any step must now be
// rejected just like post-warm-up duplicates
function s3bPreStepDuplicateProbe() {
  const sim = freshSim();
  sim.spawnBlock(0, 1, 0, 16, 10, 16);
  const n1 = sim.count;
  sim.spawnBlock(0, 1, 0, 16, 10, 16);
  const added = sim.count - n1;
  // tiny jitter means near-perfect duplicates; allow a small residue (<1%) but
  // a FULL second block means the guard no-op'd
  if (added > n1 * 0.01) {
    return [`pre-step duplicate spawnBlock accepted ${added}/${n1} particles (overlap guard inert before first step)`];
  }
  return [];
}

// ================= S4: PARAM CHAOS =================
// Every 30 steps mutate one parameter randomly within hostile ranges —
// directly into sim.p, mid-run.
function s4ParamChaos(steps) {
  const sim = freshSim();
  sim.spawnBlock(0, 1, 0, 14, 8, 14);
  const mutations = [];
  const RANGES = {
    h: [0.2, 0.7],
    restDensity: [1.5, 6],
    stiffness: [5, 40],
    viscositySigma: [5, 80],
  };
  const keys = Object.keys(RANGES);
  const viols = runScenario(sim, steps, 10, (s, sim2) => {
    if (s > 0 && s % 30 === 0) {
      const key = keys[Math.floor(Math.random() * keys.length)];
      const [lo, hi] = RANGES[key];
      const val = lo + Math.random() * (hi - lo);
      sim2.p[key] = val;
      mutations.push(`step=${s}:${key}=${val.toFixed(2)}`);
    }
  });
  return { violations: viols, detail: `mutations=[${mutations.join(', ')}]` };
}

// ================= S5: DRAIN+SPAWN RACE =================
// Spawn ~50/frame into region A while draining region A the same frame.
const REGION_A = { min: [-3, 0, -3], max: [3, 4, 3] };
function s5DrainSpawnRace(steps) {
  const sim = freshSim();
  sim.spawnBlock(0, 1, 0, 12, 8, 12);
  let spawned = 0, drainedSeen = 0;
  const viols = runScenario(sim, steps, 10, (s, sim2) => {
    // spawn up to 50 candidates inside region A (overlap guard may reject some)
    for (let t = 0; t < 50; t++) {
      const x = REGION_A.min[0] + Math.random() * (REGION_A.max[0] - REGION_A.min[0]);
      const y = REGION_A.min[1] + Math.random() * (REGION_A.max[1] - REGION_A.min[1]);
      const z = REGION_A.min[2] + Math.random() * (REGION_A.max[2] - REGION_A.min[2]);
      const before = sim2.count;
      sim2.spawn(x, y, z);
      spawned += sim2.count - before;
    }
    const before = sim2.count;
    sim2.drain(REGION_A);
    drainedSeen += before - sim2.count;
  });
  return { violations: viols, detail: `spawned=${spawned} drained=${drainedSeen} finalCount=${sim.count}` };
}

// ================= S6: GRAVITY FLIPS =================
// Gravity toggles between -9.81, 0, +9.81, -98 every 40 steps.
function s6GravityFlips(steps) {
  const sim = freshSim();
  sim.spawnBlock(0, 1, 0, 14, 8, 14);
  const G = [-9.81, 0, 9.81, -98];
  const viols = runScenario(sim, steps, 10, (s, sim2) => {
    sim2.p.gravity = G[Math.floor(s / 40) % G.length];
  });
  return { violations: viols, detail: `finalGravity=${sim.p.gravity} finalCount=${sim.count}` };
}

// ================= S7: MAX PRESSURE =================
// Fill to exactly maxParticles via repeated spawnBlock, then attempt 1000
// more spawns (reject path), continue stepping at capacity.
function s7MaxPressure(steps) {
  const sim = freshSim();
  let fills = 0;
  while (sim.count < sim.p.maxParticles && fills < 200) {
    sim.spawnBlock(
      -15 + (fills % 8) * 4 + Math.random(), 1 + Math.floor(fills / 8) * 0.5, -8 + Math.random() * 16,
      20, 20, 20,
    );
    fills++;
    if (fills > 60 && sim.count === sim.p.maxParticles) break;
  }
  const filledTo = sim.count;
  // rejection path: 1000 more attempts must not raise the count
  let rejected = 0;
  for (let t = 0; t < 1000; t++) {
    const [x, y, z] = randInBounds();
    const before = sim.count;
    sim.spawn(x, y, z);
    if (sim.count === before) rejected++;
  }
  const overfilled = sim.count > sim.p.maxParticles;
  const viols = runScenario(sim, steps, 10);
  return {
    violations: overfilled ? ['count EXCEEDED maxParticles during reject-path test', ...viols] : viols,
    detail: `filledTo=${filledTo}/${sim.p.maxParticles} blocksUsed=${fills} rejectedOf1000=${rejected}`,
  };
}

// ---------- runner ----------
const SCENARIOS = [
  ['S1 nan-injection ', s1NanInjection],
  ['S2 teleport-storm', s2TeleportStorm],
  ['S3 overlap-bomb  ', s3OverlapBomb],
  ['S4 param-chaos   ', s4ParamChaos],
  ['S5 drain-spawn   ', s5DrainSpawnRace],
  ['S6 gravity-flips ', s6GravityFlips],
  ['S7 max-pressure  ', s7MaxPressure],
];

console.log(`=== water-pack fuzz suite: seed=${SEED} steps=${STEPS}/scenario ===\n`);
let failed = 0;
for (const [name, fn] of SCENARIOS) {
  Math.random = mulberry32(SEED); // identical RNG stream per scenario
  const t0 = performance.now();
  let res;
  try {
    res = fn(STEPS);
  } catch (err) {
    console.log(`${name} | CRASH     | ${String(err).slice(0, 90)} | seed=${SEED}`);
    failed++;
    continue;
  }
  const ms = (performance.now() - t0).toFixed(0);
  const first = res.violations[0]?.replace(/\s+/g, ' ').slice(0, 72) || '-';
  const status = res.violations.length ? `VIOLATIONS(${res.violations.length})` : 'PASS';
  if (res.violations.length) failed++;
  console.log(`${name} | ${status.padEnd(15)} | ${first} | seed=${SEED}`);
  console.log(`${''.padEnd(18)} detail: ${res.detail} (${ms} ms)`);
}
// S3b: standalone regression probe (no stepping — pure spawn-path check)
Math.random = mulberry32(SEED);
{
  const s3b = s3bPreStepDuplicateProbe();
  if (s3b.length) {
    failed++;
    console.log(`S3b prestep-dup    | VIOLATIONS(${s3b.length})     | ${s3b[0]} | seed=${SEED}`);
  } else {
    console.log(`S3b prestep-dup    | PASS            | - | seed=${SEED}`);
  }
}
console.log(failed ? `\nFAIL: ${failed} scenario(s) with violations` : '\nALL SCENARIOS PASS');
process.exit(failed ? 1 : 0);
