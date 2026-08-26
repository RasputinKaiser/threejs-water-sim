// water-pack/sim-worker.mjs — Web Worker that owns a WaterSim instance off the
// render thread (perf-headroom.md, Priority 1 / option "e").
//
// Message protocol (main → worker, postMessage):
//   {type:'init', params, maxParticles, bounds?, sab:{ctl,pos,vel,nCount}}
//       Creates `new WaterSim({...params, maxParticles})`. The four
//       SharedArrayBuffers are allocated by async-sim.js and passed in the
//       message (structured clone handles SABs; no transfer list needed).
//       Replies {type:'ready', maxParticles} and sets ctl[3]=1 when done.
//   {type:'colliders', colliders}  — replaces the collider set (plain
//       serializable objects: plane/box/heightfield; heightfield heights may be
//       a TypedArray — structured clone copies it).
//   {type:'step', dt, steps?, frameId} — REAL-TIME PACED (see below): the
//       worker decides how many solver passes of `dt` to run so SIM TIME
//       tracks wall-clock elapsed since its last completed batch, clamped to
//       [1, maxStepsPerBatch] (anti spiral-of-death). Send `fixedSteps:true`
//       to force exactly `steps` passes (tests / deterministic replay).
//       Then syncs state into the SABs (double-buffered pos flip) and replies
//       {type:'frame', frameId, count, simMs}.
//   {type:'spawn', x,y,z,vx,vy,vz}
//   {type:'spawnBlock', cx,cy,cz,nx,ny,nz,jitter,v0}
//   {type:'drain', region:{min:[x,y,z], max:[x,y,z]}}
//   {type:'reset'}
//   {type:'setParams', patch}      — Object.assign'd onto sim.p
//   {type:'setBounds', bounds}     — bounds-leak kill region ({min,max})
//   {type:'dispose'}               — self-terminate
//
// Worker → main:
//   {type:'ready', maxParticles}
//   {type:'frame', frameId, count, simMs, leakedTotal, ke}   (simMs = worker-side step time)
//   {type:'debug', kind:'phases', batches, meanStepMs, perPhaseMs,
//    pairsPerStep, dominant}                                 (every 30 batches)
//
// BUFFERING SCHEME: DOUBLE-BUFFERED positions + Atomics frame counter.
// The worker owns plain WaterSim arrays and, after each step batch, copies
// pos/vel/nCount into the SABs (~0.2 ms at 26k particles). Positions live in
// TWO halves of one SAB (prev + curr): the worker always writes the INACTIVE
// half, then flips ctl[4] (the active/curr index) before bumping ctl[0].
// Renderers lerp(prev, curr, alpha) — see async-sim.js `.posPrev` / `.alpha` /
// `.posLerp()` — for buttery motion even when sim cadence < render cadence.
// vel/nCount stay single-buffered (renderers use latest-value semantics).
// Tear note: the copy window is still ~0.2 ms and a reader may observe a
// partially-written PREV half during it; at render cadence this is invisible
// (same accepted risk as v1). Only the worker writes, so the flip is a plain
// Atomics.store on ctl[4] (JS Atomics are sequentially consistent).
//
// REAL-TIME PACING (step batches): instead of a fixed steps=N per message,
// the worker measures wall-clock elapsed since its last COMPLETED batch and
// runs ceil(elapsed/dt) solver passes (debt carry-over: un-simulated leftover
// time is naturally included in the next measurement), clamped to
// [1, maxStepsPerBatch] (default 3; init `maxStepsPerBatch` / per-message
// `maxSteps` override) so a slow machine degrades to slow-motion instead of
// spiraling. When messages arrive back-to-back (worker saturated), elapsed≈0
// → exactly 1 step/message = graceful degradation. After a render hitch the
// next batch catches up (up to maxSteps). `fixedSteps:true` restores exact-N.
//
// ADAPTIVE BATCH SIZE (Lane F2): init `maxBatchMs` (default 12) caps the
// WALL-CLOCK budget of a step batch: after the pacing computation above, the
// worker further clamps the step count so the batch fits inside maxBatchMs,
// estimated from an EMA of measured per-step solve time (min 1 step — never
// zero). Fast machines push more steps per batch (higher sim rate); slow ones
// shrink batches so each frame lands sooner (lower interpolation latency,
// steadier posLerp cadence). Still bounded by the pacing debt + maxSteps
// clamp, so sim time never outruns wall clock.
//
// PHASE PROFILING (Lane F2): the worker monkey-patches WaterSim.prototype's
// phase methods (_buildGrid/_buildPairs/_viscosity/_relax/_collide) with
// timed wrappers and brackets the instance `step` — solver.js itself is NOT
// edited. Per-step phase split: gravity/predict = step entry → _buildGrid
// entry; derive(+diagnostics) = _collide exit → step exit. Timings
// accumulate across batches; every 30 completed batches the worker posts
//   {type:'debug', kind:'phases', batches, meanStepMs, perPhaseMs:{...},
//    pairsPerStep, dominant}
// and resets its accumulators. async-sim.js stores the latest report on
// `.phaseStats` / `.dominantPhase` for HUDs.
//
// Control SAB layout (Int32Array):
//   [0] frameId of last completed step batch (Atomics.store/load)
//   [1] current particle count
//   [2] simMs of the last step batch, stored as microseconds (int)
//   [3] ready flag (1 after init)
//   [4] active (curr) POSITION buffer index: 0=first half, 1=second half.
//       prev is the OTHER half. Flipped by the worker after each step batch.
//
// Environment note: this module uses ONLY browser worker globals
// (self/postMessage/addEventListener). For the headless Node test harness it
// runs inside a worker_threads Worker whose bootstrap shims those globals onto
// parentPort — see /tmp/test-async.mjs. No Node imports here, so vite bundles
// it cleanly for the browser.

import { WaterSim } from './solver.js';

let sim = null;
let ctl = null;
let posViews = null; // [half0, half1] over the (2× capacity) position SAB
let velView = null;
let ncView = null;
let colliders = [];
// Real-time pacing state + double-buffer flip index (worker is the only writer).
let activeBuf = 0;        // mirrors ctl[4]
let lastBatchEnd = 0;     // performance.now() of the last completed step batch
let maxStepsPerBatch = 3;
let maxBatchMs = 12;      // Lane F2: wall-clock budget per step batch (0 disables)
let emaStepMs = 0;        // EMA of measured per-step solve time (adaptive budget)

function post(msg) { postMessage(msg); }

// ---- Lane F2: phase profiling (worker-side monkey-patch; solver.js untouched) ----
const PHASE_REPORT_EVERY = 30;
const PHASE_KEYS = ['gravity', 'grid', 'pairs', 'viscosity', 'relax', 'collide', 'derive'];
// prototype method → accumulator key (called in solver.step order)
const PHASE_METHODS = [
  ['_buildGrid', 'grid'],
  ['_buildPairs', 'pairs'],
  ['_viscosity', 'viscosity'],
  ['_relax', 'relax'],
  ['_collide', 'collide'],
];
let prof = null; // accumulator object once installed

function installProfiler(sim) {
  const proto = Object.getPrototypeOf(sim);
  const acc = {
    steps: 0, totalMs: 0, npairs: 0,
    gravity: 0, grid: 0, pairs: 0, viscosity: 0, relax: 0, collide: 0, derive: 0,
  };
  let cursor = 0; // timestamp where the currently-running phase began

  for (const [method, key] of PHASE_METHODS) {
    const orig = proto[method];
    proto[method] = function (...args) {
      const enter = performance.now();
      if (key === 'grid') {
        // gravity/predict ran between the step bracket's t0 (or the previous
        // phase exit) and _buildGrid entry — charge that window to `gravity`.
        acc.gravity += enter - cursor;
      }
      const out = orig.apply(this, args);
      const exit = performance.now();
      acc[key] += exit - enter;
      cursor = exit;
      return out;
    };
  }

  const origStep = sim.step;
  sim.step = function (dt, cols) {
    const t0 = performance.now();
    cursor = t0;
    try { return origStep.call(this, dt, cols); }
    finally {
      const tEnd = performance.now();
      // derive (+ leak/KE diagnostics) runs after _collide exits.
      acc.derive += tEnd - cursor;
      acc.totalMs += tEnd - t0;
      acc.steps++;
      acc.npairs += this._npairs || 0;
      // feed the adaptive-budget EMA with the real single-step solve time
      emaStepMs = emaStepMs > 0 ? emaStepMs * 0.9 + (tEnd - t0) * 0.1 : tEnd - t0;
      maybePostPhases();
    }
  };

  prof = acc;
}

function maybePostPhases() {
  if (!prof || prof.steps < PHASE_REPORT_EVERY) return;
  const n = prof.steps;
  const perPhaseMs = {};
  let dominant = null;
  for (const k of PHASE_KEYS) {
    const v = prof[k] / n;
    perPhaseMs[k] = v;
    if (dominant === null || v > perPhaseMs[dominant]) dominant = k;
  }
  post({
    type: 'debug', kind: 'phases',
    batches: n,
    meanStepMs: prof.totalMs / n,
    perPhaseMs,
    pairsPerStep: Math.round(prof.npairs / n),
    dominant,
  });
  for (const k of PHASE_KEYS) prof[k] = 0;
  prof.steps = 0; prof.totalMs = 0; prof.npairs = 0;
}

/**
 * Copy the live sim state (up to count) into the shared buffers.
 * flip=true  → write the INACTIVE pos half, then flip ctl[4] to publish it as
 *              curr (step batches; prev keeps the previous frame for lerp).
 * flip=false → write the ACTIVE half in place (spawn/drain/reset: a state
 *              teleport that must NOT be interpolated from old positions).
 */
function syncStateToSAB(flip = false) {
  if (!sim || !ctl) return;
  const dst = flip ? 1 - activeBuf : activeBuf;
  posViews[dst].set(sim.pos.subarray(0, sim.count * 3));
  velView.set(sim.vel.subarray(0, sim.count * 3));
  ncView.set(sim.nCount.subarray(0, sim.count));
  Atomics.store(ctl, 1, sim.count);
  if (flip) {
    activeBuf = dst;
    Atomics.store(ctl, 4, activeBuf);
  }
}

self.addEventListener('message', (e) => {
  const msg = e.data;
  switch (msg?.type) {
    case 'init': {
      const params = { ...msg.params };
      if (msg.maxParticles != null) params.maxParticles = msg.maxParticles;
      sim = new WaterSim(params);
      ctl = new Int32Array(msg.sab.ctl);
      const cap3 = sim.p.maxParticles * 3;
      // Position SAB holds TWO capacity-sized halves (prev + curr).
      posViews = [
        new Float32Array(msg.sab.pos, 0, cap3),
        new Float32Array(msg.sab.pos, cap3 * 4, cap3),
      ];
      velView = new Float32Array(msg.sab.vel);
      ncView = new Int32Array(msg.sab.nCount);
      maxStepsPerBatch = Math.max(1, msg.maxStepsPerBatch ?? 3);
      maxBatchMs = msg.maxBatchMs != null ? Math.max(0, msg.maxBatchMs) : 12;
      emaStepMs = 0; // fresh EMA per sim (capacity/params may change step cost)
      installProfiler(sim);
      if (msg.bounds) sim.bounds = msg.bounds;
      Atomics.store(ctl, 0, 0);
      Atomics.store(ctl, 2, 0);
      activeBuf = 0;
      Atomics.store(ctl, 4, 0);
      syncStateToSAB();
      syncStateToSAB(); // fill BOTH halves identically: prev == curr at t0
      Atomics.store(ctl, 3, 1); // ready
      post({ type: 'ready', maxParticles: sim.p.maxParticles });
      break;
    }

    case 'colliders':
      colliders = msg.colliders ?? [];
      break;

    case 'step': {
      if (msg.colliders) colliders = msg.colliders; // per-step override (scenes pass them every frame)
      const t0 = performance.now();
      let steps;
      if (msg.fixedSteps) {
        // Exact-N escape hatch (tests / deterministic replay).
        steps = Math.max(1, msg.steps ?? 1);
      } else {
        // REAL-TIME PACING: advance SIM TIME to match wall-clock elapsed since
        // the last completed batch. ceil() carries un-simulated debt into the
        // next batch; the maxSteps clamp trades accuracy for no death-spiral.
        const dtMs = Math.max(1e-4, msg.dt * 1000);
        const elapsed = lastBatchEnd > 0 ? t0 - lastBatchEnd : dtMs;
        const cap_ = Math.max(1, msg.maxSteps ?? maxStepsPerBatch);
        steps = Math.min(cap_, Math.max(1, Math.ceil(elapsed / dtMs)));
        // ADAPTIVE BATCH SIZE (Lane F2): clamp the step count so the batch's
        // predicted WALL-CLOCK cost stays under maxBatchMs (EMA-estimated per
        // step time, min 1 — a single step always runs so the frame advances).
        // Faster machines → more steps/batch (higher sim rate); slower ones →
        // smaller batches (frames land sooner, steadier posLerp cadence).
        if (maxBatchMs > 0 && emaStepMs > 0) {
          const budget = Math.max(1, Math.floor(maxBatchMs / emaStepMs));
          if (budget < steps) steps = budget;
        }
      }
      for (let s = 0; s < steps; s++) sim.step(msg.dt, colliders);
      const simMs = performance.now() - t0;
      syncStateToSAB(true); // write inactive pos half + flip → new curr frame
      Atomics.store(ctl, 2, Math.round(simMs * 1000)); // µs precision, int cell
      const frameId = msg.frameId ?? (Atomics.load(ctl, 0) + 1);
      Atomics.store(ctl, 0, frameId);
      lastBatchEnd = performance.now();
      // leakedTotal / ke ride along every frame message so the main thread can
      // expose sync-parity `sim.leakedTotal` / `sim.kineticEnergy` properties.
      post({
        type: 'frame', frameId, count: sim.count, simMs,
        leakedTotal: sim.leakedTotal ?? 0,
        ke: sim.kineticEnergy ?? 0,
      });
      break;
    }

    case 'spawn':
      sim.spawn(msg.x, msg.y, msg.z, msg.vx, msg.vy, msg.vz);
      syncStateToSAB();
      break;

    case 'spawnBlock':
      sim.spawnBlock(msg.cx, msg.cy, msg.cz, msg.nx, msg.ny, msg.nz, msg.jitter, msg.v0);
      syncStateToSAB();
      break;

    case 'drain':
      sim.drain(msg.region);
      syncStateToSAB();
      break;

    case 'reset':
      sim.reset();
      syncStateToSAB();
      break;

    case 'setParams':
      Object.assign(sim.p, msg.patch ?? {});
      break;

    case 'setBounds':
      sim.bounds = msg.bounds;
      break;

    case 'dispose':
      self.close();
      break;
  }
});
