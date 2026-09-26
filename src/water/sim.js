// water/sim.js — the fluid simulation behind one API, in two modes:
//
//   'threaded'  solver runs in K workers over SharedArrayBuffers (needs a
//               cross-origin isolated page: COOP/COEP headers). The main thread
//               only posts commands and reads the last published frame, so a
//               step costs it ~nothing; results arrive one frame later.
//   'inline'    solver runs synchronously on the calling thread (fallback, and
//               the simplest mode for tests/tools).
//
// Time: update(frameDt) advances in fixed steps (fixedDt, default 1/60 s)
// with an accumulator, at most maxStepsPerFrame per call (slow machines run
// in slow motion instead of spiraling). Render positions should be
// interpolate()d: lerp(previous step, latest step, alpha).
//
// Positions/velocities are in the solver's current particle order, which
// changes every step; `prevPositions` is permuted along, so interpolation is
// always per particle. Use `ids` for identity across frames.

import { deriveParams } from './core/params.js';
import { allocateBuffers, solverStats, H } from './core/fluid-core.js';
import { createSolver } from './core/create-solver.js';
import { packWhitewater } from './core/whitewater.js';
import { COLLIDER_STRIDE, writeCollider, packHeightfield } from './core/colliders.js';
import { CTL_SIZE } from './core/threads.js';
import { mulberry32 } from 'math/random';

const isNode = typeof process !== 'undefined' && !!process.versions?.node;

export function threadsAvailable() {
  return typeof SharedArrayBuffer !== 'undefined' &&
    typeof Atomics !== 'undefined' &&
    (isNode || globalThis.crossOriginIsolated === true);
}

function defaultThreads() {
  const hc = globalThis.navigator?.hardwareConcurrency ?? 4;
  return Math.max(1, Math.min(8, hc - 1));
}

/**
 * @param {object} params   solver params (see core/params.js DEFAULTS)
 * @param {object} [opts]
 *   threads          'auto' | number of solver threads | 0 (inline)
 *   maxColliders     collider capacity (default 64)
 *   fixedDt          solver step (s), default 1/60
 *   maxStepsPerFrame default 3
 *   workerFactory    () => Worker-like; default: a module Worker on core/worker.js
 */
export async function createSimulation(params = {}, opts = {}) {
  const dp = deriveParams(params);
  const threads = opts.threads === 'auto' || opts.threads == null ? defaultThreads() : opts.threads;
  const useThreads = threads > 0 && threadsAvailable();
  const sim = useThreads ? await createThreaded(dp, threads, opts) : createInline(dp, opts);
  return sim;
}

/* ============================== shared ============================== */

function makeBase(dp, opts) {
  const fixedDt = opts.fixedDt ?? 1 / 60;
  const maxSteps = opts.maxStepsPerFrame ?? 3;
  const maxColliders = opts.maxColliders ?? 64;
  let acc = 0;
  return {
    params: dp,
    fixedDt,
    maxColliders,
    /** Seconds of simulated time to run this frame (accumulator). */
    takeSteps(frameDt) {
      acc += Math.max(0, Math.min(frameDt, 0.25));
      let k = Math.floor(acc / fixedDt + 1e-9);
      if (k > maxSteps) { k = maxSteps; acc = fixedDt * 0.999; } else acc -= k * fixedDt;
      return k;
    },
    get alpha() { return Math.min(1, acc / fixedDt); },
  };
}

// fillBox → Float32Array of [x,y,z,vx,vy,vz] on the solver's rest lattice.
// jitter (fraction of spacing) breaks the perfect lattice; pass `seed` for a
// reproducible fill.
export function latticeBox(dp, min, max, { velocity = [0, 0, 0], jitter = 0.01, seed = null } = {}) {
  const rng = mulberry32.create(seed ?? mulberry32.seed());
  const random = () => mulberry32.sample(rng);
  const s = dp.spacing, j = jitter * s;
  // +1e-6: (1.2 − 0) / 0.1 is 11.999… in floating point
  const nx = Math.max(0, Math.floor((max[0] - min[0]) / s + 1e-6));
  const ny = Math.max(0, Math.floor((max[1] - min[1]) / s + 1e-6));
  const nz = Math.max(0, Math.floor((max[2] - min[2]) / s + 1e-6));
  const out = new Float32Array(nx * ny * nz * 6);
  let k = 0;
  for (let y = 0; y < ny; y++) for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
    out[k++] = min[0] + (x + 0.5) * s + (random() - 0.5) * j;
    out[k++] = min[1] + (y + 0.5) * s + (random() - 0.5) * j;
    out[k++] = min[2] + (z + 0.5) * s + (random() - 0.5) * j;
    out[k++] = velocity[0]; out[k++] = velocity[1]; out[k++] = velocity[2];
  }
  return out;
}

function toSpawnArray(particles) {
  if (particles instanceof Float32Array) return particles;
  const out = new Float32Array(particles.length * 6);
  particles.forEach((p, i) => {
    out[i * 6] = p[0]; out[i * 6 + 1] = p[1]; out[i * 6 + 2] = p[2];
    out[i * 6 + 3] = p[3] ?? 0; out[i * 6 + 4] = p[4] ?? 0; out[i * 6 + 5] = p[5] ?? 0;
  });
  return out;
}

function interpolateInto(out, prev, cur, n, alpha) {
  const m = n * 3;
  if (alpha >= 1) { out.set(cur.subarray(0, m)); return out; }
  for (let i = 0; i < m; i++) out[i] = prev[i] + (cur[i] - prev[i]) * alpha;
  return out;
}

/* ============================== inline ============================== */

function createInline(dp, opts) {
  const base = makeBase(dp, opts);
  const heightfields = [];
  const buffers = allocateBuffers(dp, { maxColliders: base.maxColliders });
  const solver = createSolver(dp, buffers, { heightfields });
  const diffusePacked = new Float32Array((buffers.D || 0) * 4);
  const impulses = new Float64Array(base.maxColliders * 6);
  const scratch = new Float64Array(base.maxColliders * 6);
  let impulseTime = 0;
  let lastMs = 0;

  function runSteps(k) {
    const t0 = performance.now();
    for (let s = 0; s < k; s++) {
      if (s === k - 1) solver.markFrame(); // interpolation origin = previous step
      solver.step(base.fixedDt);
      solver.reduceImpulses(scratch);
      for (let i = 0; i < scratch.length; i++) impulses[i] += scratch[i];
    }
    impulseTime += k * base.fixedDt;
    lastMs = performance.now() - t0;
  }

  const sim = {
    mode: 'inline',
    maxColliders: base.maxColliders,
    fixedDt: base.fixedDt,
    threads: 0,
    ready: Promise.resolve(),
    solver,
    get params() { return solver.p; },
    get count() { return solver.header[H.count]; },
    get positions() { return solver.pos; },
    get prevPositions() { return solver.framePrev; },
    get velocities() { return solver.vel; },
    get neighborCounts() { return solver.nbrCount; },
    get ids() { return solver.id; },
    get alpha() { return base.alpha; },
    get stepMs() { return lastMs; },
    get stats() { return solverStats(solver); },
    /** Whitewater: { count, data: [x y z type+alpha]* } (packed on each read). */
    get diffuse() {
      const m = solver.D ? solver.header[H.diffuse] : 0;
      return { count: m ? packWhitewater(solver.diffuse, m, diffusePacked) : 0, data: diffusePacked };
    },
    /**
     * Per-slot fluid impulses [Jx,Jy,Jz, Lx,Ly,Lz] (N·s, moments about the
     * collider origin) accumulated since the last read over `time` seconds,
     * plus the latest step's contacts per slot [count, mean fluid vx,vy,vz].
     */
    takeImpulses() {
      const t = impulseTime; impulseTime = 0;
      const out = impulses.slice(0, base.maxColliders * 6);
      impulses.fill(0);
      return { impulses: out, time: t, contacts: solver.contactStats.slice() };
    },
    setParams(patch) { solver.setUniforms(patch); },
    spawn(particles) {
      const d = toSpawnArray(particles);
      let added = 0;
      for (let k = 0; k + 5 < d.length; k += 6) if (solver.addParticle(d[k], d[k + 1], d[k + 2], d[k + 3], d[k + 4], d[k + 5])) added++;
      return added;
    },
    fillBox(min, max, o) { return sim.spawn(latticeBox(solver.p, min, max, o)); },
    removeInBox(min, max) { return solver.removeInBox(min, max); },
    reset() { solver.header[H.count] = 0; },
    setColliders(list) {
      const n = Math.min(list.length, base.maxColliders);
      for (let i = 0; i < n; i++) writeCollider(solver.colliders, i, list[i]);
      solver.header[H.colliders] = n;
    },
    addHeightfield(desc) {
      heightfields.push(desc instanceof Float32Array ? desc : packHeightfield(desc));
      return heightfields.length - 1;
    },
    /** Advance by frameDt of real time (fixed steps). Returns steps taken. */
    update(frameDt) {
      const k = base.takeSteps(frameDt);
      if (k > 0) runSteps(k);
      return k;
    },
    /** Run exactly `steps` solver steps now (tests / baking). */
    stepNow(steps = 1) { runSteps(steps); },
    interpolate(out, alpha = base.alpha) {
      return interpolateInto(out, solver.framePrev, solver.pos, solver.header[H.count], alpha);
    },
    dispose() {},
  };
  return sim;
}

/* ============================= threaded ============================= */

async function createThreaded(dp, K, opts) {
  const base = makeBase(dp, opts);
  const sab = (n) => new SharedArrayBuffer(n);
  const buffers = allocateBuffers(dp, { alloc: sab, threads: K, maxColliders: base.maxColliders });
  const N = dp.maxParticles;
  const ctl = sab(CTL_SIZE * 4);
  const staging = sab(base.maxColliders * COLLIDER_STRIDE * 4);
  const D = buffers.D;
  const pubBuf = {
    header: sab(32), pos: sab(2 * N * 3 * 4), prev: sab(2 * N * 3 * 4),
    vel: sab(2 * N * 3 * 4), nbr: sab(2 * N * 4), id: sab(2 * N * 4),
    diffuse: sab(2 * D * 4 * 4), // whitewater, packed [x y z type+alpha]
  };
  const pub = {
    header: new Int32Array(pubBuf.header),
    pos: new Float32Array(pubBuf.pos), prev: new Float32Array(pubBuf.prev),
    vel: new Float32Array(pubBuf.vel), nbr: new Int32Array(pubBuf.nbr), id: new Int32Array(pubBuf.id),
    diffuse: new Float32Array(pubBuf.diffuse),
  };
  // The active half is LATCHED once per update(): count and every array view
  // then come from the same published frame even if the worker flips halves
  // meanwhile (reading them through separate Atomics loads raced with the
  // flip and mixed a new count with the other half's stale positions).
  let latched = 0, latchedCount = 0, latchedDiffuse = 0;
  function latch() {
    latched = Atomics.load(pub.header, 0);
    latchedCount = Atomics.load(pub.header, 1 + latched);
    latchedDiffuse = Atomics.load(pub.header, 4 + latched);
  }
  const diffuseHalves = [pub.diffuse.subarray(0, D * 4), pub.diffuse.subarray(D * 4, 2 * D * 4)];
  const view = (arr, stride) => {
    const cache = [arr.subarray(0, N * stride), arr.subarray(N * stride, 2 * N * stride)];
    return () => cache[latched];
  };
  const curPos = view(pub.pos, 3), curPrev = view(pub.prev, 3), curVel = view(pub.vel, 3);
  const curNbr = view(pub.nbr, 1), curId = view(pub.id, 1);
  const stagingF = new Float32Array(staging);

  const makeWorker = opts.workerFactory ?? (() => new Worker(new URL('./core/worker.js', import.meta.url), { type: 'module' }));
  const workers = [];
  for (let t = 0; t < K; t++) workers.push(makeWorker());
  const coord = workers[0];

  const heightfieldBuffers = [];
  let busy = false;
  let frameId = 0;
  let lastStats = {};
  let lastMs = 0;
  let pendingSpawn = [];
  let collidersCount = 0;
  let collidersDirty = false;
  const impulseAcc = new Float64Array(base.maxColliders * 6);
  let impulseTime = 0;
  let contactStats = new Float64Array(base.maxColliders * 4);
  const inflightSteps = [];
  let onFrame = null;

  const ready = new Promise((resolve, reject) => {
    coord.addEventListener('message', (e) => {
      const m = e.data;
      if (m.type === 'ready') resolve();
      else if (m.type === 'frame') {
        busy = false;
        latch();
        lastStats = m.stats;
        lastMs = m.ms;
        const steps = inflightSteps.shift() ?? 0;
        impulseTime += steps * base.fixedDt;
        for (let i = 0; i < m.impulses.length; i++) impulseAcc[i] += m.impulses[i];
        contactStats = m.contacts;
        onFrame?.(m);
      }
    });
    coord.addEventListener('error', (e) => reject(e.error ?? e));
  });
  for (let t = 0; t < K; t++) {
    workers[t].postMessage({
      type: 'init', tid: t, threads: K, params: dp, buffers, ctl,
      staging, pub: pubBuf, heightfields: [],
    });
  }
  await ready;

  function flushSpawns() {
    if (!pendingSpawn.length) return;
    const total = pendingSpawn.reduce((s, a) => s + a.length, 0);
    const data = new Float32Array(total);
    let o = 0;
    for (const a of pendingSpawn) { data.set(a, o); o += a.length; }
    pendingSpawn = [];
    coord.postMessage({ type: 'spawn', data }, [data.buffer]);
  }

  function postSteps(k) {
    flushSpawns();
    busy = true;
    inflightSteps.push(k);
    const msg = { type: 'step', dt: base.fixedDt, steps: k, frameId: ++frameId };
    if (collidersDirty) { msg.colliders = collidersCount; collidersDirty = false; }
    coord.postMessage(msg);
  }

  let owed = 0; // steps accumulated while the workers were busy
  const sim = {
    mode: 'threaded',
    maxColliders: base.maxColliders,
    fixedDt: base.fixedDt,
    threads: K,
    ready,
    get params() { return dp; },
    get count() { return latchedCount; },
    get positions() { return curPos(); },
    get prevPositions() { return curPrev(); },
    get velocities() { return curVel(); },
    get neighborCounts() { return curNbr(); },
    get ids() { return curId(); },
    get alpha() { return base.alpha; },
    get stepMs() { return lastMs; },
    get stats() { return lastStats; },
    /** Whitewater of the latest frame: { count, data: [x y z type+alpha]* }. */
    get diffuse() { return { count: latchedDiffuse, data: diffuseHalves[latched] }; },
    get busy() { return busy; },
    takeImpulses() {
      const t = impulseTime; impulseTime = 0;
      const out = impulseAcc.slice();
      impulseAcc.fill(0);
      return { impulses: out, time: t, contacts: contactStats };
    },
    setParams(patch) { coord.postMessage({ type: 'params', patch }); },
    spawn(particles) { const d = toSpawnArray(particles); pendingSpawn.push(d); return d.length / 6; },
    fillBox(min, max, o) { return sim.spawn(latticeBox(dp, min, max, o)); },
    removeInBox(min, max) { flushSpawns(); coord.postMessage({ type: 'removeBox', min, max }); },
    reset() { pendingSpawn = []; coord.postMessage({ type: 'reset' }); },
    setColliders(list) {
      const n = Math.min(list.length, base.maxColliders);
      for (let i = 0; i < n; i++) writeCollider(stagingF, i, list[i]);
      collidersCount = n;
      collidersDirty = true;
    },
    addHeightfield(desc) {
      const packed = desc instanceof Float32Array ? desc : packHeightfield(desc);
      const buf = sab(packed.byteLength);
      new Float32Array(buf).set(packed);
      const index = heightfieldBuffers.length;
      heightfieldBuffers.push(buf);
      for (let t = 1; t < K; t++) workers[t].postMessage({ type: 'heightfield', index, buffer: buf });
      coord.postMessage({ type: 'heightfield', index, buffer: buf });
      return index;
    },
    /** Advance by frameDt of real time. Posts a step batch when the workers are idle. */
    update(frameDt) {
      owed += base.takeSteps(frameDt);
      if (busy || owed === 0) return 0;
      const k = Math.min(owed, opts.maxStepsPerFrame ?? 3);
      owed = 0;
      postSteps(k);
      return k;
    },
    /** Run exactly `steps` steps; resolves when the frame is published. */
    stepNow(steps = 1) {
      return new Promise((resolve) => {
        const run = () => { onFrame = (m) => { onFrame = null; resolve(m); }; postSteps(steps); };
        if (!busy) run();
        else { const prev = onFrame; onFrame = (m) => { prev?.(m); run(); }; }
      });
    },
    interpolate(out, alpha = base.alpha) {
      return interpolateInto(out, curPrev(), curPos(), sim.count, alpha);
    },
    dispose() {
      coord.postMessage({ type: 'dispose' });
      setTimeout(() => workers.forEach((w) => w.terminate?.()), 100);
    },
  };
  return sim;
}
