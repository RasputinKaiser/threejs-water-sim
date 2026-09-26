// water-pack/async-sim.js — sync-API-compatible wrapper around the off-thread
// WaterSim worker (sim-worker.mjs). Perf-headroom.md Priority 1 / option "e".
//
// Scenes keep their exact call surface: read .pos / .vel / .nCount (now
// Float32Array/Int32Array views over SharedArrayBuffers — zero-copy for
// THREE.BufferAttribute), .count, step(dt) (fire-and-forget, like the sync
// path inside a rAF loop), spawn/spawnBlock/drain/reset/setParams.
//
// BUFFERING SCHEME: DOUBLE-BUFFERED positions + Atomics frame counter with
// v1 copy semantics (worker copies its plain arrays into the SABs after each
// step batch, ~0.2 ms @ 26k). Positions live in TWO halves of one SAB; the
// worker writes the inactive half then flips ctl[4] (curr index) before
// bumping ctl[0] (frameId). vel/nCount stay single-buffered latest-value.
//
// INTERPOLATION CONTRACT (three.js BufferAttribute holds ONE array):
//   - `.pos` returns the CURR half and `.posPrev` the PREVIOUS half as plain
//     Float32Array views. Because the underlying objects flip every batch,
//     scenes must NOT cache `.pos` once into a BufferAttribute and expect
//     motion — re-read it per frame, or (better) interpolate:
//   - `.alpha` ∈ [0,1]: time since the curr frame landed ÷ estimated batch
//     interval (EMA-smoothed from observed frame arrivals, falling back to the
//     worker's ctl[2] simMs). Feed scenes' render loop:
//         attr.array = ... // per frame:
//         sim.posLerp(attr.array); attr.needsUpdate = true;
//   - `.posLerp(out)` fills a preallocated array (length ≥ count*3) with
//     lerp(prev, curr, .alpha) — zero alloc, safe to call every rAF.
//   - `.posExtrap(out, tau)` (Lane B) goes further: hybrid interpolation +
//     capped velocity extrapolation with a per-particle acceleration gate —
//         out = lerp(posPrev, posCurr, α) + avgVel·τ·trust
//         avgVel = (velCurr+velPrev)/2, trust = clamp(1−acc/aRef, 0, 1)
//     Steady-flow particles extrapolate (full-rate motion at render Hz);
//     high-acceleration ones (splash impacts) fall back to pure lerp.
//     τ is defensively clamped to [0, 0.6·dtSim] inside; tune with the
//     `.kExt` (global gain) and `.accelRef` (gate threshold, m/s²) knobs.
//   - Alternatively bind `.posPrev` as a second BufferAttribute and do the
//     lerp in a shader/custom attribute update.
//   - spawn/drain/reset write the ACTIVE half in place (no flip): state
//     teleports are never interpolated from stale positions.
//
// Control SAB layout (Int32Array):
//   [0] last completed frameId   [1] particle count
//   [2] simMs of last batch (µs) [3] ready flag
//   [4] active (curr) pos buffer index — prev is the other half
//
// Feature gate: createAsyncSim/maybeCreateAsyncSim return null unless
// crossOriginIsolated && SharedArrayBuffer are available — callers fall back
// to the synchronous `new WaterSim()` path unchanged. Integration lane wires
// the fallback via maybeCreateAsyncSim; index.js/scenes untouched here.

import { DEFAULT_PARAMS } from './solver.js';

export function asyncSimSupported() {
  return (
    typeof SharedArrayBuffer !== 'undefined' &&
    typeof Worker !== 'undefined' &&
    globalThis.crossOriginIsolated === true
  );
}

/** Returns an async sim, or null when unsupported/failed — drop-in for callers. */
export function maybeCreateAsyncSim(opts) {
  if (!asyncSimSupported()) return null;
  try {
    return createAsyncSim(opts);
  } catch {
    return null;
  }
}

/**
 * @param {object} [opts]
 * @param {object} [opts.params]       solver params patch (see DEFAULT_PARAMS)
 * @param {number} [opts.maxParticles] capacity (defaults to params.maxParticles)
 * @param {{min:[x,y,z],max:[x,y,z]}} [opts.bounds] leak-kill region
 * @param {number} [opts.maxStepsPerBatch] hard cap on paced steps per batch
 * @param {number} [opts.maxBatchMs=12] wall-clock budget per step batch — the
 *      worker adaptively sizes batches so each lands within this budget
 *      (Lane F2); 0 disables the adaptive clamp.
 */
export function createAsyncSim({ params = {}, maxParticles = null, bounds = null, maxStepsPerBatch = 3, maxBatchMs = 12 } = {}) {
  if (!asyncSimSupported()) return null;

  const cap = maxParticles ?? params.maxParticles ?? 9000;
  const ctlSAB = new SharedArrayBuffer(64);
  // Positions are DOUBLE-buffered: one SAB, two capacity-sized halves
  // (worker writes the inactive half each batch, then flips ctl[4]).
  const posSAB = new SharedArrayBuffer(cap * 3 * 4 * 2);
  const velSAB = new SharedArrayBuffer(cap * 3 * 4);
  const nCountSAB = new SharedArrayBuffer(cap * 4);
  const ctl = new Int32Array(ctlSAB);

  const worker = new Worker(new URL('./sim-worker.mjs', import.meta.url), { type: 'module' });
  worker.postMessage({
    type: 'init',
    params,
    maxParticles: cap,
    bounds,
    maxStepsPerBatch,
    maxBatchMs,
    sab: { ctl: ctlSAB, pos: posSAB, vel: velSAB, nCount: nCountSAB },
  });

  let nextFrameId = 0;
  const pendingFrames = new Map(); // frameId -> {resolve, timer}

  // Lane F2 micro-win: scenes typically pass the SAME collider objects every
  // frame (creek/pool/terrain build one heightfield/plane set up front and
  // call step(dt, [collider]) with a fresh array literal). Structured-cloning
  // that set — for a heightfield, a full heights Float32Array — per STEP is
  // pure overhead. We remember the last-sent set (shallow content identity:
  // same length + same element references) and omit `colliders` from the step
  // message when unchanged; the worker keeps its previous set. CAVEAT: if a
  // scene MUTATES a collider payload in place (e.g. rewrites heights), pass a
  // different wrapper array or call setColliders() to force a resend.
  let _lastCollidersSent = null;
  function collidersChanged(c) {
    if (c === _lastCollidersSent) return false;
    if (!Array.isArray(c) || !Array.isArray(_lastCollidersSent)) return true;
    if (c.length !== _lastCollidersSent.length) return true;
    for (let i = 0; i < c.length; i++) if (c[i] !== _lastCollidersSent[i]) return true;
    return false;
  }

  // Latest worker phase-profiling report (Lane F2): {batches, meanStepMs,
  // perPhaseMs:{predict,sort,pairs,relax,scatter,collide,derive},
  // pairsPerStep, dominant} — posted by the worker every 30 batches.
  let _phaseStats = null;

  // Worker-reported per-frame stats (updated on every 'frame' message; before
  // the first step they hold sync-path defaults so HUD code can rely on them).
  let leakedTotal = 0;
  let kineticEnergy = 0;

  // Interpolation bookkeeping (main thread): when the curr frame landed +
  // an EMA estimate of the batch interval, so `.alpha` can sweep [0..1]
  // between sim frames. EMA is fed by observed frame-message arrivals; the
  // worker's reported simMs (ctl[2]) is the fallback until two arrivals.
  const posViews = [
    new Float32Array(posSAB, 0, cap * 3),
    new Float32Array(posSAB, cap * 3 * 4, cap * 3),
  ];
  // Lane B: velocity history. The worker's vel SAB is single-buffered
  // latest-value (and sim-worker.mjs is outside this lane), so prev/curr vel
  // are snapshotted MAIN-SIDE: each 'frame' message (guaranteed to arrive
  // after the worker's SAB write) flips a pair of preallocated snapshots —
  // mirroring how the pos halves swap roles on every batch flip.
  const velSABView = new Float32Array(velSAB);
  const _velSnaps = [new Float32Array(cap * 3), new Float32Array(cap * 3)];
  let _velIdx = 0; // indexes the CURR snapshot; the other one is PREV
  let _lastSeenFrame = -1;
  let _currLandedAt = performance.now();
  let _lastFrameArrival = 0;
  let _emaIntervalMs = 0;
  // Lane B extrapolation knobs (tunable live; see .posExtrap below).
  let _kExt = 1.0;    // global extrapolation gain (0 = pure lerp, 1 = full)
  let _accelRef = 20; // gate threshold m/s² — trust hits 0 at this Δv/dtSim

  // Local params copy mirroring WaterSim's `{ ...DEFAULT_PARAMS, ...params }`.
  // Exposed as `sim.p` — reads hit the local copy (zero latency, same as the
  // sync path); WRITES forward a setParams message to the worker, so GUI
  // mutation (`paramsFolder.add(sim.p, key)`) keeps driving the real sim.
  const pTarget = { ...DEFAULT_PARAMS, ...params };
  const p = new Proxy(pTarget, {
    set(t, key, v) {
      t[key] = v;
      worker.postMessage({ type: 'setParams', patch: { [key]: v } });
      return true;
    },
    deleteProperty(t, key) {
      delete t[key];
      return true;
    },
  });

  worker.addEventListener('message', (e) => {
    const m = e.data;
    if (m?.type === 'debug' && m.kind === 'phases') {
      _phaseStats = m; // HUD-facing: sim.phaseStats / sim.dominantPhase
      return;
    }
    if (m?.type === 'frame') {
      // Lane B: snapshot vel into the prev/curr pair BEFORE anything else —
      // this fires after the worker's SAB write for THIS frameId, so the
      // snapshot is the curr frame's velocity and the other slot holds the
      // previous one. Zero-alloc (.set into preallocated buffers).
      _velIdx = 1 - _velIdx;
      const n3snap = Atomics.load(ctl, 1) * 3;
      _velSnaps[_velIdx].set(
        n3snap <= cap * 3 ? velSABView.subarray(0, n3snap) : velSABView.subarray(0, cap * 3));
      // Feed the batch-interval EMA from observed arrivals (robust to drops:
      // a skipped frame just makes one interval ~2× and the EMA absorbs it).
      const now = performance.now();
      if (_lastFrameArrival > 0) {
        const iv = now - _lastFrameArrival;
        _emaIntervalMs = _emaIntervalMs > 0 ? _emaIntervalMs * 0.85 + iv * 0.15 : iv;
      }
      _lastFrameArrival = now;
      if (m.leakedTotal != null) leakedTotal = m.leakedTotal;
      if (m.ke != null) kineticEnergy = m.ke;
      const pf = pendingFrames.get(m.frameId);
      if (pf) {
        clearTimeout(pf.timer);
        pendingFrames.delete(m.frameId);
        pf.resolve(m);
      }
    }
  });
  worker.addEventListener('error', () => {
    for (const [, p] of pendingFrames) { clearTimeout(p.timer); p.resolve(null); }
    pendingFrames.clear();
  });

  let _bounds = bounds;

  const sim = {
    isAsync: true,

    // Views over the shared buffers — drop-in for THREE.BufferAttribute arrays.
    // `pos`/`posPrev` are GETTERS: they return whichever half is currently
    // curr/prev (the worker flips ctl[4] each batch). Re-read them per frame;
    // do not cache the array object across frames. Prefer `.posLerp(out)` —
    // see INTERPOLATION CONTRACT in the header.
    get pos() { return posViews[Atomics.load(ctl, 4)]; },
    /** Previous frame's position view (for lerp(prev, curr, .alpha)). */
    get posPrev() { return posViews[1 - Atomics.load(ctl, 4)]; },
    vel: new Float32Array(velSAB),
    nCount: new Int32Array(nCountSAB),

    get maxParticles() { return cap; },
    /** Worker-updated particle count (fresh even between frames). */
    get count() { return Atomics.load(ctl, 1); },
    /** Sync-parity alias. */
    get particleCount() { return Atomics.load(ctl, 1); },
    /** Last completed step-batch id (monotonic). */
    get frame() { return Atomics.load(ctl, 0); },
    /** Worker-side ms spent in the most recent step batch. */
    get simMs() { return Atomics.load(ctl, 2) / 1000; },

    /**
     * Interpolation factor [0..1]: time since the curr position frame landed
     * ÷ estimated batch interval (EMA of observed frame arrivals, falling
     * back to the worker-reported simMs until two arrivals). Clamped to
     * [0,1]; returns 1 when no interval estimate exists yet.
     */
    get alpha() {
      const f = Atomics.load(ctl, 0);
      if (f !== _lastSeenFrame) {
        _lastSeenFrame = f;
        _currLandedAt = performance.now();
      }
      const intervalMs = _emaIntervalMs > 0 ? _emaIntervalMs : Atomics.load(ctl, 2) / 1000;
      if (!(intervalMs > 0)) return 1;
      return Math.min(1, Math.max(0, (performance.now() - _currLandedAt) / intervalMs));
    },

    /**
     * Zero-alloc interpolated positions: out[i] = prev + (curr − prev)·alpha
     * for every live particle (count*3 floats). Pass a preallocated
     * Float32Array (e.g. your BufferAttribute's array) each render frame,
     * then set attr.needsUpdate = true.
     */
    posLerp(out) {
      const a = this.alpha; // read first: locks landed-timestamp bookkeeping
      const idx = Atomics.load(ctl, 4);
      const cur = posViews[idx];
      const prv = posViews[1 - idx];
      const n3 = Atomics.load(ctl, 1) * 3;
      for (let i = 0; i < n3; i++) out[i] = prv[i] + (cur[i] - prv[i]) * a;
      return out;
    },

    // ---- Lane B: velocity history + hybrid extrapolation -----------------
    /** Curr-frame velocity snapshot (taken main-side when the frame landed).
     * Zeros until two frames have arrived; re-read per frame like `.pos`. */
    get velCurr() { return _velSnaps[_velIdx]; },
    /** One-batch-old velocity snapshot (zeros before the second frame). */
    get velPrev() { return _velSnaps[1 - _velIdx]; },
    /** Raw curr pos SAB half — stable accessor that survives scene-side
     * rebinding of `.pos` to an interpolated buffer (creek's interp hookup). */
    get posCurr() { return posViews[Atomics.load(ctl, 4)]; },

    /**
     * Batch-interval estimate in seconds: EMA of observed frame arrivals,
     * falling back to worker-reported simMs, clamped to [1e-4, 0.05].
     * Doubles as dtSim for posExtrap's gate — real-time pacing keeps sim
     * time ≈ wall time between batches, so Δvel across a batch divided by
     * this interval IS the per-batch mean acceleration.
     */
    get dtSim() {
      let s = _emaIntervalMs > 0 ? _emaIntervalMs / 1000 : Atomics.load(ctl, 2) / 1e6;
      if (!(s > 0)) s = 1 / 30;
      return Math.min(0.05, Math.max(1e-4, s));
    },

    /** Global extrapolation gain (research/motion-quality.md §1 kExt). */
    get kExt() { return _kExt; },
    set kExt(v) { const n = Number(v); _kExt = Number.isFinite(n) && n > 0 ? n : 0; },
    /** Acceleration-gate threshold m/s² (aRef): trust reaches 0 here. */
    get accelRef() { return _accelRef; },
    set accelRef(v) {
      const n = Number(v);
      _accelRef = Number.isFinite(n) && n > 0 ? n : 20;
    },

    /**
     * Zero-alloc hybrid interpolation + capped velocity extrapolation
     * (motion-quality.md §1 "hybrid", highest-leverage change):
     *   outᵢ = lerp(posPrevᵢ, posCurrᵢ, α) + avgVelᵢ·τ·trustᵢ
     *   avgVel = (velCurr+velPrev)/2 per component
     *   trust  = clamp(1 − |Δvel|/dtSim/aRef, 0, 1)   ← accel gate
     * Steady flow extrapolates at full render rate; splash/impact particles
     * (high accel) degrade to pure lerp exactly where prediction fails.
     * τ and α are clamped defensively inside ([0, 0.6·dtSim] / [0,1]) —
     * callers should clamp too. `out` is caller-provided, length ≥ count*3.
     */
    posExtrap(out, tau) {
      const aRaw = this.alpha; // read first: locks landed-timestamp bookkeeping
      const al = aRaw < 0 ? 0 : aRaw > 1 ? 1 : aRaw;
      let t = Number(tau);
      if (!Number.isFinite(t) || t < 0) t = 0;
      const tauMax = this.dtSim * 0.6;
      if (t > tauMax) t = tauMax;
      const idx = Atomics.load(ctl, 4);
      const cur = posViews[idx];
      const prv = posViews[1 - idx];
      const vc = _velSnaps[_velIdx];
      const vp = _velSnaps[1 - _velIdx];
      let n3 = Atomics.load(ctl, 1) * 3;
      const outLen = out.length | 0;
      if (n3 > outLen) n3 = outLen - (outLen % 3); // defensive: caller's array
      const invDtOverRef = 1 / this.dtSim / _accelRef;
      const kT = t * _kExt * 0.5; // ×½ folds avgVel=(vc+vp)/2 into one scalar
      for (let i = 0; i < n3; i += 3) {
        const dvx = vc[i] - vp[i], dvy = vc[i + 1] - vp[i + 1], dvz = vc[i + 2] - vp[i + 2];
        const w = 1 - Math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz) * invDtOverRef;
        if (w <= 0) {
          out[i] = prv[i] + (cur[i] - prv[i]) * al;
          out[i + 1] = prv[i + 1] + (cur[i + 1] - prv[i + 1]) * al;
          out[i + 2] = prv[i + 2] + (cur[i + 2] - prv[i + 2]) * al;
        } else {
          const s = w * kT;
          out[i] = prv[i] + (cur[i] - prv[i]) * al + (vc[i] + vp[i]) * s;
          out[i + 1] = prv[i + 1] + (cur[i + 1] - prv[i + 1]) * al + (vc[i + 1] + vp[i + 1]) * s;
          out[i + 2] = prv[i + 2] + (cur[i + 2] - prv[i + 2]) * al + (vc[i + 2] + vp[i + 2]) * s;
        }
      }
      return out;
    },

    // ---- sync-API parity members (added by integration lane P2) ----
    /** Params mirror: reads local, writes forwarded to the worker (see above). */
    get p() { return p; },
    get h() { return pTarget.h; },
    get h2() { return pTarget.h * pTarget.h; },
    /** Cumulative out-of-bounds removals — worker-reported each step frame
     * (one frame of latency vs the sync path's immediate value). */
    get leakedTotal() { return leakedTotal; },
    /** Total kinetic energy — worker-reported each step frame. */
    get kineticEnergy() { return kineticEnergy; },

    /** Leak-kill region. Assignment (the way scenes set it:
     * `sim.bounds = {min,max}`) stores locally AND forwards a setBounds
     * message to the worker. Reads return the last-set value. */
    get bounds() { return _bounds; },
    set bounds(b) {
      _bounds = b;
      worker.postMessage({ type: 'setBounds', bounds: b });
    },

    /** Fire-and-forget like the sync path. Colliders may be passed per call
     * (plain serializable objects) or once via setColliders(). Optionally
     * await via waitForFrame(frameId).
     * The worker REAL-TIME PACES batches itself (see sim-worker.mjs): opts
     * .fixedSteps forces exact-N solver passes instead (tests/replay), and
     * opts.maxSteps caps the paced step count for this call. */
    step(dt, colliders = null, steps = 1, opts = null) {
      // Backlog guard: if the worker is still chewing on the previous batch,
      // skip this post. Without it the queue grows unboundedly when solve time
      // exceeds frame time and the visible water lags further and further
      // behind wall-clock (the "choppy / slow-motion" symptom).
      if (Atomics.load(ctl, 0) < nextFrameId - 1) return -1; // dropped frame
      const frameId = ++nextFrameId;
      const msg = { type: 'step', dt, steps, frameId };
      if (colliders != null && collidersChanged(colliders)) {
        msg.colliders = colliders;
        _lastCollidersSent = colliders;
      } // else: worker reuses its previous collider set (Lane F2 dedupe)
      if (opts?.fixedSteps) msg.fixedSteps = true;
      if (opts?.maxSteps != null) msg.maxSteps = opts.maxSteps;
      worker.postMessage(msg);
      return frameId;
    },

    /** Install/replace the collider set on the worker side. */
    setColliders(c) {
      worker.postMessage({ type: 'colliders', colliders: c });
      _lastCollidersSent = c; // keep the dedupe mirror in sync
    },

    /** Latest worker phase-profiling report (null until 30 batches ran). */
    get phaseStats() { return _phaseStats; },
    /** Dominant phase name from the latest report (e.g. 'pairs'). */
    get dominantPhase() { return _phaseStats ? _phaseStats.dominant : null; },

    /** Resolves when the given step batch lands in the SABs (tests/debug). */
    waitForFrame(frameId, timeoutMs = 5000) {
      if (Atomics.load(ctl, 0) >= frameId) return Promise.resolve({ frameId });
      return new Promise((resolve) => {
        const timer = setTimeout(() => { pendingFrames.delete(frameId); resolve(null); }, timeoutMs);
        pendingFrames.set(frameId, { resolve, timer });
      });
    },

    spawn(x, y, z, vx = 0, vy = 0, vz = 0) {
      worker.postMessage({ type: 'spawn', x, y, z, vx, vy, vz });
    },

    spawnBlock(cx, cy, cz, nx, ny, nz, jitter = 0.02, v0 = 0) {
      worker.postMessage({ type: 'spawnBlock', cx, cy, cz, nx, ny, nz, jitter, v0 });
    },

    drain(region) {
      worker.postMessage({ type: 'drain', region });
    },

    reset() {
      worker.postMessage({ type: 'reset' });
    },

    setParams(patch) {
      worker.postMessage({ type: 'setParams', patch });
    },

    setBounds(b) {
      worker.postMessage({ type: 'setBounds', bounds: b });
    },

    dispose() {
      try { worker.postMessage({ type: 'dispose' }); } catch { /* already gone */ }
      worker.terminate();
      pendingFrames.clear();
    },
  };

  return sim;
}
