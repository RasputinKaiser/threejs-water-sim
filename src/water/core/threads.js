// water/core/threads.js — phase-parallel execution across worker threads.
//
// Thread 0 (the coordinator worker) drives the step; threads 1..K-1 are
// helpers parked on an Atomics barrier. For every parallel phase the
// coordinator publishes the phase id, bumps a generation counter and wakes
// the helpers; every thread (coordinator included) runs its contiguous slice
// of the particle range, and the coordinator waits until all helpers report
// done. Phases are gather-form, so the result does not depend on K.
//
// Control block (Int32Array over a SharedArrayBuffer):

export const CTL = {
  gen: 0,       // generation counter (bumped once per dispatched phase)
  done: 1,      // helpers finished with the current generation
  phase: 2,     // phase id (solver PHASE.*) or a control op below
  n: 3,         // particle count for this phase
  hfCount: 4,   // heightfields every helper must hold before resuming
  resumed: 5,   // helpers back in the loop after a YIELD
};
export const CTL_SIZE = 8;

export const OP_YIELD = 100; // helpers return to their event loop (to receive messages)
export const OP_QUIT = 101;

// Below this particle count one thread is faster than a barrier round-trip.
export const MIN_PARALLEL = 1024;

/** Particle range [i0, i1) of thread `tid` out of `K` for `n` particles. */
export function sliceOf(n, K, tid) {
  const chunk = Math.ceil(n / K);
  const i0 = Math.min(n, tid * chunk);
  return [i0, Math.min(n, i0 + chunk)];
}

/**
 * Coordinator side: returns a `parallel(id, n?)` function for solver.step().
 * `n` is the range to split (default: the particle count).
 */
export function makeParallel(solver, ctl, K) {
  return function parallel(id, n = solver.header[0]) {
    if (K <= 1 || n < MIN_PARALLEL) { solver.runPhase(id, 0, n, 0); return; }
    dispatch(ctl, K, id, n);
    const [i0, i1] = sliceOf(n, K, 0);
    solver.runPhase(id, i0, i1, 0);
    waitHelpers(ctl, K);
  };
}

/** Coordinator side: send a control op (YIELD/QUIT) to every helper. */
export function broadcast(ctl, K, op) {
  if (K <= 1) return;
  Atomics.store(ctl, CTL.resumed, 0);
  dispatch(ctl, K, op, 0);
  waitHelpers(ctl, K);
}

/** Coordinator side: block until every helper is (back) in its loop. */
export function waitResumed(ctl, K) {
  let r;
  while ((r = Atomics.load(ctl, CTL.resumed)) < K - 1) Atomics.wait(ctl, CTL.resumed, r, 50);
}

function dispatch(ctl, K, id, n) {
  Atomics.store(ctl, CTL.phase, id);
  Atomics.store(ctl, CTL.n, n);
  Atomics.store(ctl, CTL.done, 0);
  Atomics.add(ctl, CTL.gen, 1);
  Atomics.notify(ctl, CTL.gen);
}

function waitHelpers(ctl, K) {
  // brief spin first: phases are short and a futex round-trip costs ~10-50 µs
  for (let spin = 0; spin < 2000; spin++) if (Atomics.load(ctl, CTL.done) >= K - 1) return;
  let d;
  while ((d = Atomics.load(ctl, CTL.done)) < K - 1) Atomics.wait(ctl, CTL.done, d, 50);
}

/**
 * Helper side: announce readiness, then run phases until YIELD/QUIT. Returns
 * the op that ended the loop. Must be called from a worker (it blocks in
 * Atomics.wait). The generation baseline is read BEFORE announcing, so the
 * coordinator (which waits for the announcement) can never bump it unseen.
 */
export function helperLoop(solver, ctl, tid, K) {
  let gen = Atomics.load(ctl, CTL.gen);
  Atomics.add(ctl, CTL.resumed, 1);
  Atomics.notify(ctl, CTL.resumed);
  for (;;) {
    let g = Atomics.load(ctl, CTL.gen);
    if (g === gen) {
      for (let spin = 0; spin < 2000 && g === gen; spin++) g = Atomics.load(ctl, CTL.gen);
      while (g === gen) { Atomics.wait(ctl, CTL.gen, gen, 100); g = Atomics.load(ctl, CTL.gen); }
    }
    gen = g;
    const id = Atomics.load(ctl, CTL.phase);
    if (id === OP_YIELD || id === OP_QUIT) {
      Atomics.add(ctl, CTL.done, 1);
      Atomics.notify(ctl, CTL.done);
      return id;
    }
    const [i0, i1] = sliceOf(Atomics.load(ctl, CTL.n), K, tid);
    solver.runPhase(id, i0, i1, tid);
    Atomics.add(ctl, CTL.done, 1);
    Atomics.notify(ctl, CTL.done);
  }
}
