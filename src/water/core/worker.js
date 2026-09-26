// water/core/worker.js — worker entry for the threaded solver.
//
// Every worker builds a PBFSolver over the same SharedArrayBuffers. Worker 0
// is the COORDINATOR: it receives commands from the main thread, runs the
// serial parts of each step and dispatches parallel phases. Workers 1..K-1 are
// HELPERS that sit in threads.helperLoop() executing their slice of each phase.
//
// Main → coordinator messages (processed in order, between steps):
//   {type:'step', dt, steps, frameId, colliders?}  run `steps` solver steps,
//        then publish positions + stats. `colliders` = count staged in the
//        shared staging records (copied in before the first step).
//   {type:'spawn', data}          Float32Array of [x,y,z,vx,vy,vz] records
//   {type:'removeBox', min, max}
//   {type:'reset'} · {type:'params', patch}
//   {type:'heightfield', index, buffer}   (also sent to every helper first)
//   {type:'dispose'}
// Coordinator → main:
//   {type:'ready'} · {type:'frame', frameId, count, stats, impulses}
//
// Uses only worker globals (self / postMessage / addEventListener), so the
// same file runs in browsers and under node:worker_threads with a shim.

import { PBFSolver, H, U } from './solver.js';
import { COLLIDER_STRIDE } from './colliders.js';
import { CTL, OP_YIELD, OP_QUIT, makeParallel, broadcast, waitResumed, helperLoop } from './threads.js';

let solver = null;
let ctl = null;
let K = 1;
let tid = 0;
const heightfields = [];

// published frame (double-buffered; see sim.js for the layout)
let pub = null;

function initSolver(msg) {
  tid = msg.tid;
  K = msg.threads;
  ctl = new Int32Array(msg.ctl);
  for (const b of msg.heightfields ?? []) heightfields.push(new Float32Array(b));
  solver = new PBFSolver(msg.params, msg.buffers, { tid, init: false, heightfields });
}

/* ------------------------------ helper ------------------------------ */

let yielded = false;

function enterLoop() {
  const op = helperLoop(solver, ctl, tid, K);
  if (op === OP_QUIT) { self.close?.(); return; }
  yielded = true;
  tryResume();
}

function tryResume() {
  if (yielded && heightfields.length >= Atomics.load(ctl, CTL.hfCount)) {
    yielded = false;
    setTimeout(enterLoop, 0);
  }
}

/* ---------------------------- coordinator --------------------------- */

let parallel = null;
let staging = null;      // Float32Array: collider records written by the main thread
let impulseOut = null;

function publish(frameId) {
  const n = solver.header[H.count];
  const active = 1 - pub.header[0];
  const o3 = active * pub.N * 3, o1 = active * pub.N;
  pub.pos.set(solver.pos.subarray(0, n * 3), o3);
  pub.prev.set(solver.framePrev.subarray(0, n * 3), o3);
  pub.vel.set(solver.vel.subarray(0, n * 3), o3);
  pub.nbr.set(solver.nbrCount.subarray(0, n), o1);
  pub.id.set(solver.id.subarray(0, n), o1);
  Atomics.store(pub.header, 1 + active, n);
  Atomics.store(pub.header, 0, active);
  Atomics.store(pub.header, 3, frameId);
}

function stats() {
  const u = solver.u, h = solver.header;
  return {
    kineticEnergy: u[U.kineticEnergy], maxDensityError: u[U.maxDensityError],
    overflow: h[H.overflow], leaked: h[H.leaked], quarantined: h[H.quarantined],
    drained: h[H.drained],
  };
}

function coordinator(msg) {
  switch (msg.type) {
    case 'step': {
      if (msg.colliders != null) {
        solver.colliders.set(staging.subarray(0, msg.colliders * COLLIDER_STRIDE));
        solver.header[H.colliders] = msg.colliders;
      }
      const t0 = performance.now();
      for (let s = 0; s < msg.steps; s++) {
        if (s === msg.steps - 1) solver.markFrame(); // interpolation origin = previous step
        solver.step(msg.dt, parallel);
      }
      const ms = performance.now() - t0;
      publish(msg.frameId);
      solver.reduceImpulses(impulseOut);
      const impulses = impulseOut.slice();
      const contacts = solver.contactStats.slice();
      const st = stats();
      st.phaseMs = Array.from(solver.phaseMs);
      postMessage({ type: 'frame', frameId: msg.frameId, count: solver.header[H.count], ms, stats: st, impulses, contacts },
        [impulses.buffer, contacts.buffer]);
      break;
    }
    // Commands take effect in the next published frame: publishing only once
    // per step batch keeps the double buffer safe (the main thread reads the
    // latest half synchronously and only then posts the next batch).
    case 'spawn': {
      const d = msg.data;
      for (let k = 0; k + 5 < d.length; k += 6) solver.addParticle(d[k], d[k + 1], d[k + 2], d[k + 3], d[k + 4], d[k + 5]);
      break;
    }
    case 'removeBox':
      solver.removeInBox(msg.min, msg.max);
      break;
    case 'reset':
      solver.header[H.count] = 0;
      break;
    case 'params':
      solver.setUniforms(msg.patch);
      break;
    case 'heightfield':
      heightfields[msg.index] = new Float32Array(msg.buffer);
      // helpers must pick up the same heightfield before the next phase
      Atomics.store(ctl, CTL.hfCount, heightfields.length);
      broadcast(ctl, K, OP_YIELD);
      waitResumed(ctl, K);
      break;
    case 'dispose':
      broadcast(ctl, K, OP_QUIT);
      self.close?.();
      break;
  }
}

self.addEventListener('message', (e) => {
  const msg = e.data;
  if (msg.type === 'init') {
    initSolver(msg);
    if (tid === 0) {
      solver.setUniforms(solver.p);
      solver.header[H.tableMask] = solver.T - 1;
      Atomics.store(ctl, CTL.hfCount, heightfields.length);
      parallel = makeParallel(solver, ctl, K);
      staging = new Float32Array(msg.staging);
      impulseOut = new Float64Array(solver.maxColliders * 6);
      pub = {
        N: solver.N,
        header: new Int32Array(msg.pub.header),
        pos: new Float32Array(msg.pub.pos), prev: new Float32Array(msg.pub.prev),
        vel: new Float32Array(msg.pub.vel), nbr: new Int32Array(msg.pub.nbr), id: new Int32Array(msg.pub.id),
      };
      waitResumed(ctl, K);
      postMessage({ type: 'ready' });
    } else {
      yielded = true;
      tryResume();
    }
    return;
  }
  if (tid === 0) coordinator(msg);
  else if (msg.type === 'heightfield') {
    heightfields[msg.index] = new Float32Array(msg.buffer);
    tryResume();
  }
});
