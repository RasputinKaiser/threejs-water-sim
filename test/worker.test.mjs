// test/worker.test.mjs — protocol test for the off-thread solver
// (src/water-pack/sim-worker.mjs): init → spawn → paced/fixed step batches →
// double-buffered position flip → phase reports → drain/reset.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { PHASES } from '../src/water-pack/solver.js';

const CAP = 4000;

function startWorker() {
  const worker = new Worker(new URL('./helpers/worker-shim.mjs', import.meta.url));
  const sab = {
    ctl: new SharedArrayBuffer(64),
    pos: new SharedArrayBuffer(CAP * 3 * 4 * 2),
    vel: new SharedArrayBuffer(CAP * 3 * 4),
    nCount: new SharedArrayBuffer(CAP * 4),
  };
  const ctl = new Int32Array(sab.ctl);
  const pos = [new Float32Array(sab.pos, 0, CAP * 3), new Float32Array(sab.pos, CAP * 3 * 4, CAP * 3)];
  const waiters = [];
  const debug = [];
  worker.on('message', (m) => {
    if (m.type === 'debug') debug.push(m);
    for (let i = waiters.length - 1; i >= 0; i--) {
      if (waiters[i].type === m.type) { waiters[i].resolve(m); waiters.splice(i, 1); }
    }
  });
  const next = (type) => new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`timeout waiting for '${type}'`)), 20000);
    waiters.push({ type, resolve: (m) => { clearTimeout(timer); resolve(m); } });
  });
  return { worker, sab, ctl, pos, next, debug };
}

test('worker: init, spawn, step batches, flip, phase report, drain, reset', async () => {
  const w = startWorker();
  try {
    const ready = w.next('ready');
    w.worker.postMessage({
      type: 'init', params: { h: 0.3 }, maxParticles: CAP,
      bounds: { min: [-10, -2, -10], max: [10, 10, 10] }, sab: w.sab,
    });
    assert.equal((await ready).maxParticles, CAP);
    assert.equal(Atomics.load(w.ctl, 3), 1, 'ready flag');

    w.worker.postMessage({ type: 'spawnBlock', cx: 0, cy: 0.5, cz: 0, nx: 10, ny: 10, nz: 10, jitter: 0.02, v0: 0 });
    const floor = [{ type: 'plane', o: [0, 0, 0], n: [0, 1, 0] }];

    let frame = null;
    const active0 = Atomics.load(w.ctl, 4);
    for (let f = 1; f <= 32; f++) {
      const p = w.next('frame');
      w.worker.postMessage({ type: 'step', dt: 1 / 60, steps: 1, fixedSteps: true, frameId: f, colliders: f === 1 ? floor : undefined });
      frame = await p;
      assert.equal(frame.frameId, f);
    }
    assert.equal(frame.count, 1000);
    assert.equal(Atomics.load(w.ctl, 1), 1000, 'ctl count');
    assert.equal(Atomics.load(w.ctl, 0), 32, 'ctl frameId');
    assert.equal(Atomics.load(w.ctl, 4), active0, 'even number of flips returns to the start half');

    // curr half holds finite, floor-respecting positions; prev half differs
    const curr = w.pos[Atomics.load(w.ctl, 4)], prev = w.pos[1 - Atomics.load(w.ctl, 4)];
    let moved = 0;
    for (let i = 0; i < 3000; i++) {
      assert.ok(Number.isFinite(curr[i]));
      if (curr[i] !== prev[i]) moved++;
    }
    for (let i = 0; i < 1000; i++) assert.ok(curr[i * 3 + 1] > -0.05, 'above floor');
    assert.ok(moved > 0, 'prev/curr halves differ (double buffering)');

    // one phase report after 30 steps, keyed by the solver's PHASES
    assert.ok(w.debug.length >= 1, 'phase report posted');
    const rep = w.debug[0];
    assert.equal(rep.kind, 'phases');
    assert.deepEqual(Object.keys(rep.perPhaseMs).sort(), [...PHASES].sort());
    assert.ok(PHASES.includes(rep.dominant));
    assert.ok(rep.pairsPerStep > 0);

    // drain half, then reset — both reflected in ctl[1] after a step
    w.worker.postMessage({ type: 'drain', region: { min: [-10, -2, -10], max: [0, 10, 10] } });
    let p = w.next('frame');
    w.worker.postMessage({ type: 'step', dt: 1 / 60, steps: 1, fixedSteps: true, frameId: 33 });
    const afterDrain = (await p).count;
    assert.ok(afterDrain > 0 && afterDrain < 1000, `drained to ${afterDrain}`);
    w.worker.postMessage({ type: 'reset' });
    p = w.next('frame');
    w.worker.postMessage({ type: 'step', dt: 1 / 60, steps: 1, fixedSteps: true, frameId: 34 });
    assert.equal((await p).count, 0);
  } finally {
    await w.worker.terminate();
  }
});
