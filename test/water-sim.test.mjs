// test/water-sim.test.mjs — createSimulation() in inline and threaded modes.
// Threads run under node:worker_threads through test/helpers/node-worker.mjs.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createSimulation } from '../src/water/sim.js';
import { nodeWorkerFactory } from './helpers/node-worker.mjs';

const PARAMS = { spacing: 0.1, maxParticles: 20000 };

function scene(sim) {
  sim.setColliders([
    { type: 'container', position: [0, 1, 0], size: [1.2, 1, 1.2] },
    { type: 'sphere', position: [0.3, 0.3, 0.2], radius: 0.25 },
  ]);
  sim.fillBox([-1.2, 0, -1.2], [0, 1.2, 1.2], { jitter: 0 }); // 12×12×24 = 3456
}

async function runScript(sim) {
  scene(sim);
  const step = (k) => (sim.mode === 'inline' ? sim.stepNow(k) : sim.stepNow(k));
  await step(20);
  sim.fillBox([0.5, 1.2, -0.3], [0.9, 1.6, 0.3], { jitter: 0, velocity: [0, -1, 0] });
  await step(10);
  sim.removeInBox([-1.2, 0, -1.2], [-0.8, 2, -0.8]);
  sim.setColliders([
    { type: 'container', position: [0, 1, 0], size: [1.2, 1, 1.2] },
    { type: 'sphere', position: [0.1, 0.3, 0.2], radius: 0.25, velocity: [-1, 0, 0], dynamic: true },
  ]);
  await step(15);
}

test('threaded steps are bit-identical to inline (spawns, removals, collider changes)', async () => {
  const inline = await createSimulation(PARAMS, { threads: 0 });
  const threaded = await createSimulation(PARAMS, { threads: 3, workerFactory: nodeWorkerFactory });
  try {
    assert.equal(inline.mode, 'inline');
    assert.equal(threaded.mode, 'threaded');
    await runScript(inline);
    await runScript(threaded);
    assert.equal(threaded.count, inline.count);
    // 3456 block + 4×6×4 spawned column − whatever had flowed into the drained corner
    assert.ok(inline.count > 3000 && inline.count < 3456 + 96, `count ${inline.count}`);
    assert.deepEqual(threaded.positions.subarray(0, inline.count * 3), inline.positions.subarray(0, inline.count * 3));
    assert.deepEqual(threaded.ids.subarray(0, inline.count), inline.ids.subarray(0, inline.count));
    assert.deepEqual(threaded.prevPositions.subarray(0, inline.count * 3), inline.prevPositions.subarray(0, inline.count * 3));
    // the moving sphere is dynamic: both modes report the same impulse on it
    const a = inline.takeImpulses(), b = threaded.takeImpulses();
    assert.ok(Math.abs(a.impulses[6] - b.impulses[6]) < 1e-9 && Math.abs(a.impulses[6]) > 0, `sphere impulse ${a.impulses[6]} vs ${b.impulses[6]}`);
  } finally {
    threaded.dispose();
  }
});

test('threaded: a heightfield added mid-run reaches every thread', async () => {
  const nx = 31, nz = 31, heights = new Float32Array(nx * nz);
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) heights[z * nx + x] = 0.3 + 0.05 * Math.sin(x * 0.4) * Math.cos(z * 0.3);
  const hf = { minX: -1.5, minZ: -1.5, dx: 0.1, dz: 0.1, nx, nz, heights };
  const make = async (threads) => {
    const sim = await createSimulation(PARAMS, { threads, workerFactory: nodeWorkerFactory });
    sim.setColliders([{ type: 'plane', position: [0, 0, 0] }]);
    sim.fillBox([-0.6, 0.6, -0.6], [0.6, 1.4, 0.6], { jitter: 0 });
    await sim.stepNow(3);
    const index = sim.addHeightfield(hf);
    sim.setColliders([{ type: 'plane', position: [0, 0, 0] }, { type: 'heightfield', heightfield: index }]);
    await sim.stepNow(40);
    return sim;
  };
  const inline = await make(0);
  const threaded = await make(3);
  try {
    assert.equal(threaded.count, inline.count);
    assert.deepEqual(threaded.positions.subarray(0, inline.count * 3), inline.positions.subarray(0, inline.count * 3));
    // over the terrain's footprint every particle sits on top of it (water
    // that ran off the 3 m patch rests on the floor plane instead)
    let onTerrain = 0;
    for (let i = 0; i < threaded.count; i++) {
      const x = threaded.positions[i * 3], y = threaded.positions[i * 3 + 1], z = threaded.positions[i * 3 + 2];
      if (Math.abs(x) < 1.4 && Math.abs(z) < 1.4) { onTerrain++; assert.ok(y > 0.24, `particle ${i} at y ${y.toFixed(3)} under the terrain`); }
    }
    assert.ok(onTerrain > threaded.count / 2, `${onTerrain} of ${threaded.count} on the terrain`);
  } finally {
    threaded.dispose();
  }
});

test('update() takes fixed steps from real time; interpolate() stays between frames', async () => {
  const sim = await createSimulation(PARAMS, { threads: 0, fixedDt: 1 / 60, maxStepsPerFrame: 3 });
  scene(sim);
  assert.equal(sim.update(1 / 120), 0);          // half a step accumulated
  assert.equal(sim.update(1 / 120), 1);          // completes one
  assert.equal(sim.update(1), 3);                // clamped: slow motion, no spiral
  sim.update(1 / 120);
  const out = new Float32Array(sim.count * 3);
  sim.interpolate(out);
  const a = sim.alpha;
  assert.ok(a > 0 && a < 1, `alpha ${a}`);
  for (let i = 0; i < sim.count * 3; i++) {
    const lo = Math.min(sim.prevPositions[i], sim.positions[i]) - 1e-6;
    const hi = Math.max(sim.prevPositions[i], sim.positions[i]) + 1e-6;
    assert.ok(out[i] >= lo && out[i] <= hi);
  }
});
