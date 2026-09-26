// test/whitewater.test.mjs — diffuse whitewater (src/water/core/whitewater.js):
// generated where water plunges and breaks, never in still water, classified
// by fluid neighborhood, published by the threaded solver too.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createSimulation } from '../src/water/sim.js';
import { nodeWorkerFactory } from './helpers/node-worker.mjs';

const TANK = { type: 'container', position: [0, 1, 0], size: [0.8, 1, 0.8] };

function types(sim) {
  const { count, data } = sim.diffuse;
  const t = [0, 0, 0];
  for (let k = 0; k < count; k++) t[Math.floor(data[k * 4 + 3])]++;
  return t;
}

async function pool(threads) {
  const sim = await createSimulation({ solver: 'dfsph', spacing: 0.1, maxParticles: 12000 },
    { threads, workerFactory: nodeWorkerFactory, maxStepsPerFrame: 1 });
  sim.setColliders([TANK]);
  sim.fillBox([-0.8, 0, -0.8], [0.8, 0.5, 0.8], { jitter: 0 });
  return sim;
}
const step = async (sim, n) => { for (let i = 0; i < n; i++) { if (sim.mode === 'inline') sim.stepNow(1); else await sim.stepNow(1); } };

test('still water makes no whitewater', async () => {
  const sim = await pool(0);
  await step(sim, 120);
  assert.equal(sim.diffuse.count, 0, `diffuse particles in a resting pool: ${sim.diffuse.count}`);
});

test('a block of water plunging into a pool makes spray, foam and bubbles', async () => {
  const sim = await pool(0);
  await step(sim, 30);
  sim.fillBox([-0.3, 1.2, -0.3], [0.3, 1.8, 0.3], { velocity: [0, -3, 0] });
  let peak = [0, 0, 0];
  for (let f = 0; f < 90; f++) {
    await step(sim, 1);
    const t = types(sim);
    peak = peak.map((v, i) => Math.max(v, t[i]));
  }
  const [spray, foam, bubble] = peak;
  assert.ok(foam > 50, `foam ${foam}`);
  assert.ok(spray > 0, `spray ${spray}`);
  assert.ok(bubble > 0, `bubbles ${bubble}`);
  // foam ages out once the pool calms down
  const before = sim.diffuse.count;
  await step(sim, 360);
  assert.ok(sim.diffuse.count < before * 0.5, `whitewater ${before} → ${sim.diffuse.count} after 6 s`);
});

test('the threaded solver publishes whitewater', async () => {
  const sim = await pool(2);
  await step(sim, 10);
  sim.fillBox([-0.3, 1.2, -0.3], [0.3, 1.8, 0.3], { velocity: [0, -3, 0] });
  let peak = 0;
  for (let f = 0; f < 60; f++) { await step(sim, 1); peak = Math.max(peak, sim.diffuse.count); }
  sim.dispose();
  assert.ok(peak > 50, `published whitewater peak ${peak}`);
});
