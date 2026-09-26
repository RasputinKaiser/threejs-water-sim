// test/create-water.test.mjs — the drop-in createWater() API, headless
// (render: false, main-thread solver).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createWater } from '../src/water/index.js';

const TANK = { type: 'container', position: [0, 1, 0], size: [1, 1, 1] };

async function make(extra = {}) {
  return createWater({ threads: 0, render: false, quality: 'medium', colliders: [TANK], ...extra });
}

test('fillBox fills at rest density; update returns the simulated time', async () => {
  const water = await make();
  const n = water.fillBox([-1, 0, -1], [1, 0.5, 1]);
  assert.equal(n, 20 * 5 * 20);
  assert.equal(water.update(1 / 120), 0, 'half a step: nothing simulated yet');
  assert.ok(Math.abs(water.update(1 / 120) - 1 / 60) < 1e-12, 'one fixed step');
  assert.ok(Math.abs(water.update(1 / 20) - 3 / 60) < 1e-12, 'three steps');
});

test('a source delivers π·r²·v of water per second on the rest lattice', async () => {
  const water = await make();
  const r = 0.2, v = 2;
  water.addSource({ position: [0, 1.5, 0], direction: [0, -1, 0], radius: r, speed: v });
  const T = 1;
  for (let t = 0; t < T * 60; t++) water.update(1 / 60);
  const s = water.params.spacing;
  const expected = (Math.PI * r * r * v * T) / (s * s * s);
  assert.ok(Math.abs(water.count - expected) / expected < 0.25, `${water.count} particles vs ${expected.toFixed(0)} expected`);
  // particles leave at the nozzle speed, not faster (no overlap blow-up)
  let maxV = 0;
  for (let i = 0; i < water.count * 3; i++) maxV = Math.max(maxV, Math.abs(water.sim.velocities[i]));
  assert.ok(maxV < v + 9.81 * T + 0.5, `max speed ${maxV.toFixed(2)}`);
});

test('drains remove water that reaches them; sources can be switched off', async () => {
  const water = await make();
  const src = water.addSource({ position: [0.5, 1.2, 0.5], direction: [0, -1, 0], radius: 0.15, speed: 3 });
  const drain = water.addDrain({ min: [-1, -0.1, -1], max: [1, 0.2, 1] });
  for (let t = 0; t < 120; t++) water.update(1 / 60);
  assert.ok(water.stats.drained > 0, 'drained some');
  src.enabled = false;
  for (let t = 0; t < 120; t++) water.update(1 / 60);
  const n = water.count;
  for (let t = 0; t < 60; t++) water.update(1 / 60);
  assert.ok(water.count <= n, 'no new water with the source off');
  drain.remove();
});

test('surfaceHeight reports the free surface over water and -Infinity elsewhere', async () => {
  const water = await make();
  water.fillBox([-1, 0, -1], [1, 0.6, 1]);
  for (let t = 0; t < 120; t++) water.update(1 / 60);
  const h = water.surfaceHeight(0.3, -0.2);
  assert.ok(Math.abs(h - 0.6) < 0.08, `surface ${h.toFixed(3)} (filled to 0.6)`);
  assert.equal(water.surfaceHeight(5, 5), -Infinity);
});

test('static colliders can be added and removed at runtime', async () => {
  const water = await make({ colliders: [] });
  const floor = water.addCollider({ type: 'plane', position: [0, 0, 0] });
  water.fillBox([-0.5, 0.2, -0.5], [0.5, 0.6, 0.5]);
  for (let t = 0; t < 60; t++) water.update(1 / 60);
  let minY = Infinity;
  for (let i = 0; i < water.count; i++) minY = Math.min(minY, water.sim.positions[i * 3 + 1]);
  assert.ok(minY > 0, 'held by the floor');
  water.removeCollider(floor);
  for (let t = 0; t < 30; t++) water.update(1 / 60);
  minY = Infinity;
  for (let i = 0; i < water.count; i++) minY = Math.min(minY, water.sim.positions[i * 3 + 1]);
  assert.ok(minY < -0.2, 'falls once the floor is removed');
});
