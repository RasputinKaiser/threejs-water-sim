// test/box3d-coupling.test.mjs — two-way fluid ↔ Box3D coupling with the real
// Box3D WASM build. Buoyancy is checked against Archimedes on shapes whose
// floating orientation is stable (a cube between ~0.21 and ~0.79 relative
// density floats tilted, so those densities are not used for depth checks).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import Box3DInit from 'box3d.js/inline';
import { createSimulation } from '../src/water/sim.js';
import { createBox3DCoupling } from '../src/water/box3d.js';
import { nodeWorkerFactory } from './helpers/node-worker.mjs';

const b3 = await Box3DInit();
const DT = 1 / 60;
const TANK = { type: 'container', position: [0, 1, 0], size: [0.8, 1, 0.8] };

function makeWorld() {
  const wd = b3.b3DefaultWorldDef();
  wd.gravity = [0, -9.81, 0];
  const world = b3.b3CreateWorld(wd);
  const d = b3.b3DefaultBodyDef();
  d.position = [0, -0.5, 0];
  b3.b3CreateBoxShape(b3.b3CreateBody(world, d), b3.b3DefaultShapeDef(), 5, 0.5, 5);
  return world;
}

function dynamicBody(world, position, density, addShape) {
  const d = b3.b3DefaultBodyDef();
  d.type = b3.b3BodyType.b3_dynamicBody;
  d.position = position;
  const body = b3.b3CreateBody(world, d);
  const sd = b3.b3DefaultShapeDef();
  sd.density = density;
  addShape(body, sd);
  return body;
}

async function run({ density, shape = 'box', half = 0.2, frames = 420, threads = 0, dropY = 1.0 }) {
  const world = makeWorld();
  const body = dynamicBody(world, [0, dropY, 0], density, (b, sd) => {
    if (shape === 'sphere') b3.b3CreateSphereShape(b, sd, { center: [0, 0, 0], radius: half });
    else b3.b3CreateBoxShape(b, sd, half, half, half);
  });
  const sim = await createSimulation({ spacing: 0.1, maxParticles: 8000 }, { threads, workerFactory: nodeWorkerFactory, maxStepsPerFrame: 1 });
  sim.fillBox([-0.8, 0, -0.8], [0.8, 0.6, 0.8], { jitter: 0 });
  const coupling = createBox3DCoupling({ sim, b3, world, colliders: [TANK] });
  const p = [0, 0, 0];
  let ySum = 0, yN = 0, maxY = -Infinity;
  for (let f = 0; f < frames; f++) {
    coupling.update();
    b3.b3World_Step(world, DT, 4);
    if (sim.mode === 'inline') sim.update(DT);
    else { sim.update(DT); await new Promise((r) => setTimeout(r, 0)); while (sim.busy) await new Promise((r) => setTimeout(r, 1)); }
    b3.b3Body_GetPosition(p, body);
    if (f > 30) maxY = Math.max(maxY, p[1]);
    if (f >= frames - 60) { ySum += p[1]; yN++; }
  }
  // free surface away from the body: mean top particle per 0.2 m column + half a spacing
  const cols = new Map();
  for (let i = 0; i < sim.count; i++) {
    const x = sim.positions[i * 3], z = sim.positions[i * 3 + 2];
    if (Math.max(Math.abs(x), Math.abs(z)) < half + 0.25) continue;
    const key = `${Math.floor(x / 0.2)},${Math.floor(z / 0.2)}`;
    cols.set(key, Math.max(cols.get(key) ?? -Infinity, sim.positions[i * 3 + 1]));
  }
  const surface = [...cols.values()].reduce((a, b) => a + b, 0) / cols.size + 0.05;
  sim.dispose();
  return { y: ySum / yN, surface, maxY };
}

test('a half-density sphere floats with its center at the surface', async () => {
  const r = await run({ density: 500, shape: 'sphere' });
  assert.ok(Math.abs(r.y - r.surface) < 0.06, `sphere center ${r.y.toFixed(3)} vs surface ${r.surface.toFixed(3)}`);
});

test('cubes and spheres float at their Archimedes depth; a dense cube sinks', async () => {
  const cube = await run({ density: 150 });
  const submerged = (cube.surface - (cube.y - 0.2)) / 0.4;
  assert.ok(Math.abs(submerged - 0.15) < 0.1, `cube 150 kg/m³: submerged ${submerged.toFixed(2)} vs 0.15`);
  // a sphere with 70% of its volume under water sits with its center 0.26 R below the surface
  const sphere = await run({ density: 700, shape: 'sphere' });
  const depth = sphere.surface - sphere.y;
  assert.ok(Math.abs(depth - 0.26 * 0.2) < 0.05, `sphere 700 kg/m³: center ${depth.toFixed(3)} m below surface vs 0.052`);
  const heavy = await run({ density: 2000, frames: 240 });
  assert.ok(heavy.y < 0.3, `dense cube rests on the floor (y ${heavy.y.toFixed(3)})`);
});

test('a light body dropped into water is not launched out of it', async () => {
  const r = await run({ density: 150, frames: 200, dropY: 1.2 });
  assert.ok(r.maxY < 1.25, `max height after entry ${r.maxY.toFixed(2)} m (dropped from 1.2 m)`);
});

test('threaded solver couples the same way (sphere floats at the surface)', async () => {
  const r = await run({ density: 500, shape: 'sphere', threads: 2 });
  assert.ok(Math.abs(r.y - r.surface) < 0.06, `sphere center ${r.y.toFixed(3)} vs surface ${r.surface.toFixed(3)}`);
});

test('a kinematic paddle pushes the water it sweeps through', async () => {
  const world = makeWorld();
  const d = b3.b3DefaultBodyDef();
  d.type = b3.b3BodyType.b3_kinematicBody;
  d.position = [-0.6, 0.3, 0];
  d.linearVelocity = [1.5, 0, 0];
  const paddle = b3.b3CreateBody(world, d);
  b3.b3CreateBoxShape(paddle, b3.b3DefaultShapeDef(), 0.05, 0.3, 0.6);
  const sim = await createSimulation({ spacing: 0.1, maxParticles: 8000 }, { threads: 0 });
  sim.setColliders([TANK]);
  sim.fillBox([-0.8, 0, -0.8], [0.8, 0.4, 0.8], { jitter: 0 });
  sim.stepNow(60);
  const coupling = createBox3DCoupling({ sim, b3, world, colliders: [TANK] });
  for (let f = 0; f < 30; f++) { coupling.update(); b3.b3World_Step(world, DT, 4); sim.update(DT); }
  let px = 0;
  for (let i = 0; i < sim.count; i++) px += sim.velocities[i * 3];
  assert.ok(px / sim.count > 0.1, `mean water x-velocity ${(px / sim.count).toFixed(3)} m/s`);
});

test('a current carries a floating log downstream (drag through the coupling)', async () => {
  const world = makeWorld();
  const d = b3.b3DefaultBodyDef();
  d.type = b3.b3BodyType.b3_dynamicBody;
  d.position = [-2.2, 0.45, 0];
  const log = b3.b3CreateBody(world, d);
  const sd = b3.b3DefaultShapeDef();
  sd.density = 600;
  b3.b3CreateCapsuleShape(log, sd, { center1: [0, 0, -0.25], center2: [0, 0, 0.25], radius: 0.1 });
  const sim = await createSimulation({ spacing: 0.1, maxParticles: 12000 }, { threads: 0 });
  const channel = { type: 'container', position: [0, 1, 0], size: [3, 1, 0.5] };
  sim.fillBox([-3, 0, -0.5], [3, 0.4, 0.5], { jitter: 0, velocity: [1.5, 0, 0] });
  const coupling = createBox3DCoupling({ sim, b3, world, colliders: [channel] });
  const p = [0, 0, 0];
  for (let f = 0; f < 60; f++) { coupling.update(); b3.b3World_Step(world, DT, 4); sim.update(DT); }
  b3.b3Body_GetPosition(p, log);
  sim.dispose();
  // no push but the water's: the log started at rest
  assert.ok(p[0] > -1.6, `log x after 1 s: ${p[0].toFixed(2)} m (start −2.2, water at 1.5 m/s)`);
  assert.ok(p[1] > 0.2 && p[1] < 0.6, `log still floating (y ${p[1].toFixed(2)})`);
});
