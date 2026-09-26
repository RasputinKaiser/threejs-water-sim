// main.js — Box3D Debug World · Pool Lab
//
// A pool built from Box3D static bodies, filled by the water pack. The pool
// walls, the ramp and the balls reach the fluid through the Box3D coupling
// (no hand-written collider list): anything in the Box3D world is solid water
// can hit, and dynamic bodies float, sink and get pushed around.

import { createDebugHarness } from './debug-harness.js';
import { setupAutoShots } from './debug-shot.js';
import * as THREE from 'three';
import { createPhysicsSync, createDebugDraw } from './box3d-debug.js';
import { createWater } from './water/index.js';
import { attachWater, fillProbe, waterHudLine, addWaterGui } from './scenes/water-harness.js';

import Box3DInit from 'box3d.js/inline';

const qp = new URLSearchParams(location.search);
const b3 = await Box3DInit();
window.pushDbg?.('Box3D WASM loaded');

const harness = createDebugHarness({ cameraPos: [9, 7, 11], target: [0, 1, 0] });
const { scene } = harness;

/* ================= physics world (Box3D) ================= */

const worldDef = b3.b3DefaultWorldDef();
worldDef.gravity = [0, -9.81, 0];
const world = b3.b3CreateWorld(worldDef);

// Pool interior: x, z ∈ [-2.5, 2.5], floor at y = 0, wall height 2.2
const WALL_H = 2.2;
const WALL_T = 0.25;
const POOL_HALF = 2.5;

function addStaticBox(x, y, z, ex, ey, ez, rotation) {
  const def = b3.b3DefaultBodyDef();
  def.position = [x, y, z];
  if (rotation) def.rotation = rotation;
  const body = b3.b3CreateBody(world, def);
  b3.b3CreateBoxShape(body, b3.b3DefaultShapeDef(), ex, ey, ez);
}

addStaticBox(0, -0.25, 0, 12, 0.25, 12);
addStaticBox(-(POOL_HALF + WALL_T / 2), WALL_H / 2, 0, WALL_T / 2, WALL_H / 2, POOL_HALF + WALL_T);
addStaticBox(POOL_HALF + WALL_T / 2, WALL_H / 2, 0, WALL_T / 2, WALL_H / 2, POOL_HALF + WALL_T);
addStaticBox(0, WALL_H / 2, -(POOL_HALF + WALL_T / 2), POOL_HALF + WALL_T, WALL_H / 2, WALL_T / 2);
addStaticBox(0, WALL_H / 2, POOL_HALF + WALL_T / 2, POOL_HALF + WALL_T, WALL_H / 2, WALL_T / 2);
// a ramp leaning on the +x wall: water poured on it runs down into the pool
addStaticBox(4.2, 1.2, 0, 1.6, 0.1, 1.0, [0, 0, Math.sin(Math.PI / 12), Math.cos(Math.PI / 12)]);

function dropBalls() {
  const densities = [250, 500, 800, 2000];
  for (let i = 0; i < 4; i++) {
    const d = b3.b3DefaultBodyDef();
    d.type = b3.b3BodyType.b3_dynamicBody;
    d.position = [-1.2 + i * 0.8, 3 + i * 0.5, 1.5]; // beside the dropped water block, not inside it
    d.angularDamping = 0.2;
    const ball = b3.b3CreateBody(world, d);
    const sd = b3.b3DefaultShapeDef();
    sd.density = densities[i];
    b3.b3CreateSphereShape(ball, sd, { center: [0, 0, 0], radius: 0.3 });
  }
}
function dropCrate() {
  const d = b3.b3DefaultBodyDef();
  d.type = b3.b3BodyType.b3_dynamicBody;
  d.position = [(Math.random() - 0.5) * 2, 3.5, (Math.random() - 0.5) * 2];
  d.rotation = [0.2, 0.1, 0, Math.sqrt(1 - 0.05)];
  d.angularDamping = 0.2;
  const body = b3.b3CreateBody(world, d);
  const sd = b3.b3DefaultShapeDef();
  sd.density = 450;
  b3.b3CreateBoxShape(body, sd, 0.35, 0.25, 0.5);
}

const physSync = createPhysicsSync(b3, world);
scene.add(physSync.object3d);
const debugDraw = createDebugDraw(b3, world, scene);
debugDraw.addGui(harness.gui);
if (qp.get('contacts') === '1') debugDraw.state.showContacts = true;
if (qp.get('aabb') === '1') debugDraw.state.showAABBs = true;

/* ================= water ================= */

const water = await createWater({
  renderer: harness.renderer, scene, b3, world,
  quality: qp.get('quality') ?? 'medium',
  render: qp.get('render') ?? 'screen', backend: qp.get('backend') ?? 'cpu',
  // particles leaving this box are removed (and counted as leaked)
  params: { bounds: { min: [-7, -0.5, -7], max: [7, 9, 7] } },
});
const probe = fillProbe(water, { min: [-POOL_HALF, 0, -POOL_HALF], max: [POOL_HALF, WALL_H, POOL_HALF] });

const flow = {
  dropSize: 1.2, dropHeight: 3.5, pourX: 0, pourZ: 0,
};
const pour = water.addSource({
  position: [0, 4, 0], direction: [0, -1, 0], radius: 0.15, speed: 4, enabled: qp.has('pour'),
});
const PLUG = { min: [-0.35, -0.1, -0.35], max: [0.35, 0.25, 0.35] };
const plughole = water.addDrain(PLUG);
plughole.enabled = qp.get('autodrain') === '1';
{
  const plug = new THREE.Mesh(
    new THREE.CylinderGeometry(0.35, 0.35, 0.02, 24),
    new THREE.MeshStandardMaterial({ color: 0x1b1e24, roughness: 0.9 }),
  );
  plug.position.set(0, 0.01, 0);
  scene.add(plug);
}

function dropBlock() {
  const s = flow.dropSize / 2;
  water.fillBox([flow.pourX - s, flow.dropHeight, flow.pourZ - s], [flow.pourX + s, flow.dropHeight + 2 * s, flow.pourZ + s]);
}
function splash(n = 400) {
  const out = new Float32Array(n * 6);
  for (let k = 0; k < n; k++) {
    const a = Math.random() * Math.PI * 2, r = Math.sqrt(Math.random()) * 0.8;
    out.set([flow.pourX + Math.cos(a) * r, flow.dropHeight + Math.random() * 0.6, flow.pourZ + Math.sin(a) * r,
      (Math.random() - 0.5) * 3, -2 - Math.random() * 3, (Math.random() - 0.5) * 3], k * 6);
  }
  water.spawn(out);
}

const f = harness.gui.addFolder('🚰 Pool Lab Flow');
f.add(flow, 'dropSize', 0.4, 2, 0.1).name('drop size (m)');
f.add(flow, 'dropHeight', 1, 6, 0.25).name('drop height');
f.add(flow, 'pourX', -2.5, 5, 0.25).name('pour x').onChange(() => { pour.position = [flow.pourX, 4, flow.pourZ]; });
f.add(flow, 'pourZ', -2.5, 2.5, 0.25).name('pour z').onChange(() => { pour.position = [flow.pourX, 4, flow.pourZ]; });
f.add({ dropBlock }, 'dropBlock').name('💧 DROP BLOCK');
f.add({ splash: () => splash() }, 'splash').name('💦 SPLASH BURST');
f.add(pour, 'enabled').name('continuous pour');
f.add(pour, 'speed', 0.5, 8, 0.1).name('pour speed (m/s)');
f.add(plughole, 'enabled').name('🕳 plughole drain');
f.add({ dropBalls }, 'dropBalls').name('⚽ drop balls (250–2000 kg/m³)');
f.add({ dropCrate }, 'dropCrate').name('📦 drop crate (450 kg/m³)');
f.add({ clear: () => water.reset() }, 'clear').name('🗑 clear water');
f.add({ drain: () => water.removeInBox(PLUG.min, [PLUG.max[0], WALL_H, PLUG.max[2]]) }, 'drain').name('🕳 drain column now');
addWaterGui(harness.gui, water);

/* ================= HUD ================= */

let probeData = probe.measure();
let hudTick = 0;
harness.setHudProvider(() => {
  const flat = Number.isFinite(probeData.flatness) ? probeData.flatness.toFixed(3) : '—';
  const level = Number.isFinite(probeData.level) ? probeData.level.toFixed(2) : '—';
  return [
    `<b>Pool Lab</b> — Box3D world + particle water (hash grid)`,
    waterHudLine(harness, water),
    `bodies awake <b>${b3.b3World_GetAwakeBodyCount(world)}</b>  colliders <b>${water.stats.colliders ?? 0}</b>`,
    `pool: <b>${probeData.count}</b> particles  level <b>${level}</b> m`,
    `surface σ <b class="${probeData.flatness < 0.05 ? 'ok' : ''}">${flat}</b> m  (< 0.05 = settled)  leaked ${water.stats.leaked ?? 0}`,
  ];
});

/* ================= main loop ================= */

attachWater(harness, water, {
  b3, world,
  after() {
    physSync.update();
    debugDraw.update();
    if (++hudTick % 15 === 0) probeData = probe.measure();
  },
});

water.fillBox([-POOL_HALF, 0, -POOL_HALF], [POOL_HALF, 0.4, POOL_HALF]);
dropBalls();
dropBlock();

setupAutoShots(harness.renderer, 4);
window.__dbg = { b3, world, water, harness, probe };
window.pushDbg?.('Ready — press P for screenshot, Space pause, . step. Use GUI right side.');
harness.tidyGui();
harness.start();
