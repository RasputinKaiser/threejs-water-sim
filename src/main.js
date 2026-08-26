// main.js — Box3D Debug World · Water Lab
//
// A pool built from Box3D static bodies, with a PBF water sim that pours into it.
// Spawn water above the pool → it falls, splashes, and settles flat like real water.

import { createDebugHarness } from './debug-harness.js';
import { setupAutoShots } from './debug-shot.js';
import * as THREE from 'three';
import { createPhysicsSync, createDebugDraw } from './box3d-debug.js';
import { WaterSim, loadWaterLane, createScreenBridge } from './water-pack/index.js';
import { createWaterRenderer, createFillProbe } from './water-render.js';

import Box3DInit from 'box3d.js/inline';

const b3 = await Box3DInit();
window.pushDbg?.('Box3D WASM loaded');

const harness = createDebugHarness({ cameraPos: [11, 9, 14], target: [0, 1.5, 0] });
const { scene } = harness;

/* ================= physics world (Box3D) ================= */

const worldDef = b3.b3DefaultWorldDef();
worldDef.gravity = [0, -9.81, 0];
const world = b3.b3CreateWorld(worldDef);

/* ================= scene layout ================= */
// Pool: outer walls + floor made of Box3D boxes. Interior region:
//   x ∈ [-2.5, 2.5], z ∈ [-2.5, 2.5], floor at y = 0, wall height 2.2

const WALL_H = 2.2;
const WALL_T = 0.25;
const POOL_HALF = 2.5;

function addStaticBox(x, y, z, ex, ey, ez) {
  const def = b3.b3DefaultBodyDef();
  def.position = [x, y, z];
  const body = b3.b3CreateBody(world, def);
  b3.b3CreateBoxShape(body, b3.b3DefaultShapeDef(), ex, ey, ez);
}

// floor
addStaticBox(0, -0.25, 0, 12, 0.25, 12);
// pool walls
addStaticBox(-(POOL_HALF + WALL_T / 2), WALL_H / 2, 0, WALL_T / 2, WALL_H / 2, POOL_HALF + WALL_T);
addStaticBox(POOL_HALF + WALL_T / 2, WALL_H / 2, 0, WALL_T / 2, WALL_H / 2, POOL_HALF + WALL_T);
addStaticBox(0, WALL_H / 2, -(POOL_HALF + WALL_T / 2), POOL_HALF + WALL_T, WALL_H / 2, WALL_T / 2);
addStaticBox(0, WALL_H / 2, POOL_HALF + WALL_T / 2, POOL_HALF + WALL_T, WALL_H / 2, WALL_T / 2);
// a ramp + a couple of dynamic props to prove rigid bodies coexist with the fluid
(function addProps() {
  const rampDef = b3.b3DefaultBodyDef();
  rampDef.position = [5.5, 1.0, 0];
  const q = new Float32Array([Math.sin(Math.PI / 8), 0, 0, Math.cos(Math.PI / 8)]);
  rampDef.rotation = [q[0], q[1], q[2], q[3]];
  const ramp = b3.b3CreateBody(world, rampDef);
  b3.b3CreateBoxShape(ramp, b3.b3DefaultShapeDef(), 2.5, 0.15, 1.2);

  for (let i = 0; i < 4; i++) {
    const d = b3.b3DefaultBodyDef();
    d.type = b3.b3BodyType.b3_dynamicBody;
    d.position = [-6 + Math.random() * 2, 4 + i * 1.2, -1 + Math.random() * 2];
    const ball = b3.b3CreateBody(world, d);
    b3.b3CreateSphereShape(ball, b3.b3DefaultShapeDef(), { center: [0, 0, 0], radius: 0.45 });
  }
})();

const physSync = createPhysicsSync(b3, world);
scene.add(physSync.object3d);
const debugDraw = createDebugDraw(b3, world, scene);
debugDraw.addGui(harness.gui);

/* ================= water sim colliders ================= */
// Static collider list mirroring the pool geometry (PBF particles collide here).
function boxCollider(c, e) {
  // identity rotation → matrix terms
  return {
    type: 'box', c, e,
    m00: 1, m01: 0, m02: 0,
    m10: 0, m11: 1, m12: 0,
    m20: 0, m21: 0, m22: 1,
  };
}
const colliders = [
  { type: 'plane', o: [0, 0, 0], n: [0, 1, 0] },                       // ground plane y=0
  boxCollider([-(POOL_HALF + WALL_T / 2), WALL_H / 2, 0], [WALL_T / 2, WALL_H, POOL_HALF + WALL_T]),
  boxCollider([POOL_HALF + WALL_T / 2, WALL_H / 2, 0], [WALL_T / 2, WALL_H, POOL_HALF + WALL_T]),
  boxCollider([0, WALL_H / 2, -(POOL_HALF + WALL_T / 2)], [POOL_HALF + WALL_T, WALL_H, WALL_T / 2]),
  boxCollider([0, WALL_H / 2, POOL_HALF + WALL_T / 2], [POOL_HALF + WALL_T, WALL_H, WALL_T / 2]),
];

const sim = new WaterSim();
const bounds = {
  min: [-7, -0.5, -7],
  size: [14, 9, 14],
};
// pack solver needs explicit leak bounds (killLeaks removes escapers + counts them)
sim.bounds = { min: bounds.min, max: [bounds.min[0] + bounds.size[0], bounds.min[1] + bounds.size[1], bounds.min[2] + bounds.size[2]] };
const waterRender = createWaterRenderer(sim, scene, bounds);
waterRender.addGui(harness.gui);
const probe = createFillProbe(sim, { min: [-POOL_HALF, 0, -POOL_HALF], max: [POOL_HALF, WALL_H, POOL_HALF] });

/* Lane C2: optional screen-space fluid render mode + FoamSystem.
 * Pool Lab defaults are smaller than Big Pool's (budget 300 foam points). */
const screenBridge = createScreenBridge(sim, {
  renderer: harness.renderer,
  scene,
  bounds: { min: sim.bounds.min.slice(), max: sim.bounds.max.slice() },
  surfaceGroup: waterRender.group, // hidden while screen composites
});
screenBridge.setSink(harness); // lets setMode install its composite pass
Promise.resolve(screenBridge.setMode('auto')).catch((e) =>
  window.pushDbg?.(`[pool-lab] screen mode init failed: ${e?.message ?? e}`));

const FOAM_BUDGET = 300;
const FOAM_FLAGS = { maxSpeed: 4.5, minNeighbors: 12 };
let foam = null;
loadWaterLane('effects').then((fx) => {
  if (!fx?.FoamSystem) { window.pushDbg?.('[pool-lab] effects.js/FoamSystem not available — skipping'); return; }
  try {
    foam = new fx.FoamSystem(sim, FOAM_BUDGET);
    foam.addGui?.(harness.gui);
    harness.tidyGui(); // keep late-added subfolders collapsed
    window.pushDbg?.(`[pool-lab] FoamSystem ready (budget ${FOAM_BUDGET})`);
  } catch (e) {
    window.pushDbg?.(`[pool-lab] FoamSystem init failed: ${e?.message ?? e}`);
    foam = null;
  }
}).catch((e) => window.pushDbg?.(`[pool-lab] effects lane load failed: ${e?.message ?? e}`));

// URL param overrides for agent-driven runs: ?mode=particles&autoshot=3
{
  const qp = new URLSearchParams(location.search);
  if (qp.get('mode') === 'particles') waterRender.state.mode = 'particles';
}

/* ================= spawn controls ================= */

const spawnState = {
  blockNx: 12, blockNy: 10, blockNz: 12,
  dropHeight: 7,
  continuous: false, continuousRate: 220, // particles/sec
  autoDrain: false,
  pourX: 0, pourZ: 0,
};

// deferred URL-param hooks (spawnState declared above)
{
  const qp = new URLSearchParams(location.search);
  if (qp.get('autodrain') === '1') spawnState.autoDrain = true;
  if (qp.get('pour') != null) {
    spawnState.continuous = true;
    if (qp.get('pour')) spawnState.continuousRate = parseFloat(qp.get('pour'));
  }
  if (qp.get('contacts') === '1') debugDraw.state.showContacts = true;
  if (qp.get('aabb') === '1') debugDraw.state.showAABBs = true;
  if (qp.get('vel') === '1') debugDraw.state.showVelocities = true;
  window.__dropBalls = () => {
    for (let i = 0; i < 4; i++) {
      const d = b3.b3DefaultBodyDef();
      d.type = b3.b3BodyType.b3_dynamicBody;
      d.position = [-6 + Math.random() * 2, 4 + i * 1.2, -1 + Math.random() * 2];
      const ball = b3.b3CreateBody(world, d);
      b3.b3CreateSphereShape(ball, b3.b3DefaultShapeDef(), { center: [0, 0, 0], radius: 0.45 });
    }
  };
}

function spawnBlock() {
  sim.spawnBlock(spawnState.pourX, spawnState.dropHeight, spawnState.pourZ,
    spawnState.blockNx, spawnState.blockNy, spawnState.blockNz);
}
function spawnSplash(n = 300) {
  for (let k = 0; k < n; k++) {
    const a = Math.random() * Math.PI * 2, r = Math.random() * 1.6;
    sim.spawn(spawnState.pourX + Math.cos(a) * r, spawnState.dropHeight + Math.random() * 1.5,
      spawnState.pourZ + Math.sin(a) * r, (Math.random() - 0.5) * 3, -2 - Math.random() * 3, (Math.random() - 0.5) * 3);
  }
}

let emitAcc = 0;

const waterFolder = harness.gui.addFolder('🚰 Pool Lab Flow');
waterFolder.add(spawnState, 'blockNx', 2, 24, 1).name('drop N×');
waterFolder.add(spawnState, 'blockNy', 2, 24, 1).name('drop N×Y');
waterFolder.add(spawnState, 'blockNz', 2, 24, 1).name('drop N×Z');
waterFolder.add(spawnState, 'dropHeight', 3, 12, 0.5).name('drop height');
waterFolder.add(spawnState, 'pourX', -8, 8, 0.25).name('pour x');
waterFolder.add(spawnState, 'pourZ', -8, 8, 0.25).name('pour z');
waterFolder.add({ drop: spawnBlock }, 'drop').name('💧 DROP BLOCK');
waterFolder.add({ splash: () => spawnSplash() }, 'splash').name('💦 SPLASH BURST');
waterFolder.add(spawnState, 'continuous').name('continuous pour');
waterFolder.add(spawnState, 'continuousRate', 40, 800, 10).name('pour rate /s');
waterFolder.add(spawnState, 'autoDrain').name('🕳 auto-drain plughole');
const renderModeCtrl = { renderMode: 'auto' };
waterFolder.add(renderModeCtrl, 'renderMode', ['auto', 'metaballs', 'screen'])
  .name('render mode')
  .onChange((m) => Promise.resolve(screenBridge.setMode(m)).catch(
    (e) => window.pushDbg?.(`[pool-lab] setRenderMode('${m}') failed: ${e?.message ?? e}`)));
waterFolder.add({ clear: () => sim.reset() }, 'clear').name('🗑 clear water');
waterFolder.add({ drain: () => sim.drain(drainRegion) }, 'drain').name('🕳 drain now');
waterFolder.close();

const paramsFolder = waterFolder.addFolder('Fluid parameters');
for (const key of ['h', 'restDensity', 'stiffness', 'nearStiffness', 'viscositySigma', 'viscosityBeta']) {
  paramsFolder.add(sim.p, key).listen().onFinishChange(() => window.pushDbg?.(`param ${key} → ${sim.p[key]}`));
}
paramsFolder.close();

// drain plughole region (center of pool floor) + visual marker
const drainRegion = { min: [-0.35, -0.1, -0.35], max: [0.35, 0.25, 0.35] };
{
  const plug = new THREE.Mesh(
    new THREE.CylinderGeometry(0.35, 0.35, 0.06, 24),
    new THREE.MeshStandardMaterial({ color: 0x1b1e24, roughness: 0.9 }),
  );
  plug.position.set(0, 0.03, 0);
  scene.add(plug);
}

/* ================= HUD ================= */
// Line structure matches the other lab scenes (see scenes/creek.html.js):
//   1 title · 2 fps/sim/particles · 3+ scene-specific metrics · extras last.

let probeData = { count: 0, meanY: NaN, stdY: NaN, topY: NaN };
let hudTick = 0;

harness.setHudProvider(() => {
  const awake = b3.b3World_GetAwakeBodyCount(world);
  const flat = Number.isFinite(probeData.stdY) ? probeData.stdY.toFixed(3) : '—';
  const level = Number.isFinite(probeData.meanY) ? probeData.meanY.toFixed(3) : '—';
  return [
    `<b>Pool Lab</b> — Box3D world + PBF water`,
    `fps <b>${harness.fps.toFixed(0)}</b>  sim ${sim.simMs.toFixed(1)}ms  particles <b>${sim.count}</b>`,
    `bodies awake <b>${awake}</b>`,
    `pool fill: <b>${probeData.count}</b> pts  level <b>${level}</b>m`,
    `flatness σy: <b class="${probeData.stdY < 0.06 ? 'ok' : ''}">${flat}</b>  (< 0.06 = settled)`,
    `MC tris: <b>${waterRender.mcTris}</b>  mode ${waterRender.state.mode}`,
    `leaked: n/a  render <b>${screenBridge.state.active ? 'screen' : 'metaballs'}</b>` +
      (screenBridge.state.error ? ` <span class="warn">(${screenBridge.state.error})</span>` : '') +
      `  foam <b>${foam ? (foam.count ?? foam.points?.count ?? 0) : 'n/a'}</b>` +
      `  screen ${screenBridge.state.screenMs != null ? screenBridge.state.screenMs.toFixed(1) + 'ms' : '—'}`,
  ];
});

/* ================= main loop ================= */

harness.onFixed((dt) => {
  if (spawnState.continuous) {
    emitAcc += spawnState.continuousRate * dt;
    while (emitAcc >= 1 && sim.count < sim.p.maxParticles) {
      const a = Math.random() * Math.PI * 2, r = Math.random() * 0.35;
      sim.spawn(spawnState.pourX + Math.cos(a) * r, spawnState.dropHeight + Math.random() * 0.4,
        spawnState.pourZ + Math.sin(a) * r);
      emitAcc--;
    }
  }

  sim.step(dt, colliders);
  try { foam?.update?.(dt, FOAM_FLAGS); } catch (e) { window.pushDbg?.(`[pool-lab] foam update failed: ${e?.message ?? e}`); }
  if (spawnState.autoDrain) sim.drain(drainRegion);
  b3.b3World_Step(world, dt, 4);
  physSync.update();
  debugDraw.update();
  waterRender.update();

  if (++hudTick % 10 === 0) {
    probeData = probe.measure();
  }
});

// initial demo: one drop so something is happening immediately
spawnBlock();

setupAutoShots(harness.renderer, 4);

window.__dbg = { b3, world, sim, harness, probe }; // console access for me
window.pushDbg?.('Ready — press P for screenshot, Space pause, . step. Use GUI right side.');
harness.tidyGui(); // all param subfolders (Fluid parameters, Water Render, Debug Draw…) collapsed

harness.start();
