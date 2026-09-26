// scenes/bucket.html.js — small-scale test: a bucket (0.9m wide) filling from a
// narrow spout. The pack's hardest test: tiny containers amplify solver noise.
// URL params: ?autoshot=N&label=x&metrics=bucket&pour=120

import * as THREE from 'three';
import { createDebugHarness } from '../debug-harness.js';
import { setupAutoShots } from '../debug-shot.js';
import { createWaterPack, createFillProbe, setupMetrics } from '../water-pack/index.js';

import Box3DInit from 'box3d.js/inline';
const b3 = await Box3DInit();

const harness = createDebugHarness({ cameraPos: [2.6, 2.2, 3.2], target: [0, 0.5, 0] });
const { scene } = harness;

/* Box3D world: bucket walls + floor */
const worldDef = b3.b3DefaultWorldDef();
worldDef.gravity = [0, -9.81, 0];
const world = b3.b3CreateWorld(worldDef);

function staticBox(x, y, z, ex, ey, ez) {
  const def = b3.b3DefaultBodyDef();
  def.position = [x, y, z];
  const body = b3.b3CreateBody(world, def);
  b3.b3CreateBoxShape(body, b3.b3DefaultShapeDef(), ex, ey, ez);
}

// ground + bucket (interior 0.9 × 0.9 m, walls 0.8 high, 6cm thick)
staticBox(0, -0.25, 0, 6, 0.25, 6);
const BW = 0.6, BT = 0.24, BH = 1.8;
staticBox(-(BW + BT / 2), BH / 2, 0, BT / 2, BH / 2, BW + BT);
staticBox(BW + BT / 2, BH / 2, 0, BT / 2, BH / 2, BW + BT);
staticBox(0, BH / 2, -(BW + BT / 2), BW + BT, BH / 2, BT / 2);
staticBox(0, BH / 2, BW + BT / 2, BW + BT, BH / 2, BT / 2);

// visible walls (this scene doesn't render Box3D bodies directly)
{
  const wallMat = new THREE.MeshStandardMaterial({ color: 0x4a5058, roughness: 0.6, metalness: 0.1 });
  const mk = (x, z, sx, sz) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(sx, BH, sz), wallMat);
    m.position.set(x, BH / 2, z); m.castShadow = m.receiveShadow = true; scene.add(m);
  };
  mk(-(BW + BT / 2), 0, BT, (BW + BT) * 2);
  mk(BW + BT / 2, 0, BT, (BW + BT) * 2);
  mk(0, -(BW + BT / 2), (BW + BT) * 2, BT);
  mk(0, BW + BT / 2, (BW + BT) * 2, BT);
}

/* water pack — smaller h for the small container */
const bounds = { min: [-3.5, -0.5, -3.5], size: [7, 5, 7] };
const pack = createWaterPack({
  scene, bounds,
  surfaceBounds: { min: [-2, -0.5, -2], size: [4, 3.5, 4] },
  params: { h: 0.28, stiffness: 10, nearStiffness: 30, viscositySigma: 4, viscosityBeta: 1, maxParticles: 5000 },
  substeps: 2,
  gui: harness.gui,
  waterline: { min: [-BW, 0, -BW], max: [BW, BH, BW] },
  renderer: harness.renderer,
  renderMode: 'metaballs', // default stays metaballs here; switch via GUI dropdown
});
pack.attachCompositor(harness);

const probe = createFillProbe(pack.sim, { min: [-BW, 0, -BW], max: [BW, BH, BW] });

/* spout above the bucket */
const spout = new THREE.Mesh(
  new THREE.CylinderGeometry(0.06, 0.06, 0.5, 12),
  new THREE.MeshStandardMaterial({ color: 0x8a939f, roughness: 0.4, metalness: 0.6 }),
);
spout.position.set(0, 2.2, 0);
scene.add(spout);

/* controls */
const ctrl = { pour: false, rate: 60, dropY: 2.0, clear: () => pack.sim.reset() };
{
  const qp = new URLSearchParams(location.search);
  if (qp.get('pour') != null) ctrl.pour = true;
}
const f = harness.gui.addFolder('🪣 Bucket Flow');
f.add(ctrl, 'pour').name('spout on/off');
f.add(ctrl, 'rate', 30, 400, 10).name('rate /s');
f.add(ctrl, 'dropY', 1.2, 3, 0.1).name('spout height');
const renderModeCtrl = { renderMode: 'metaballs' };
f.add(renderModeCtrl, 'renderMode', ['auto', 'metaballs', 'screen'])
  .name('render mode')
  .onChange((m) => Promise.resolve(pack.setRenderMode(m)).catch(
    (e) => window.pushDbg?.(`[bucket] setRenderMode('${m}') failed: ${e?.message ?? e}`)));
f.add(ctrl, 'clear').name('clear');
f.close();

/* HUD + metrics */
let probeData = { count: 0, meanY: NaN, stdY: NaN, fillPct: 0 };
let hudTick = 0;
harness.setHudProvider(() => [
  `<b>Bucket Lab</b> — 1.2m container, h=0.26`,
  `fps <b>${harness.fps.toFixed(0)}</b>  sim ${pack.sim.simMs.toFixed(1)}ms  particles <b>${pack.sim.count}</b>`,
  `fill: <b>${probeData.count}</b> pts  level <b>${Number.isFinite(probeData.meanY) ? probeData.meanY.toFixed(3) : '—'}</b>m (${probeData.fillPct.toFixed(0)}%)`,
  `flatness σy: <b class="${probeData.stdY < 0.045 ? 'ok' : ''}">${Number.isFinite(probeData.stdY) ? probeData.stdY.toFixed(3) : '—'}</b>  (< 0.045 = settled)`,
  `leaked: <b>${pack.sim.leakedTotal ?? 0}</b>  KE: ${(pack.sim.kineticEnergy ?? 0).toFixed(0)}`,
  `render <b>${pack.screenState.active ? 'screen' : 'metaballs'}</b>` +
    (pack.screenState.error ? ` <span class="warn">(${pack.screenState.error})</span>` : ''),
]);

setupMetrics(() => ({
  scene: 'bucket', particles: pack.sim.count, simMs: +pack.sim.simMs.toFixed(2),
  fill: probeData, leaked: pack.sim.leakedTotal ?? 0,
  kineticEnergy: +(pack.sim.kineticEnergy ?? 0).toFixed(1),
}));

setupAutoShots(harness.renderer, 4);

let emitAcc = 0;
harness.onFixed((dt) => {
  if (ctrl.pour) {
    emitAcc += ctrl.rate * dt;
    const s = pack.sim.h * 0.55;
    while (emitAcc >= 1 && pack.sim.count < pack.sim.p.maxParticles) {
      pack.sim.spawn((Math.random() - 0.5) * 0.06, ctrl.dropY, (Math.random() - 0.5) * 0.06,
        (Math.random() - 0.5) * 0.2, -1, (Math.random() - 0.5) * 0.2);
      emitAcc--;
    }
  }
  pack.step(dt, [
    { type: 'plane', o: [0, 0, 0], n: [0, 1, 0] },
    { type: 'box', c: [-(BW + BT / 2), BH / 2, 0], e: [BT / 2, BH / 2, BW + BT] },
    { type: 'box', c: [BW + BT / 2, BH / 2, 0], e: [BT / 2, BH / 2, BW + BT] },
    { type: 'box', c: [0, BH / 2, -(BW + BT / 2)], e: [BW + BT, BH / 2, BT / 2] },
    { type: 'box', c: [0, BH / 2, BW + BT / 2], e: [BW + BT, BH / 2, BT / 2] },
  ]);
  b3.b3World_Step(world, dt, 4);
  if (++hudTick % 10 === 0) probeData = probe.measure();
});

window.__dbg = { b3, world, pack, probe, harness };
window.pushDbg?.('Bucket Lab ready');
harness.tidyGui(); // all param subfolders collapsed by default
harness.start();
