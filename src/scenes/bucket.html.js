// scenes/bucket.html.js — small-scale test: a 1.2 m bucket filling from a
// narrow spout at 5 cm particle spacing. Small containers amplify solver
// noise, so this is where settling and wall handling show first.
// URL params: ?pour ?quality=low|medium|high ?autoshot=N&label=x&metrics=bucket

import * as THREE from 'three';
import { createDebugHarness } from '../debug-harness.js';
import { setupAutoShots } from '../debug-shot.js';
import { createWater } from '../water/index.js';
import { attachWater, fillProbe, waterHudLine, addWaterGui, setupMetrics } from './water-harness.js';

import Box3DInit from 'box3d.js/inline';
const b3 = await Box3DInit();
const qp = new URLSearchParams(location.search);

const harness = createDebugHarness({ cameraPos: [2.6, 2.4, 3.2], target: [0, 0.6, 0] });
const { scene } = harness;

/* Box3D world: ground + bucket (interior 1.2 × 1.2 m, walls 1.8 m high) */
const worldDef = b3.b3DefaultWorldDef();
worldDef.gravity = [0, -9.81, 0];
const world = b3.b3CreateWorld(worldDef);

const BW = 0.6, BT = 0.24, BH = 1.8;
const wallMat = new THREE.MeshStandardMaterial({ color: 0x4a5058, roughness: 0.6, metalness: 0.1 });
function staticBox(x, y, z, ex, ey, ez, visible = true) {
  const def = b3.b3DefaultBodyDef();
  def.position = [x, y, z];
  b3.b3CreateBoxShape(b3.b3CreateBody(world, def), b3.b3DefaultShapeDef(), ex, ey, ez);
  if (!visible) return;
  const m = new THREE.Mesh(new THREE.BoxGeometry(ex * 2, ey * 2, ez * 2), wallMat);
  m.position.set(x, y, z);
  scene.add(m);
}
staticBox(0, -0.25, 0, 6, 0.25, 6, false);
staticBox(-(BW + BT / 2), BH / 2, 0, BT / 2, BH / 2, BW + BT);
staticBox(BW + BT / 2, BH / 2, 0, BT / 2, BH / 2, BW + BT);
staticBox(0, BH / 2, -(BW + BT / 2), BW + BT, BH / 2, BT / 2);
staticBox(0, BH / 2, BW + BT / 2, BW + BT, BH / 2, BT / 2);

/* water: 5 cm particles for the small container */
const spacing = { low: 0.07, medium: 0.05, high: 0.04 }[qp.get('quality') ?? 'medium'] ?? 0.05;
const water = await createWater({
  renderer: harness.renderer, scene, b3, world,
  quality: qp.get('quality') ?? 'medium',
  render: qp.get('render') ?? 'screen',
  params: { spacing, maxParticles: 32768, bounds: { min: [-3.5, -0.5, -3.5], max: [3.5, 4.5, 3.5] } },
});
const probe = fillProbe(water, { min: [-BW, 0, -BW], max: [BW, BH, BW] });

/* spout above the bucket */
const SPOUT_Y = 2.2;
const spoutMesh = new THREE.Mesh(
  new THREE.CylinderGeometry(0.06, 0.06, 0.5, 12),
  new THREE.MeshStandardMaterial({ color: 0x8a939f, roughness: 0.4, metalness: 0.6 }),
);
spoutMesh.position.set(0, SPOUT_Y + 0.25, 0);
scene.add(spoutMesh);
const spout = water.addSource({ position: [0, SPOUT_Y, 0], direction: [0, -1, 0], radius: 0.06, speed: 2, enabled: qp.has('pour') });

const f = harness.gui.addFolder('🪣 Bucket Flow');
f.add(spout, 'enabled').name('spout on/off');
f.add(spout, 'speed', 0.5, 6, 0.1).name('spout speed (m/s)');
f.add({ fill: () => water.fillBox([-BW, 0, -BW], [BW, 0.6, BW]) }, 'fill').name('fill 0.6 m');
f.add({ clear: () => water.reset() }, 'clear').name('clear');
addWaterGui(harness.gui, water);

/* HUD + metrics */
let probeData = probe.measure();
let hudTick = 0;
const num = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '—');
harness.setHudProvider(() => [
  `<b>Bucket Lab</b> — 1.2 m container, spacing ${water.params.spacing} m`,
  waterHudLine(harness, water),
  `fill: <b>${probeData.count}</b> particles  level <b>${num(probeData.level, 2)}</b> m (${(Math.max(0, probeData.level || 0) / BH * 100).toFixed(0)}%)`,
  `surface σ <b class="${probeData.flatness < 0.03 ? 'ok' : ''}">${num(probeData.flatness)}</b> m  (< 0.03 = settled)`,
  `leaked: <b>${water.stats.leaked ?? 0}</b>  KE ${num(water.stats.kineticEnergy, 1)} J`,
]);
setupMetrics(() => ({
  scene: 'bucket', particles: water.count, stepMs: water.stats.stepMs, fill: probeData,
  leaked: water.stats.leaked ?? 0, kineticEnergy: water.stats.kineticEnergy,
}));
setupAutoShots(harness.renderer, 4);

attachWater(harness, water, { b3, world, after() { if (++hudTick % 15 === 0) probeData = probe.measure(); } });
water.fillBox([-BW, 0, -BW], [BW, 0.3, BW]);

window.__dbg = { b3, world, water, probe, harness };
window.pushDbg?.('Bucket Lab ready');
harness.tidyGui();
harness.start();
