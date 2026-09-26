// scenes/pool.html.js — regulation lap pool: 25m × 12.5m interior, 2m walls.
// Scale test for the water pack at large particle counts (~25k at 25 cm
// spacing). No Box3D here: the pool is plain solver colliders.
// URL params: ?fill=0 (start empty), ?pour, ?quality=low|medium|high,
// ?autoshot=N&label=x&metrics=bigpool

import * as THREE from 'three';
import { createDebugHarness } from '../debug-harness.js';
import { setupAutoShots } from '../debug-shot.js';
import { createWater } from '../water/index.js';
import { attachWater, fillProbe, waterHudLine, addWaterGui, setupMetrics } from './water-harness.js';

const qp = new URLSearchParams(location.search);
const harness = createDebugHarness({ cameraPos: [26, 18, 30], target: [0, 0.8, 0] });
const { scene } = harness;

/* pool geometry — interior 25m (x) × 12.5m (z), walls 2m tall, 0.5m thick */
const PL = 25, PW = 12.5, WH = 2, WT = 0.5;
const HX = PL / 2, HZ = PW / 2;

/* visible shell: gray walls + darker pool floor + deck ring */
{
  const wallMat = new THREE.MeshStandardMaterial({ color: 0x6b7280, roughness: 0.7, metalness: 0.05 });
  const mkWall = (x, z, sx, sz) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(sx, WH, sz), wallMat);
    m.position.set(x, WH / 2, z); m.castShadow = m.receiveShadow = true; scene.add(m);
  };
  mkWall(-(HX + WT / 2), 0, WT, PW + 2 * WT);
  mkWall(HX + WT / 2, 0, WT, PW + 2 * WT);
  mkWall(0, -(HZ + WT / 2), PL + 2 * WT, WT);
  mkWall(0, HZ + WT / 2, PL + 2 * WT, WT);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(PL, PW),
    new THREE.MeshStandardMaterial({ color: 0x3d5a73, roughness: 0.9 }),
  );
  floor.rotation.x = -Math.PI / 2; floor.position.y = 0.001; floor.receiveShadow = true;
  scene.add(floor);

  // lane lines: thin white stripes every 2.5m across the width, running the length
  const lineMat = new THREE.MeshStandardMaterial({ color: 0xf5f5f5, roughness: 0.8 });
  for (let z = -HZ + 2.5; z <= HZ - 2.5; z += 2.5) {
    const line = new THREE.Mesh(new THREE.BoxGeometry(PL - 1, 0.02, 0.15), lineMat);
    line.position.set(0, 0.012, z); scene.add(line);
  }

  // deck: large flat gray plane around the whole pool
  const deck = new THREE.Mesh(
    new THREE.PlaneGeometry(PL + 24, PW + 24),
    new THREE.MeshStandardMaterial({ color: 0x9aa0a6, roughness: 0.95 }),
  );
  deck.rotation.x = -Math.PI / 2; deck.position.y = -0.02; deck.receiveShadow = true;
  scene.add(deck);
}

/* water — 25 cm particles keep the count sane at this volume */
const quality = qp.get('quality') ?? 'medium';
const spacing = { low: 0.33, medium: 0.25, high: 0.2 }[quality] ?? 0.25;
const water = await createWater({
  renderer: harness.renderer, scene, quality,
  render: qp.get('render') ?? 'screen',
  params: { spacing, maxParticles: 65536, bounds: { min: [-16, -1, -10], max: [16, 8, 10] } },
  colliders: [
    { type: 'plane', position: [0, 0, 0] },
    { type: 'box', position: [-(HX + WT / 2), WH / 2, 0], size: [WT / 2, WH / 2, HZ + WT] },
    { type: 'box', position: [HX + WT / 2, WH / 2, 0], size: [WT / 2, WH / 2, HZ + WT] },
    { type: 'box', position: [0, WH / 2, -(HZ + WT / 2)], size: [HX + WT, WH / 2, WT / 2] },
    { type: 'box', position: [0, WH / 2, HZ + WT / 2], size: [HX + WT, WH / 2, WT / 2] },
  ],
});
const probe = fillProbe(water, { min: [-HX, 0, -HZ], max: [HX, WH, HZ] });

const FILL_DEPTH = 1.25; // 5 layers of 25 cm
const fill = () => water.fillBox([-HX, 0, -HZ], [HX, FILL_DEPTH, HZ]);
if (qp.get('fill') !== '0') fill();

/* a wide inlet over the shallow (-x) end */
const inlet = water.addSource({ position: [-11, WH + 0.8, 0], direction: [0.3, -1, 0], radius: 0.6, speed: 3, enabled: qp.has('pour') });
const f = harness.gui.addFolder('🤽 Big Pool Flow');
f.add({ fill }, 'fill').name(`fill to ${FILL_DEPTH} m`);
f.add({ wave: () => water.fillBox([-HX, FILL_DEPTH, -HZ], [-HX + 2, FILL_DEPTH + 1.5, HZ], { velocity: [3, 0, 0] }) }, 'wave').name('🌊 wave from the shallow end');
f.add(inlet, 'enabled').name('inlet on/off');
f.add(inlet, 'speed', 0.5, 8, 0.1).name('inlet speed (m/s)');
f.add({ clear: () => water.reset() }, 'clear').name('clear');
addWaterGui(harness.gui, water);

/* HUD + metrics */
let probeData = probe.measure();
let hudTick = 0;
const num = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '—');
harness.setHudProvider(() => [
  `<b>Big Pool</b> — 25×12.5×2 m, spacing ${water.params.spacing} m`,
  waterHudLine(harness, water),
  `fill: <b>${probeData.count}</b> particles  level <b>${num(probeData.level, 2)}</b> m  surface σ <b>${num(probeData.flatness)}</b> m`,
  `leaked: <b>${water.stats.leaked ?? 0}</b>  KE ${num(water.stats.kineticEnergy, 0)} J`,
]);
setupMetrics(() => ({
  scene: 'bigpool', particles: water.count, stepMs: water.stats.stepMs, fill: probeData,
  leaked: water.stats.leaked ?? 0, kineticEnergy: water.stats.kineticEnergy,
}));
setupAutoShots(harness.renderer, 6);

attachWater(harness, water, { after() { if (++hudTick % 15 === 0) probeData = probe.measure(); } });

window.__dbg = { water, probe, harness };
window.pushDbg?.('Big Pool ready');
harness.tidyGui();
harness.start();
