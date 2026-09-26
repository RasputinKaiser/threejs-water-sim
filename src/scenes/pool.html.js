// scenes/pool.html.js — regulation lap pool: 25m × 12.5m interior, 2m walls.
// Scale test for the water pack at big-particle counts (~16-20k). Half-filled
// competition pool (water to ~1.1m) with lane-line stripes on the floor.
// URL params: ?fillblock=1 (stacked spawnBlock fill), ?pour=N (rain at N/s
// into the shallow end), ?autoshot=N&label=x&metrics=bigpool

import * as THREE from 'three';
import { createDebugHarness } from '../debug-harness.js';
import { setupAutoShots } from '../debug-shot.js';
import { createWaterPack, createFillProbe, setupMetrics, loadWaterLane } from '../water-pack/index.js';

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

/* water pack — h=0.5 keeps the particle budget sane at this volume */
const bounds = { min: [-15, -1, -9], size: [30, 7, 18] };
const waterlineRegion = { min: [-HX, 0, -HZ], max: [HX, WH, HZ] };
const pack = createWaterPack({
  scene, bounds,
  surfaceBounds: { min: [-13, 0, -6.5], size: [26, 2.5, 13] },
  params: { h: 0.5, stiffness: 10, nearStiffness: 30, viscositySigma: 4, viscosityBeta: 1, maxParticles: 22000 },
  substeps: 1,
  gui: harness.gui,
  waterline: waterlineRegion,
  renderer: harness.renderer,
  renderMode: 'auto', // screen-space fluid when the lane is available, else metaballs
});
pack.attachCompositor(harness); // lets setRenderMode install its composite pass

/* Lane C2: optional FoamSystem from water-pack/effects.js (concurrent lane).
 * Rule flags: speed > 4.5 || neighbors < 12; budget ~800 points/frame. */
const FOAM_BUDGET = 800;
const FOAM_FLAGS = { maxSpeed: 4.5, minNeighbors: 12 };
let foam = null;
loadWaterLane('effects').then((fx) => {
  if (!fx?.FoamSystem) { window.pushDbg?.('[bigpool] effects.js/FoamSystem not available — skipping'); return; }
  try {
    foam = new fx.FoamSystem(pack.sim, FOAM_BUDGET);
    foam.addGui?.(harness.gui);
    harness.tidyGui(); // keep late-added subfolders collapsed
    window.pushDbg?.(`[bigpool] FoamSystem ready (budget ${FOAM_BUDGET})`);
  } catch (e) {
    window.pushDbg?.(`[bigpool] FoamSystem init failed: ${e?.message ?? e}`);
    foam = null;
  }
}).catch((e) => window.pushDbg?.(`[bigpool] effects lane load failed: ${e?.message ?? e}`));

const probe = createFillProbe(pack.sim, waterlineRegion);

/* colliders: ground plane + four thick wall boxes */
/* fill: stacked spawnBlock layers covering the floor in tiles (< ~1500 each).
 * NOTE: the solver compacts ~2x below spawn-spacing volume as it settles, so we
 * spawn past the visual target depth to actually reach the 22k particle cap. */
const FILL_DEPTH = 1.6;
const colliders = [
  { type: 'plane', o: [0, 0, 0], n: [0, 1, 0] },
  { type: 'box', c: [-(HX + WT / 2), WH / 2, 0], e: [WT / 2, WH / 2, HZ + WT] },
  { type: 'box', c: [HX + WT / 2, WH / 2, 0], e: [WT / 2, WH / 2, HZ + WT] },
  { type: 'box', c: [0, WH / 2, -(HZ + WT / 2)], e: [HX + WT, WH / 2, WT / 2] },
  { type: 'box', c: [0, WH / 2, HZ + WT / 2], e: [HX + WT, WH / 2, WT / 2] },
];

/* fill: stacked spawnBlock layers covering the floor in tiles (< ~1500 each) */
function fillBlock() {
  const s = pack.sim.h * 0.55;
  const ny = Math.floor(FILL_DEPTH / s);
  const TX = 4, TZ = 4;
  let placed = 0;
  for (let x0 = -HX; x0 < HX - 1e-6; x0 += TX) {
    const tw = Math.min(TX, HX - x0);
    const nx = Math.max(1, Math.round(tw / s));
    for (let z0 = -HZ; z0 < HZ - 1e-6; z0 += TZ) {
      const td = Math.min(TZ, HZ - z0);
      const nz = Math.max(1, Math.round(td / s));
      for (let iy = 0; iy < ny; iy++) {
        const y = s * 0.5 + iy * s;
        pack.sim.spawnBlock(x0 + tw / 2, y, z0 + td / 2, nx, 1, nz);
        placed += nx * nz;
      }
    }
  }
  return placed;
}

/* controls */
const ctrl = { fillBlock, pour: false, rate: 300, clear: () => pack.sim.reset() };
{
  const qp = new URLSearchParams(location.search);
  if (qp.get('fillblock') != null && qp.get('fillblock') !== '0') ctrl.fillBlock();
  if (qp.get('pour') != null) {
    ctrl.pour = true;
    const r = parseFloat(qp.get('pour'));
    if (Number.isFinite(r)) ctrl.rate = r;
  }
}
const f = harness.gui.addFolder('🤽 Big Pool Flow');
f.add(ctrl, 'fillBlock').name('fill block');
f.add(ctrl, 'pour').name('pour on/off');
f.add(ctrl, 'rate', 50, 800, 10).name('rate /s');
const renderModeCtrl = { renderMode: 'auto' };
f.add(renderModeCtrl, 'renderMode', ['auto', 'metaballs', 'screen'])
  .name('render mode')
  .onChange((m) => Promise.resolve(pack.setRenderMode(m)).catch(
    (e) => window.pushDbg?.(`[bigpool] setRenderMode('${m}') failed: ${e?.message ?? e}`)));
f.add(ctrl, 'clear').name('clear');
f.close();

/* HUD + metrics */
let probeData = { count: 0, meanY: NaN, stdY: NaN, fillPct: 0 };
let hudTick = 0;
function foamCount() {
  if (!foam) return null;
  return foam.count ?? foam.points?.count ?? 0;
}
harness.setHudProvider(() => [
  `<b>Big Pool</b> — 25×12.5×2m, h=0.55`,
  `fps <b>${harness.fps.toFixed(0)}</b>  sim ${pack.sim.simMs.toFixed(1)}ms  particles <b>${pack.sim.count}</b>`,
  `fill: <b>${probeData.count}</b> pts  level <b>${Number.isFinite(probeData.meanY) ? probeData.meanY.toFixed(3) : '—'}</b>m (${probeData.fillPct.toFixed(0)}%)`,
  `leaked: <b>${pack.sim.leakedTotal ?? 0}</b>  KE: ${(pack.sim.kineticEnergy ?? 0).toFixed(0)}`,
  `render <b>${pack.screenState.active ? 'screen' : 'metaballs'}</b>` +
    (pack.screenState.error ? ` <span class="warn">(${pack.screenState.error})</span>` : '') +
    `  foam <b>${foamCount() ?? 'n/a'}</b>` +
    `  screen ${pack.screenState.screenMs != null ? pack.screenState.screenMs.toFixed(1) + 'ms' : '—'}`,
]);

setupMetrics(() => ({
  scene: 'bigpool', particles: pack.sim.count, simMs: +pack.sim.simMs.toFixed(2),
  fill: probeData, leaked: pack.sim.leakedTotal ?? 0,
  kineticEnergy: +(pack.sim.kineticEnergy ?? 0).toFixed(1),
  // Lane C2: render pipeline telemetry
  renderMode: pack.screenState.active ? 'screen' : 'metaballs',
  screenMs: pack.screenState.screenMs != null ? +pack.screenState.screenMs.toFixed(2) : null,
  foamCount: foamCount(),
}));

setupAutoShots(harness.renderer, 6);

/* pour: rain-style emitter over the shallow (-x) end */
let emitAcc = 0;
harness.onFixed((dt) => {
  if (ctrl.pour) {
    emitAcc += ctrl.rate * dt;
    while (emitAcc >= 1 && pack.sim.count < pack.sim.p.maxParticles) {
      pack.sim.spawn(-11 + (Math.random() - 0.5) * 3, WH + 0.5 + Math.random() * 0.5,
        (Math.random() - 0.5) * (PW - 2), 0.3, -1, (Math.random() - 0.5) * 0.3);
      emitAcc--;
    }
  }
  pack.step(dt, colliders);
  try { foam?.update?.(dt, FOAM_FLAGS); } catch (e) { window.pushDbg?.(`[bigpool] foam update failed: ${e?.message ?? e}`); }
  if (++hudTick % 10 === 0) probeData = probe.measure();
});

window.__dbg = { pack, probe, harness };
window.pushDbg?.('Big Pool ready');
harness.tidyGui(); // all param subfolders collapsed by default
harness.start();
