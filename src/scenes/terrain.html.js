// scenes/terrain.html.js — heightfield terrain with a crevice/channel.
// Water pours on high ground, flows downhill, and pools in the crevice.
// Tests: heightfield colliders + flow + multi-region census.
// URL params: ?autoshot=N&label=x&metrics=terrain&pour=300

import * as THREE from 'three';
import { createDebugHarness } from '../debug-harness.js';
import { setupAutoShots } from '../debug-shot.js';
import { createWaterPack, createFillProbe, createRegionCensus, setupMetrics } from '../water-pack/index.js';

const harness = createDebugHarness({ cameraPos: [10, 9, 12], target: [0, 1, 0] });
const { scene } = harness;

/* terrain: 24×24 m heightfield, a valley channel curving to a deep crevice */
const NX = 97, NZ = 97, SIZE = 24, DX = SIZE / (NX - 1);
const heights = new Float32Array(NX * NZ);
for (let iz = 0; iz < NZ; iz++) {
  for (let ix = 0; ix < NX; ix++) {
    const x = -SIZE / 2 + ix * DX, z = -SIZE / 2 + iz * DX;
    // base rolling hills (gentle — local dips must not trap the channel flow)
    let h = 0.12 * Math.sin(x * 0.35) * Math.cos(z * 0.3) + 0.05 * Math.sin(x * 0.9 + z * 0.7);
    // slope down toward +x (water flows east) — NEGATIVE: h -= x*0.09 makes the
    // west (pour site) high ground and the crevice basin the lowest point.
    // (h += x*0.09 built a hill between pour and basin; water pooled at the spout.)
    h -= x * 0.09;
    // crevice: deep channel along z ≈ sin curve, centered x ≈ +3
    const channelZ = Math.sin(x * 0.5) * 2.2;
    const dist = Math.hypot(x - 3, z - channelZ);
    const crevice = -2.6 * Math.exp(-(dist * dist) / (2 * 1.1 * 1.1));
    h += crevice;
    // flat pool basin at the crevice bottom (x≈4.5) so water can settle
    const basin = Math.hypot(x - 4.5, z - Math.sin(4.5 * 0.5) * 2.2);
    if (basin < 2.2) h = Math.min(h, -1.6 + basin * 0.25);
    // close the valley: terrain rises east of x=7 so the basin is the low point
    if (x > 7) h += (x - 7) * (x - 7) * 0.35;
    // map rim: nothing can leave the play area
    const edge = Math.max(Math.abs(x), Math.abs(z)) / (SIZE / 2); // 0 center → 1 edge
    if (edge > 0.82) h += ((edge - 0.82) / 0.18) ** 2 * 6;
    heights[iz * NX + ix] = h;
  }
}

const terrainGeo = new THREE.PlaneGeometry(SIZE, SIZE, NX - 1, NZ - 1);
terrainGeo.rotateX(-Math.PI / 2);
{
  const posAttr = terrainGeo.attributes.position;
  for (let iz = 0; iz < NZ; iz++)
    for (let ix = 0; ix < NX; ix++)
      posAttr.setY(iz * NX + ix, heights[iz * NX + ix]);
  terrainGeo.computeVertexNormals();
}
const terrain = new THREE.Mesh(terrainGeo, new THREE.MeshStandardMaterial({ color: 0x5d7350, roughness: 0.95 }));
terrain.receiveShadow = true;
scene.add(terrain);

// wireframe overlay (debug: see the collider the water actually uses)
const wire = new THREE.Mesh(terrainGeo, new THREE.MeshBasicMaterial({ color: 0x8fae7e, wireframe: true, transparent: true, opacity: 0.12 }));
wire.position.y = 0.01;
scene.add(wire);

/* water pack with heightfield collider */
const bounds = { min: [-12, -3.5, -12], size: [24, 9, 24] };
const pack = createWaterPack({
  scene, bounds,
  params: { h: 0.3, maxParticles: 7000 },
  gui: harness.gui,
  renderer: harness.renderer,
  renderMode: 'metaballs', // default stays metaballs here; switch via GUI dropdown
});
pack.attachCompositor(harness);
const heightCollider = { type: 'heightfield', minX: -SIZE / 2, minZ: -SIZE / 2, nx: NX, nz: NZ, dx: DX, dz: DX, heights };
pack.sim.bounds = { min: [-12, -4, -12], max: [12, 6, 12] };

/* regions: crevice basin vs everywhere else */
const basinCenter = [4.5, Math.sin(4.5 * 0.5) * 2.2];
const census = createRegionCensus(pack.sim, {
  basin: { min: [basinCenter[0] - 2.2, -3, basinCenter[1] - 2.2], max: [basinCenter[0] + 2.2, 2, basinCenter[1] + 2.2] },
});
const basinProbe = createFillProbe(pack.sim, {
  min: [basinCenter[0] - 2.2, -3, basinCenter[1] - 2.2],
  max: [basinCenter[0] + 2.2, 2, basinCenter[1] + 2.2],
});

/* pour point: high ground on the west */
const ctrl = { pour: false, rate: 300, pourX: -8, pourZ: 1.67, clear: () => pack.sim.reset() };
{
  const qp = new URLSearchParams(location.search);
  if (qp.get('pour') != null) ctrl.pour = true;
  if (qp.get('pourX')) ctrl.pourX = parseFloat(qp.get('pourX'));
}
const f = harness.gui.addFolder('⛰ Terrain Flow');
f.add(ctrl, 'pour').name('rain on/off');
f.add(ctrl, 'rate', 50, 800, 10).name('rate /s');
f.add(ctrl, 'pourX', -10, 0, 0.5).name('pour x');
f.add(ctrl, 'pourZ', -8, 8, 0.5).name('pour z');
const renderModeCtrl = { renderMode: 'metaballs' };
f.add(renderModeCtrl, 'renderMode', ['auto', 'metaballs', 'screen'])
  .name('render mode')
  .onChange((m) => Promise.resolve(pack.setRenderMode(m)).catch(
    (e) => window.pushDbg?.(`[terrain] setRenderMode('${m}') failed: ${e?.message ?? e}`)));
f.add(ctrl, 'clear').name('clear');
f.close();

/* HUD + metrics */
let probeData = { count: 0, meanY: NaN, stdY: NaN, fillPct: 0 };
let censusData = { basin: 0, _other: 0 };
let hudTick = 0;
harness.setHudProvider(() => [
  `<b>Terrain Lab</b> — crevice channel, h=0.3`,
  `fps <b>${harness.fps.toFixed(0)}</b>  sim ${pack.sim.simMs.toFixed(1)}ms  particles <b>${pack.sim.count}</b>`,
  `basin: <b>${censusData.basin}</b> pts  elsewhere <b>${censusData._other}</b>`,
  `basin level <b>${Number.isFinite(probeData.meanY) ? probeData.meanY.toFixed(3) : '—'}</b>m  σy <b>${Number.isFinite(probeData.stdY) ? probeData.stdY.toFixed(3) : '—'}</b>`,
  `leaked: <b>${pack.sim.leakedTotal ?? 0}</b>  KE: ${(pack.sim.kineticEnergy ?? 0).toFixed(0)}`,
  `render <b>${pack.screenState.active ? 'screen' : 'metaballs'}</b>` +
    (pack.screenState.error ? ` <span class="warn">(${pack.screenState.error})</span>` : ''),
]);

setupMetrics(() => ({
  scene: 'terrain', particles: pack.sim.count, simMs: +pack.sim.simMs.toFixed(2),
  basin: probeData, census: censusData, leaked: pack.sim.leakedTotal ?? 0,
  kineticEnergy: +(pack.sim.kineticEnergy ?? 0).toFixed(1),
}));

setupAutoShots(harness.renderer, 4);

let emitAcc = 0;
harness.onFixed((dt) => {
  if (ctrl.pour) {
    emitAcc += ctrl.rate * dt;
    while (emitAcc >= 1 && pack.sim.count < pack.sim.p.maxParticles) {
      pack.sim.spawn(ctrl.pourX + (Math.random() - 0.5) * 0.5, 4 + Math.random() * 0.5,
        ctrl.pourZ + (Math.random() - 0.5) * 0.5, 0.5, -1, (Math.random() - 0.5) * 0.5);
      emitAcc--;
    }
  }
  pack.step(dt, [heightCollider]);
  if (++hudTick % 10 === 0) {
    probeData = basinProbe.measure();
    censusData = census.measure();
  }
});

window.__dbg = { pack, probe: basinProbe, census, harness, heights };
window.pushDbg?.('Terrain Lab ready');
harness.tidyGui(); // all param subfolders collapsed by default
harness.start();
