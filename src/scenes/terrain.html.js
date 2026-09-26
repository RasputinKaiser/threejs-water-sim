// scenes/terrain.html.js — heightfield terrain with a crevice/channel.
// Water pours on high ground, flows downhill, and pools in the crevice basin.
// Tests: heightfield colliders, long-distance flow, pooling.
// URL params: ?pour ?quality=low|medium|high ?autoshot=N&label=x&metrics=terrain

import * as THREE from 'three';
import { createDebugHarness } from '../debug-harness.js';
import { setupAutoShots } from '../debug-shot.js';
import { createWater } from '../water/index.js';
import { attachWater, fillProbe, waterHudLine, addWaterGui, setupMetrics } from './water-harness.js';

const qp = new URLSearchParams(location.search);
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

/* water with the terrain as a heightfield collider */
const quality = qp.get('quality') ?? 'medium';
const spacing = { low: 0.2, medium: 0.15, high: 0.12 }[quality] ?? 0.15;
const water = await createWater({
  renderer: harness.renderer, scene, quality,
  render: qp.get('render') ?? 'screen', backend: qp.get('backend') ?? 'cpu',
  params: { spacing, maxParticles: 32768, bounds: { min: [-12, -4, -12], max: [12, 7, 12] } },
});
water.addHeightfield({ minX: -SIZE / 2, minZ: -SIZE / 2, dx: DX, dz: DX, nx: NX, nz: NZ, heights });

/* the crevice basin vs everywhere else */
const basinCenter = [4.5, Math.sin(4.5 * 0.5) * 2.2];
const basin = { min: [basinCenter[0] - 2.2, -3, basinCenter[1] - 2.2], max: [basinCenter[0] + 2.2, 2, basinCenter[1] + 2.2] };
const basinProbe = fillProbe(water, basin);

/* pour point: high ground on the west */
const ctrl = { pourX: parseFloat(qp.get('pourX') ?? '-8'), pourZ: 1.67 };
const rain = water.addSource({ position: [ctrl.pourX, 4, ctrl.pourZ], direction: [0.3, -1, 0], radius: 0.35, speed: 3, enabled: qp.has('pour') });
const move = () => { rain.position = [ctrl.pourX, 4, ctrl.pourZ]; };
const f = harness.gui.addFolder('⛰ Terrain Flow');
f.add(rain, 'enabled').name('pour on/off');
f.add(rain, 'speed', 0.5, 8, 0.1).name('pour speed (m/s)');
f.add(ctrl, 'pourX', -10, 0, 0.5).name('pour x').onChange(move);
f.add(ctrl, 'pourZ', -8, 8, 0.5).name('pour z').onChange(move);
f.add({ dump: () => water.fillBox([ctrl.pourX - 0.8, 2, ctrl.pourZ - 0.8], [ctrl.pourX + 0.8, 3.2, ctrl.pourZ + 0.8]) }, 'dump').name('💧 dump a block');
f.add({ clear: () => water.reset() }, 'clear').name('clear');
addWaterGui(harness.gui, water);

/* HUD + metrics */
let probeData = basinProbe.measure();
let hudTick = 0;
const num = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '—');
harness.setHudProvider(() => [
  `<b>Terrain Lab</b> — crevice channel, spacing ${water.params.spacing} m`,
  waterHudLine(harness, water),
  `basin: <b>${probeData.count}</b> particles  elsewhere <b>${water.count - probeData.count}</b>`,
  `basin level <b>${num(probeData.level, 2)}</b> m  surface σ <b>${num(probeData.flatness)}</b> m`,
  `leaked: <b>${water.stats.leaked ?? 0}</b>  KE ${num(water.stats.kineticEnergy, 0)} J`,
]);
setupMetrics(() => ({
  scene: 'terrain', particles: water.count, stepMs: water.stats.stepMs, basin: probeData,
  leaked: water.stats.leaked ?? 0, kineticEnergy: water.stats.kineticEnergy,
}));
setupAutoShots(harness.renderer, 4);

attachWater(harness, water, { after() { if (++hudTick % 15 === 0) probeData = basinProbe.measure(); } });

window.__dbg = { water, probe: basinProbe, harness, heights };
window.pushDbg?.('Terrain Lab ready');
harness.tidyGui();
harness.start();
