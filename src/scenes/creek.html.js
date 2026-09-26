// scenes/creek.html.js — "Creek Lab": meandering carved-channel creek.
// A 40×24 m heightfield with a sinuous parabolic channel, 2% downstream grade,
// rock bumps, an inflow nozzle at the upstream end and an outflow drain at
// the downstream end — water flows the full meander and recycles.
// Tests: long-channel heightfield flow + source/drain recycling.
// URL params: ?pour ?quality=low|medium|high ?autoshot=N&label=x&metrics=creek

import * as THREE from 'three';
import { createDebugHarness } from '../debug-harness.js';
import { setupAutoShots } from '../debug-shot.js';
import { createWater } from '../water/index.js';
import { attachWater, waterHudLine, addWaterGui, setupMetrics } from './water-harness.js';

const qp = new URLSearchParams(location.search);
const harness = createDebugHarness({ cameraPos: [16, 11, 18], target: [0, -0.5, 0] });
const { scene } = harness;

/* ======================================================================
 * Heightfield: 40 × 24 m, DX = DZ = 0.25 m
 * ==================================================================== */
const NX = 161, NZ = 97, SIZE_X = 40, SIZE_Z = 24;
const DX = SIZE_X / (NX - 1), DZ = SIZE_Z / (NZ - 1); // both 0.25 m

// meander centerline: two sinusoids → 2-3 graceful bends across 40 m
const A1 = 2.5, F1 = 0.28;          // primary bend (period ≈ 22.4 m)
const A2 = 1.2, F2 = 0.71, PH2 = 1.3; // secondary wiggle (period ≈ 8.9 m)
const channelZ = (x) => A1 * Math.sin(x * F1) + A2 * Math.sin(x * F2 + PH2);
// channel tangent dz/dx — used to aim the inflow jet down-creek
const channelDz = (x) => A1 * F1 * Math.cos(x * F1) + A2 * F2 * Math.cos(x * F2 + PH2);

const HALF_W = 1.6;   // channel half-width (m)
const DEPTH = 0.9;    // centerline depth below bank base (m)
const SLOPE = 0.02;   // 2% downstream grade → 0.8 m drop over the 40 m run
const SHOULDER_W = 1.2; // bank shoulder band beyond HALF_W (+0.15 m rise)

// rolling hills ±~0.3 m (gentle so they never trap the channel flow)
function hills(x, z) {
  return 0.18 * Math.sin(x * 0.21) * Math.cos(z * 0.27) + 0.09 * Math.sin(x * 0.53 + z * 0.41);
}

/* rocks: [x, lateral offset from centerline, height, radius] — gaussian blobs */
const ROCKS = [
  [-16.0,  0.4, 0.28, 0.70],
  [-12.0, -0.5, 0.22, 0.60],
  [-8.0,   0.9, 0.30, 0.80],
  [-3.0,  -0.9, 0.25, 0.70],
  [ 1.0,   0.3, 0.20, 0.55],
  [ 6.0,  -0.6, 0.32, 0.90],
  [11.0,   0.8, 0.26, 0.75],
  [15.0,  -0.4, 0.22, 0.65],
];
function rockHeight(x, z) {
  let s = 0;
  for (const [rx, rz, rh, rr] of ROCKS) {
    const d = (x - rx) ** 2 + (z - channelZ(rx) - rz) ** 2;
    s += rh * Math.exp(-d / (2 * rr * rr));
  }
  return s;
}

/* full terrain elevation */
function terrainH(x, z) {
  const outer = hills(x, z) - SLOPE * x;
  const t = Math.abs(z - channelZ(x)) / HALF_W;
  if (t < 1) return -SLOPE * x - DEPTH * (1 - t * t); // parabolic bed
  const tSh = SHOULDER_W / HALF_W;
  if (t < 1 + tSh) {
    const s = (t - 1) / tSh; // blend bed edge → hills, plus a +0.15 m shoulder bump
    return (-SLOPE * x + s * hills(x, z)) + 0.15 * Math.sin(Math.PI * s) ** 2;
  }
  return outer;
}

/* ground as the collider sees it: terrain + rocks + a steep quadratic rim
 * past 82% of the half-extent so nothing escapes the map */
function groundH(x, z) {
  let h = terrainH(x, z) + rockHeight(x, z);
  const edge = Math.max(Math.abs(x) / (SIZE_X / 2), Math.abs(z) / (SIZE_Z / 2));
  if (edge > 0.82) h += ((edge - 0.82) / 0.18) ** 2 * 6;
  return h;
}
const heights = new Float32Array(NX * NZ);
for (let iz = 0; iz < NZ; iz++) {
  for (let ix = 0; ix < NX; ix++) {
    heights[iz * NX + ix] = groundH(-SIZE_X / 2 + ix * DX, -SIZE_Z / 2 + iz * DZ);
  }
}

const terrainGeo = new THREE.PlaneGeometry(SIZE_X, SIZE_Z, NX - 1, NZ - 1);
terrainGeo.rotateX(-Math.PI / 2);

/* ---- wet-bed tinting (S1 visual polish) ----
 * Vertices below the estimated waterline get a darker, more saturated brown;
 * vertices within ~±0.15 m of the waterline near the channel form a wet
 * margin band; banks keep the dry olive base. Pure vertex-color pass over
 * the same heights array the collider uses — zero per-frame cost. */
const DRY_COLOR = new THREE.Color(0x6b6f4a);          // dry bank olive
const WET_COLOR = new THREE.Color(0x3e3322);          // wet bed: dark saturated brown
const WATER_FILL = 0.55;                              // assumed fill fraction of channel depth
const waterSurfaceY = (x) => -SLOPE * x - DEPTH * (1 - WATER_FILL);
const clamp01 = (v) => Math.min(1, Math.max(0, v));
function wetness(x, z, h) {
  const t = Math.abs(z - channelZ(x)) / HALF_W;
  if (t > 2.5) return 0;                              // far banks stay dry
  const surf = waterSurfaceY(x);
  if (h <= surf - 0.05) return 1;                     // submerged bed: fully wet
  // wet margin: fade out over ±0.15 m around the waterline
  return 1 - clamp01((h - (surf - 0.05)) / 0.30);
}
{
  const posAttr = terrainGeo.attributes.position;
  const colAttr = new THREE.BufferAttribute(new Float32Array(NX * NZ * 3), 3);
  const c = new THREE.Color();
  for (let iz = 0; iz < NZ; iz++) {
    for (let ix = 0; ix < NX; ix++) {
      const idx = iz * NX + ix;
      const h = heights[idx];
      posAttr.setY(idx, h);
      const x = -SIZE_X / 2 + ix * DX, z = -SIZE_Z / 2 + iz * DZ;
      c.copy(DRY_COLOR).lerp(WET_COLOR, wetness(x, z, h));
      colAttr.setXYZ(idx, c.r, c.g, c.b);
    }
  }
  terrainGeo.setAttribute('color', colAttr);
  terrainGeo.computeVertexNormals();
}
const terrain = new THREE.Mesh(terrainGeo, new THREE.MeshStandardMaterial({
  color: 0xffffff,       // white base so vertex colors read as-authored
  vertexColors: true,
  roughness: 0.9,
}));
terrain.receiveShadow = true;
scene.add(terrain);

// wireframe overlay (debug: see the collider the water actually uses)
const wire = new THREE.Mesh(terrainGeo, new THREE.MeshBasicMaterial({ color: 0x9aae7e, wireframe: true, transparent: true, opacity: 0.12 }));
wire.position.y = 0.01;
scene.add(wire);

/* ---- visible rock meshes on the heightfield bumps (S1) ----
 * The gaussian bumps in ROCKS are invisible as objects; these jittered
 * boulders sit at exactly the same [x, centerline-offset] spots so what you
 * SEE matches what the water collides with. Jitter is keyed on vertex
 * position so duplicated (non-indexed) icosahedron verts stay welded. */
{
  const rockMat = new THREE.MeshStandardMaterial({
    color: 0x7a7268, roughness: 0.92, flatShading: true,
  });
  for (const [rx, rzOff, rh, rr] of ROCKS) {
    const geo = new THREE.IcosahedronGeometry(rr * 1.15, 1);
    const pos = geo.attributes.position;
    const seen = new Map();
    for (let i = 0; i < pos.count; i++) {
      const key = `${pos.getX(i).toFixed(3)},${pos.getY(i).toFixed(3)},${pos.getZ(i).toFixed(3)}`;
      let j = seen.get(key);
      if (j === undefined) {
        j = [(Math.random() - 0.5) * rr * 0.35,
             (Math.random() - 0.5) * rr * 0.35,
             (Math.random() - 0.5) * rr * 0.35];
        seen.set(key, j);
      }
      pos.setXYZ(i, pos.getX(i) + j[0], pos.getY(i) + j[1], pos.getZ(i) + j[2]);
    }
    geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, rockMat);
    const cz = channelZ(rx) + rzOff;
    // embed ~half the boulder so it reads as sitting IN the bed bump apex
    mesh.position.set(rx, terrainH(rx, cz) + rh * 0.55, cz);
    mesh.scale.y = Math.max(0.45, rh / rr); // taller bumps → prouder rocks
    mesh.rotation.y = rx * 1.7;             // deterministic per-rock spin
    mesh.castShadow = mesh.receiveShadow = true;
    scene.add(mesh);
  }
}

/* ---- background gradient sky (S1): cheap inverted sphere ----
 * Vertical gradient (zenith → horizon → below-horizon) instead of the flat
 * void; subtle fog blends distant terrain into the horizon color. The
 * ShaderMaterial ignores fog by default, so only world geometry fades. */
{
  const skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: {
      uTop: { value: new THREE.Color(0x27334c) },
      uMid: { value: new THREE.Color(0x1a2130) },
      uBottom: { value: new THREE.Color(0x11141b) },
    },
    vertexShader: /* glsl */`
      varying vec3 vDir;
      void main() {
        vDir = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: /* glsl */`
      uniform vec3 uTop; uniform vec3 uMid; uniform vec3 uBottom;
      varying vec3 vDir;
      void main() {
        float hgt = vDir.y;
        vec3 c = hgt >= 0.0
          ? mix(uMid, uTop, smoothstep(0.0, 0.6, hgt))
          : mix(uMid, uBottom, smoothstep(0.0, 0.5, -hgt));
        gl_FragColor = vec4(c, 1.0);
        #include <colorspace_fragment>
      }`,
  });
  const sky = new THREE.Mesh(new THREE.SphereGeometry(220, 24, 12), skyMat);
  sky.frustumCulled = false;
  scene.add(sky);
  scene.fog = new THREE.Fog(0x1a2130, 60, 200); // match the horizon band
}

/* ======================================================================
 * Water with the terrain as a heightfield collider
 * ==================================================================== */
const quality = qp.get('quality') ?? 'medium';
const spacing = { low: 0.2, medium: 0.15, high: 0.12 }[quality] ?? 0.15;
const water = await createWater({
  renderer: harness.renderer, scene, quality,
  render: qp.get('render') ?? 'screen',
  // a gravel bed: quadratic drag coefficient C_f 0.02
  params: { spacing, maxParticles: 32768, friction: 0.02, bounds: { min: [-20, -3, -12], max: [20, 6, 12] } },
});
water.addHeightfield({ minX: -SIZE_X / 2, minZ: -SIZE_Z / 2, dx: DX, dz: DZ, nx: NX, nz: NZ, heights });

/* outflow drain: the downstream end (x > 16.5), across the whole map width so
 * water that overtops the banks near the outlet recycles too */
const outflow = water.addDrain({ min: [16.5, -3, -12], max: [20, 4, 12] });

/* inflow nozzle: just above the bed at the upstream end, aimed down-creek */
const EMIT_X = -17.5;
const emitZ = channelZ(EMIT_X);
const tangent = [1, 0, channelDz(EMIT_X)];
const inflow = water.addSource({
  position: [EMIT_X, groundH(EMIT_X, emitZ) + 0.6, emitZ],
  direction: [tangent[0], -0.35, tangent[2]],
  radius: 0.35, speed: 2, enabled: qp.has('pour'),
});

const f = harness.gui.addFolder('🏞 Creek Flow');
f.add(inflow, 'enabled').name('inflow on/off');
f.add(inflow, 'speed', 0.5, 5, 0.1).name('inflow speed (m/s)');
f.add(inflow, 'radius', 0.15, 0.6, 0.05).name('inflow radius (m)');
f.add(outflow, 'enabled').name('outflow drain');
f.add({ clear: () => water.reset() }, 'clear').name('clear');
addWaterGui(harness.gui, water);

/* HUD + metrics */
const flowRate = () => (inflow.enabled ? Math.PI * inflow.radius ** 2 * inflow.speed : 0);
harness.setHudProvider(() => [
  `<b>Creek Lab</b> — meandering channel, 2% grade, spacing ${water.params.spacing} m`,
  waterHudLine(harness, water),
  `inflow <b>${(flowRate() * 1000).toFixed(0)}</b> L/s @ x=${EMIT_X}  drained <b>${water.stats.drained ?? 0}</b>`,
  `leaked: <b>${water.stats.leaked ?? 0}</b>  KE ${(water.stats.kineticEnergy ?? 0).toFixed(0)} J`,
]);
setupMetrics(() => ({
  scene: 'creek', particles: water.count, stepMs: water.stats.stepMs, flowRate: flowRate(),
  drained: water.stats.drained ?? 0, leaked: water.stats.leaked ?? 0, kineticEnergy: water.stats.kineticEnergy,
}));
setupAutoShots(harness.renderer, 4);

attachWater(harness, water);

window.__dbg = { water, harness, heights, channelZ, groundH };
window.pushDbg?.('Creek Lab ready');
harness.tidyGui();
harness.start();
