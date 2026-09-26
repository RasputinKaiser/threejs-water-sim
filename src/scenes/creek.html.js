// scenes/creek.html.js — "Creek Lab": water in a physical world.
//
// A 40 m meandering creek (scenes/creek-world.js): terrain and boulders exist
// in both worlds — Box3D (heightfield + static spheres) and the water solver
// (heightfield + the coupling's sphere colliders) — so logs float down the
// current, bump into boulders and strand on the banks. Water enters through
// a submerged inlet, leaves at the outlet, and whitewater (spray, foam,
// bubbles) forms where it plunges, converges and breaks.
// URL params: ?quality=low|medium|high ?logs=N ?threads=N ?render=screen|points
//             ?autoshot=N&label=x&metrics=creek

import * as THREE from 'three';
import { createDebugHarness } from '../debug-harness.js';
import { setupAutoShots } from '../debug-shot.js';
import { createWater } from '../water/index.js';
import { attachWater, waterHudLine, addWaterGui, setupMetrics } from './water-harness.js';
import * as creek from './creek-world.js';
import Box3DInit from 'box3d.js/inline';

const qp = new URLSearchParams(location.search);
const b3 = await Box3DInit();
const harness = createDebugHarness({ cameraPos: [-19, 7, 12], target: [-6, -0.8, 1] });
const { scene, renderer } = harness;

/* ---------------------------- sky + light ---------------------------- */
const SKY = { top: new THREE.Color(0x3d6fb6), horizon: new THREE.Color(0xc4d6e8), ground: new THREE.Color(0x5b6450) };
{
  const skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false,
    uniforms: { uTop: { value: SKY.top }, uMid: { value: SKY.horizon }, uBottom: { value: SKY.ground } },
    vertexShader: /* glsl */`
      varying vec3 vDir;
      void main() { vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */`
      uniform vec3 uTop; uniform vec3 uMid; uniform vec3 uBottom;
      varying vec3 vDir;
      void main() {
        float y = vDir.y;
        vec3 c = y >= 0.0 ? mix(uMid, uTop, pow(smoothstep(0.0, 0.7, y), 0.7)) : mix(uMid, uBottom, smoothstep(0.0, 0.3, -y));
        gl_FragColor = vec4(c, 1.0);
        #include <colorspace_fragment>
      }`,
  });
  const sky = new THREE.Mesh(new THREE.SphereGeometry(220, 32, 16), skyMat);
  sky.frustumCulled = false;
  scene.add(sky);
  scene.fog = new THREE.Fog(SKY.horizon, 70, 210);
  scene.background = SKY.horizon.clone();
  // the water reflects this sky, not the harness's studio environment
  const envScene = new THREE.Scene();
  envScene.add(sky.clone());
  scene.environment = new THREE.PMREMGenerator(renderer).fromScene(envScene, 0).texture;
  scene.environmentIntensity = 0.9;
  scene.traverse((o) => {
    if (o.isDirectionalLight) { o.position.set(-12, 22, 9); o.intensity = 2.8; o.color.set(0xfff0d8); }
  });
}

/* ------------------------------ terrain ------------------------------ */
const heights = creek.buildHeights();
const { NX, NZ, SIZE_X, SIZE_Z, DX, DZ } = creek;
{
  const geo = new THREE.PlaneGeometry(SIZE_X, SIZE_Z, NX - 1, NZ - 1);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  const col = new THREE.BufferAttribute(new Float32Array(NX * NZ * 3), 3);
  const dry = new THREE.Color(0x71764e), grass = new THREE.Color(0x5f7a3c), wet = new THREE.Color(0x574836);
  const gravel = new THREE.Color(0x857a68), c = new THREE.Color();
  for (let iz = 0; iz < NZ; iz++) {
    for (let ix = 0; ix < NX; ix++) {
      const k = iz * NX + ix, h = heights[k];
      pos.setY(k, h);
      const x = -SIZE_X / 2 + ix * DX, z = -SIZE_Z / 2 + iz * DZ;
      const t = Math.abs(z - creek.channelZ(x)) / creek.HALF_W;
      const surf = creek.bedY(x) + 0.5;                 // typical water line
      const wetness = h < surf - 0.05 ? 1 : Math.max(0, 1 - (h - surf + 0.05) / 0.35);
      const noise = 0.5 + 0.5 * Math.sin(x * 3.1 + z * 1.7) * Math.cos(x * 1.3 - z * 2.9);
      c.copy(t < 1.6 ? gravel : grass).lerp(dry, t < 1.6 ? 0.25 * noise : 0.4 * noise);
      c.lerp(wet, Math.min(1, wetness) * (t < 2.5 ? 1 : 0));
      col.setXYZ(k, c.r, c.g, c.b);
    }
  }
  geo.setAttribute('color', col);
  geo.computeVertexNormals();
  const terrain = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95 }));
  terrain.receiveShadow = true;
  scene.add(terrain);
}

/* ----------------------------- Box3D world ---------------------------- */
const wd = b3.b3DefaultWorldDef();
wd.gravity = [0, -9.81, 0];
const world = b3.b3CreateWorld(wd);
{
  // the same heightfield (row-major, origin at the grid corner)
  const def = b3.b3DefaultBodyDef();
  def.position = [-SIZE_X / 2, 0, -SIZE_Z / 2];
  const ground = b3.b3CreateBody(world, def);
  const hf = b3.b3CreateHeightField(heights, NX, NZ, [DX, 1, DZ]);
  const sd = b3.b3DefaultShapeDef();
  sd.baseMaterial.friction = 0.7;
  b3.b3CreateHeightFieldShape(ground, sd, hf);
}
// boulders: static spheres (the coupling makes them water colliders too)
const rockMat = new THREE.MeshStandardMaterial({ color: 0x8a8378, roughness: 0.9, flatShading: true });
let seed = 11;
const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
function rockMesh(r) {
  const geo = new THREE.IcosahedronGeometry(r * 1.05, 2);
  const pos = geo.attributes.position, jit = new Map();
  for (let i = 0; i < pos.count; i++) {
    const key = `${pos.getX(i).toFixed(3)},${pos.getY(i).toFixed(3)},${pos.getZ(i).toFixed(3)}`;
    let j = jit.get(key);
    if (!j) jit.set(key, j = 1 + (rand() - 0.5) * 0.14);
    pos.setXYZ(i, pos.getX(i) * j, pos.getY(i) * j * 0.92, pos.getZ(i) * j);
  }
  geo.computeVertexNormals();
  return new THREE.Mesh(geo, rockMat);
}
for (const b of creek.boulderSpheres()) {
  const def = b3.b3DefaultBodyDef();
  def.position = b.center;
  b3.b3CreateSphereShape(b3.b3CreateBody(world, def), b3.b3DefaultShapeDef(), { center: [0, 0, 0], radius: b.radius });
  const m = rockMesh(b.radius);
  m.position.set(...b.center);
  m.rotation.y = b.center[0] * 1.7;
  m.castShadow = m.receiveShadow = true;
  scene.add(m);
}

/* -------------------------------- logs -------------------------------- */
const barkMat = new THREE.MeshStandardMaterial({ color: 0x6b4a2f, roughness: 0.95 });
const endMat = new THREE.MeshStandardMaterial({ color: 0xb58b5a, roughness: 0.9 });
const logs = [];
function logSpawn(i) {
  // upstream of the inlet's jet, across the channel, slightly apart
  const x = -15.5 + (i % 3) * 1.4 + rand() * 0.4;
  const z = creek.channelZ(x) + (rand() - 0.5) * 0.8;
  return [x, creek.bedY(x) + 1.3 + (i % 2) * 0.35, z];
}
function makeLog(i) {
  const r = 0.11 + rand() * 0.06, half = 0.45 + rand() * 0.45;
  const def = b3.b3DefaultBodyDef();
  def.type = b3.b3BodyType.b3_dynamicBody;
  def.position = logSpawn(i);
  const yaw = rand() * Math.PI;
  def.rotation = [0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2)];
  def.angularDamping = 0.3;
  const body = b3.b3CreateBody(world, def);
  const sd = b3.b3DefaultShapeDef();
  sd.density = 600 + rand() * 150; // wet wood: floats ~⅔ submerged
  sd.baseMaterial.friction = 0.6;
  // capsule along local x
  b3.b3CreateCapsuleShape(body, sd, { center1: [-half, 0, 0], center2: [half, 0, 0], radius: r });
  const group = new THREE.Group();
  const trunk = new THREE.Mesh(new THREE.CylinderGeometry(r, r * 0.95, half * 2, 12, 1, true), barkMat);
  trunk.rotation.z = Math.PI / 2;
  const capA = new THREE.Mesh(new THREE.SphereGeometry(r, 12, 8, 0, Math.PI * 2, 0, Math.PI / 2), endMat);
  capA.rotation.z = -Math.PI / 2; capA.position.x = half;
  const capB = capA.clone(); capB.rotation.z = Math.PI / 2; capB.position.x = -half;
  group.add(trunk, capA, capB);
  group.traverse((o) => { o.castShadow = o.receiveShadow = true; });
  scene.add(group);
  logs.push({ body, mesh: group, i });
}
const LOGS = Number(qp.get('logs') ?? 4);
for (let i = 0; i < LOGS; i++) makeLog(i);
const _p = [0, 0, 0], _q = [0, 0, 0, 1];
let logsRecycled = 0;
function syncLogs() {
  for (const l of logs) {
    b3.b3Body_GetPosition(_p, l.body);
    if (_p[0] > creek.OUTLET.min[0] + 1 || _p[1] < -6) {
      // past the outlet: back upstream
      b3.b3Body_SetTransform(l.body, logSpawn(l.i), [0, 0, 0, 1]);
      b3.b3Body_SetLinearVelocity(l.body, [0, 0, 0]);
      b3.b3Body_SetAngularVelocity(l.body, [0, 0, 0]);
      logsRecycled++;
      b3.b3Body_GetPosition(_p, l.body);
    }
    b3.b3Body_GetRotation(_q, l.body);
    l.mesh.position.set(_p[0], _p[1], _p[2]);
    l.mesh.quaternion.set(_q[0], _q[1], _q[2], _q[3]);
  }
}

/* ------------------------------- water -------------------------------- */
const quality = qp.get('quality') ?? 'medium';
const spacing = { low: 0.18, medium: 0.15, high: 0.12 }[quality] ?? 0.15;
const water = await createWater({
  renderer, scene, b3, world, quality,
  threads: qp.has('threads') ? Number(qp.get('threads')) : 'auto',
  render: qp.get('render') ?? 'screen', backend: qp.get('backend') ?? 'cpu',
  params: {
    spacing, maxParticles: 65536, friction: 0.02, // gravel bed
    bounds: { min: [-20, -4, -12], max: [20, 6, 12] },
  },
  look: { absorption: [0.35, 0.1, 0.09], scatterColor: [0.05, 0.13, 0.12], scatter: 0.45, refraction: 0.05 },
});
water.addHeightfield(creek.heightfieldDesc(heights));
const inlet = water.addSource(creek.inlet());
const outlet = water.addDrain(creek.OUTLET);

// start with water in the channel, already moving down-creek
const fillChannel = () => water.spawn(creek.channelFill(water.params.spacing));
fillChannel();

/* --------------------------------- UI --------------------------------- */
const f = harness.gui.addFolder('🏞 Creek');
f.add(inlet, 'enabled').name('inflow on/off');
f.add(inlet, 'speed', 0.3, 4, 0.1).name('inflow speed (m/s)');
f.add(inlet, 'radius', 0.2, 0.7, 0.05).name('inflow radius (m)');
f.add(outlet, 'enabled').name('outflow');
f.add({ fill: fillChannel }, 'fill').name('💧 refill channel');
f.add({ log: () => makeLog(logs.length) }, 'log').name('🪵 add a log');
f.add({ clear: () => water.reset() }, 'clear').name('clear water');
addWaterGui(harness.gui, water);

/* -------------------------------- HUD --------------------------------- */
const vol = () => water.params.spacing ** 3;
let lastT = performance.now(), lastSpawned = 0, lastDrained = 0, qIn = 0, qOut = 0;
let gauges = creek.STATIONS.map(() => ({ depth: 0, speed: 0 }));
function measure() {
  const now = performance.now(), dt = (now - lastT) / 1000;
  if (dt < 1) return;
  const st = water.stats;
  qIn = ((st.spawned ?? 0) - lastSpawned) * vol() / dt * 1000;
  qOut = ((st.drained ?? 0) - lastDrained) * vol() / dt * 1000;
  lastSpawned = st.spawned ?? 0; lastDrained = st.drained ?? 0; lastT = now;
  const p = water.sim.positions, v = water.sim.velocities, n = water.count, s = water.params.spacing;
  gauges = creek.STATIONS.map((sx) => {
    const cz = creek.channelZ(sx), tz = creek.channelDz(sx), tl = Math.hypot(1, tz);
    let top = -Infinity, us = 0, k = 0;
    for (let i = 0; i < n; i++) {
      if (Math.abs(p[i * 3] - sx) > 0.3 || Math.abs(p[i * 3 + 2] - cz) > 0.5) continue;
      top = Math.max(top, p[i * 3 + 1]); us += (v[i * 3] + v[i * 3 + 2] * tz) / tl; k++;
    }
    return { depth: k ? Math.max(0, top + s / 2 - creek.terrainH(sx, cz)) : 0, speed: k ? us / k : 0 };
  });
}
harness.setHudProvider(() => {
  measure();
  const st = water.stats;
  return [
    `<b>Creek Lab</b> — 40 m meander, 2% grade, gravel bed · spacing ${water.params.spacing} m`,
    waterHudLine(harness, water),
    `inflow <b>${qIn.toFixed(0)}</b> L/s  outflow <b>${qOut.toFixed(0)}</b> L/s  whitewater <b>${st.whitewater ?? 0}</b>`,
    `depth ${gauges.map((g) => g.depth.toFixed(2)).join(' / ')} m   speed ${gauges.map((g) => g.speed.toFixed(2)).join(' / ')} m/s`,
    `logs ${logs.length} (recycled ${logsRecycled})  leaked ${st.leaked ?? 0}`,
  ];
});
setupMetrics(() => ({
  scene: 'creek', particles: water.count, stepMs: water.stats.stepMs, inflow: qIn, outflow: qOut,
  gauges, whitewater: water.stats.whitewater, leaked: water.stats.leaked, logsRecycled,
}));
setupAutoShots(renderer, 4);

attachWater(harness, water, { b3, world, after: syncLogs });

window.__dbg = { water, harness, b3, world, logs, creek };
window.pushDbg?.('Creek Lab ready');
harness.tidyGui();
harness.start();
