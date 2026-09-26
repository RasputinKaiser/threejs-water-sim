// scenes/water-lab.js — the water pack as a game would use it: a Box3D pool,
// floating crates and balls of different densities, a pouring nozzle, and
// water.render() in place of renderer.render().

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import GUI from 'lil-gui';
import Box3DInit from 'box3d.js/inline';
import { createWater } from '../water/index.js';

const qp = new URLSearchParams(location.search);

/* ---- three.js ---- */
const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x9fb4c8);
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.6;
scene.add(new THREE.HemisphereLight(0xb8c8e8, 0x39404d, 0.8));
const sun = new THREE.DirectionalLight(0xfff1dc, 2.5);
sun.position.set(6, 12, 4);
scene.add(sun);
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 200);
camera.position.set(4.2, 3.6, 5.2);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.6, 0);
controls.enableDamping = true;
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

/* ---- Box3D world: a pool of static boxes + floating props ---- */
const b3 = await Box3DInit();
const wd = b3.b3DefaultWorldDef();
wd.gravity = [0, -9.81, 0];
const world = b3.b3CreateWorld(wd);
const meshes = [];
const stone = new THREE.MeshStandardMaterial({ color: 0x8c96a6, roughness: 0.85 });

function staticBox(pos, half) {
  const d = b3.b3DefaultBodyDef();
  d.position = pos;
  b3.b3CreateBoxShape(b3.b3CreateBody(world, d), b3.b3DefaultShapeDef(), ...half);
  const m = new THREE.Mesh(new THREE.BoxGeometry(half[0] * 2, half[1] * 2, half[2] * 2), stone);
  m.position.set(...pos);
  scene.add(m);
}
const P = 1.6, WALL = 0.15, H = 0.9;
staticBox([0, -0.25, 0], [4, 0.25, 4]);
staticBox([-(P + WALL), H / 2, 0], [WALL, H / 2, P + 2 * WALL]);
staticBox([P + WALL, H / 2, 0], [WALL, H / 2, P + 2 * WALL]);
staticBox([0, H / 2, -(P + WALL)], [P + 2 * WALL, H / 2, WALL]);
staticBox([0, H / 2, P + WALL], [P + 2 * WALL, H / 2, WALL]);

function prop({ shape, size, density, color, pos }) {
  const d = b3.b3DefaultBodyDef();
  d.type = b3.b3BodyType.b3_dynamicBody;
  d.position = pos;
  d.angularDamping = 0.2;
  const body = b3.b3CreateBody(world, d);
  const sd = b3.b3DefaultShapeDef();
  sd.density = density;
  let geo;
  if (shape === 'box') { b3.b3CreateBoxShape(body, sd, size, size, size); geo = new THREE.BoxGeometry(size * 2, size * 2, size * 2); }
  else { b3.b3CreateSphereShape(body, sd, { center: [0, 0, 0], radius: size }); geo = new THREE.SphereGeometry(size, 32, 16); }
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, roughness: 0.6 }));
  scene.add(mesh);
  meshes.push({ body, mesh });
}
prop({ shape: 'box', size: 0.18, density: 400, color: 0xc0873f, pos: [-0.6, 1.4, 0.2] });
prop({ shape: 'box', size: 0.14, density: 700, color: 0x9a6b3a, pos: [0.5, 1.8, -0.4] });
prop({ shape: 'sphere', size: 0.15, density: 300, color: 0xe24a3b, pos: [0.2, 2.2, 0.6] });
prop({ shape: 'sphere', size: 0.12, density: 2500, color: 0x555a63, pos: [-0.2, 2.6, -0.5] });

/* ---- water ---- */
const water = await createWater({
  renderer, scene, b3, world,
  quality: qp.get('quality') ?? 'medium',
  params: { bounds: { min: [-6, -2, -6], max: [6, 10, 6] } },
  render: qp.get('render') ?? 'screen',
});
water.fillBox([-P, 0, -P], [P, 0.5, P]);
const nozzle = water.addSource({ position: [1.0, 2.4, 1.0], direction: [-0.3, -1, -0.2], radius: 0.12, speed: 3, enabled: qp.has('pour') });

/* ---- ui ---- */
const gui = new GUI();
gui.add(nozzle, 'enabled').name('pour');
gui.add(nozzle, 'speed', 0.5, 8, 0.1).name('pour speed');
const tune = { viscosity: water.params.viscosity, vorticity: water.params.vorticity, iterations: water.params.iterations };
gui.add(tune, 'viscosity', 0, 0.2, 0.005).onChange((v) => water.setParams({ viscosity: v }));
gui.add(tune, 'vorticity', 0, 0.3, 0.005).onChange((v) => water.setParams({ vorticity: v }));
gui.add(tune, 'iterations', 1, 10, 1).onChange((v) => water.setParams({ iterations: v }));
gui.add({ splash: () => water.fillBox([-0.5, 1.4, -0.5], [0.5, 2.0, 0.5], { velocity: [0, -2, 0] }) }, 'splash').name('drop a block of water');
gui.add({ reset: () => { water.reset(); water.fillBox([-P, 0, -P], [P, 0.5, P]); } }, 'reset');

/* ---- loop ---- */
const hud = document.getElementById('hud');
const _p = [0, 0, 0], _q = [0, 0, 0, 1];
let last = performance.now(), fps = 60, hudT = 0;
function frame(now) {
  const dt = Math.min((now - last) / 1000, 1 / 20);
  last = now;
  fps = fps * 0.95 + (1 / Math.max(dt, 1e-4)) * 0.05;
  const simDt = water.update(dt);            // fluid time advanced this frame
  if (simDt > 0) b3.b3World_Step(world, simDt, 4); // rigid bodies in lockstep with it
  for (const { body, mesh } of meshes) {
    b3.b3Body_GetPosition(_p, body); b3.b3Body_GetRotation(_q, body);
    mesh.position.set(..._p); mesh.quaternion.set(..._q);
  }
  controls.update();
  water.render(camera);
  if ((hudT += dt) > 0.25) {
    hudT = 0;
    const s = water.stats;
    hud.innerHTML = `<b>Water Lab</b>  ${water.mode} solver${s.threads ? ` · ${s.threads} threads` : ''}\n` +
      `fps <b>${fps.toFixed(0)}</b>  particles <b>${water.count}</b>  step <b>${(s.stepMs ?? 0).toFixed(1)}</b> ms\n` +
      `density error ${((s.maxDensityError ?? 0) * 100).toFixed(1)}%  colliders ${s.colliders ?? 0}`;
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
window.__water = { water, world, b3, camera };
