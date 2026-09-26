// scenes/creek.html.js — "Creek Lab": meandering carved-channel creek.
// A 40×24 m heightfield with a sinuous parabolic channel, 2% downstream grade,
// rock bumps, an inflow emitter at the upstream end and an outflow drain at
// the downstream end — water flows the full meander and recycles.
// Tests: long-channel heightfield flow + emitter/drain recycling.
// URL params: ?autoshot=N&label=x&metrics=creek&pour[=rate]

import * as THREE from 'three';
import { createDebugHarness } from '../debug-harness.js';
import { setupAutoShots } from '../debug-shot.js';
import { createWaterPack, setupMetrics, loadWaterLane } from '../water-pack/index.js';

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

const heights = new Float32Array(NX * NZ);
for (let iz = 0; iz < NZ; iz++) {
  for (let ix = 0; ix < NX; ix++) {
    const x = -SIZE_X / 2 + ix * DX, z = -SIZE_Z / 2 + iz * DZ;
    let h = terrainH(x, z) + rockHeight(x, z);
    // map rim: steep quadratic ramp past 82% of half-extent — nothing escapes
    const edge = Math.max(Math.abs(x) / (SIZE_X / 2), Math.abs(z) / (SIZE_Z / 2));
    if (edge > 0.82) h += ((edge - 0.82) / 0.18) ** 2 * 6;
    heights[iz * NX + ix] = h;
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
 * Water pack with heightfield collider
 * ==================================================================== */
const bounds = { min: [-20, -3, -12], size: [40, 9, 24] };
const pack = createWaterPack({
  scene, bounds,
  // creek tuning (R3 research): low viscosity keeps the current lively,
  // slippery riverbed + downhill assist so flow sustains on the 2% grade
  params: {
    h: 0.35, maxParticles: 26000,
    viscositySigma: 1, viscosityBeta: 0.25,
    bedFriction: 0.05, slopeAssist: 0.8,
  },
  gui: harness.gui,
  renderer: harness.renderer,
  workerSim: true, // off-thread solver via SAB (falls back to sync unsupported)
  renderMode: 'screen', // DEFAULT: screen-space flow shading (~0.5ms vs metaballs' ~43ms @26k).
                        // Switch to metaballs/auto via the GUI dropdown at runtime.
});
pack.attachCompositor(harness);
const heightCollider = { type: 'heightfield', minX: -SIZE_X / 2, minZ: -SIZE_Z / 2, nx: NX, nz: NZ, dx: DX, dz: DZ, heights };
pack.sim.bounds = { min: [-20, -3, -12], max: [20, 6, 12] };

/* FoamSystem (two-tier after B-R2 upgrade): whitewater core lines along the
 * fast channel center + short-lived spray at banks/rocks. R3 flags. */
let foam = null;
loadWaterLane('effects').then((fx) => {
  if (!fx?.FoamSystem) { window.pushDbg?.('[creek] effects lane unavailable — no foam'); return; }
  try {
    foam = new fx.FoamSystem(pack.sim, 4000);
    scene.add(foam.points);
    foam.addGui?.(harness.gui);
    FOAM_BUDGET_BASE.spawnPerFrame = foam.params?.spawnPerFrame ?? FOAM_BUDGET_BASE.spawnPerFrame;
    window.pushDbg?.('[creek] FoamSystem ready (budget 4000)');
  } catch (e) {
    window.pushDbg?.(`[creek] FoamSystem init failed: ${e?.message ?? e}`);
    foam = null;
  }
}).catch((e) => window.pushDbg?.(`[creek] effects load failed: ${e?.message ?? e}`));
/* two-tier flags (R3, S1 retune): R3 §3 puts the fast-core threshold at
 * ~0.7–1.0 m/s — coreSpeed 1.6 only lit the very fastest particles, so most
 * of the core had no whitewater line. 1.1 promotes the whole fast core;
 * coreBudget raised 60→140 (sim headroom is ample: ~3-4k particles ≈ 4-5 ms)
 * so the longer core lines don't get capped mid-stream. maxSpeed/minNeighbors
 * stay at the R3-recommended rock-white-water values. */
const FOAM_FLAGS = { maxSpeed: 0.8, minNeighbors: 8, coreSpeed: 1.1, coreBudget: 140 };

/* ---- adaptive foam budget (S1 perf) ----------------------------------
 * Foam spawning costs real main-thread ms; with screen rendering freeing
 * budget we don't want foam to eat it back. Sampled once per second against
 * the harness EMA fps: below 45 fps, core + spray spawn caps scale down
 * proportionally (floor 30%); above 55 fps they restore to authored values.
 * The 45–55 band HOLDS the current scale (hysteresis) so the budget doesn't
 * oscillate. Spray tier is capped by foam.params.spawnPerFrame; core tier by
 * FOAM_FLAGS.coreBudget — both scaled together. */
const FOAM_BUDGET_BASE = { coreBudget: FOAM_FLAGS.coreBudget, spawnPerFrame: 120 };
let foamScaleApplied = 1;
function applyFoamBudget() {
  if (!foam?.params) return;
  const fps = harness.fps;
  let target;
  if (fps > 55) target = 1;
  else if (fps < 45) target = Math.max(0.3, fps / 60);
  else target = foamScaleApplied; // hysteresis band — hold
  if (Math.abs(target - foamScaleApplied) < 0.05) return;
  foamScaleApplied = target;
  FOAM_FLAGS.coreBudget = Math.max(20, Math.round(FOAM_BUDGET_BASE.coreBudget * target));
  foam.params.spawnPerFrame = Math.max(20, Math.round(FOAM_BUDGET_BASE.spawnPerFrame * target));
  window.pushDbg?.(`[creek] foam budget ×${target.toFixed(2)} (fps ${fps.toFixed(0)})`);
}

/* ======================================================================
 * Async-sim interpolation hookup (S1 pacing fix)
 * The worker sim completes step batches at ~35 Hz @26k particles while the
 * display runs at 60 Hz — without interpolation the water visibly steps.
 *
 * STRATEGY (probed once at startup, priority order):
 *   0. sim.posExtrap is a function (Lane B) → hybrid interp+velocity
 *      extrapolation, FIRST choice: bind our scratch buffer as .pos (via
 *      configurable-property redefine — async-sim's .pos getter is
 *      configurable) and fill it each tick with posExtrap(buf, τ), where
 *      τ = time since the last sim frame landed, clamped to 0.6·(1/simHz).
 *      Steady flow then moves at FULL render rate from a ~25 Hz sim; raw
 *      SAB halves stay reachable via .posCurr/.posPrev for passthroughs.
 *   1. sim.posLerp is a function → call async-sim-owned lerp each tick
 *      (the interpolation lane's first-class API, if it lands).
 *   2. 'posLerpTarget' in sim → hand the sim our scratch Float32Array and
 *      let IT fill it between frames; we do nothing per tick.
 *   3. sim.posPrev + sim.alpha present AND .pos is a plain writable data
 *      property → swap in our own buffer and lerp scene-side: on each new
 *      completed frame, buf = prev + (raw − prev) · alpha. screen-fluid's
 *      uploadPositions() reads sim.pos live every render, so every consumer
 *      (screen points, metaballs surface, foam, emit loop) sees lerped
 *      positions with zero monkey-patching of water-pack internals.
 * If none apply (sync sim, or the interpolation lane hasn't landed yet),
 * this degrades gracefully to plain sim.pos passthrough — no behavior
 * change, just a debug log line.
 *
 * LIMITS: lerp is skipped on ticks where the particle count changed since
 * the last frame (spawned/drained particles have no valid prev position) —
 * those frames pass raw positions straight through. alpha semantics are
 * whatever async-sim defines (expected: fraction of the current step batch
 * elapsed); a torn SAB read while the worker finishes a batch is possible
 * but cosmetic and self-corrects next tick.
 * ==================================================================== */
const INTERP = {
  mode: null, buf: null, raw: null, lastFrame: -1, lastCount: -1,
  lastFrameT: 0,    // Lane B: performance.now() when the curr sim frame landed
  emaIntervalMs: 0, // Lane B: local EMA of sim-frame arrivals → simHz for τ clamp
};
(function setupInterp() {
  const s = pack.sim;
  if (!s.isAsync || !s.pos) return;
  // Lane B FIRST choice: hybrid interpolation + capped velocity extrapolation.
  // Bind INTERP.buf as .pos so every consumer (screen-fluid uploadPositions,
  // metaballs, foam, emit loop) sees full-rate positions; async-sim's `.pos`
  // is a configurable getter (or a plain data property on other sims), so the
  // redefine is safe — raw halves remain reachable via .posCurr/.posPrev.
  if (typeof s.posExtrap === 'function') {
    const desc = Object.getOwnPropertyDescriptor(s, 'pos');
    const bindable = !desc || desc.configurable === true || desc.writable === true;
    if (bindable) {
      try {
        INTERP.buf = new Float32Array(s.maxParticles * 3);
        // seed with the current frame so consumers never see zeros pre-tick
        INTERP.buf.set(s.pos.subarray(0, Math.min((s.count | 0) * 3, INTERP.buf.length)));
        Object.defineProperty(s, 'pos', {
          get() { return INTERP.buf; },
          set(v) { INTERP.buf = v; }, // keep any legacy `sim.pos = x` harmless
          configurable: true,
        });
        INTERP.mode = 'posExtrap';
        window.pushDbg?.('[creek] interpolation active: posExtrap (hybrid lerp + gated velocity extrapolation)');
        return;
      } catch { /* redefine refused — fall through to the lerp chain */ }
    }
  }
  if (typeof s.posLerp === 'function') { INTERP.mode = 'posLerp'; return; }
  if ('posLerpTarget' in s) {
    INTERP.buf = new Float32Array(s.maxParticles * 3);
    try {
      s.posLerpTarget = INTERP.buf;
      INTERP.mode = 'posLerpTarget';
      return;
    } catch { /* read-only slot — fall through to prev/alpha */ }
  }
  if ('posPrev' in s && 'alpha' in s) {
    const desc = Object.getOwnPropertyDescriptor(s, 'pos');
    // Only safe when .pos is a plain writable data property (async-sim's SAB
    // view today). A future getter-only .pos means we can't swap — degrade.
    if (desc?.writable && !desc.get) {
      INTERP.raw = s.pos;
      INTERP.buf = new Float32Array(s.maxParticles * 3);
      s.pos = INTERP.buf;
      INTERP.mode = 'prevAlphaSwap';
      window.pushDbg?.('[creek] interpolation active: posPrev+alpha scene-side lerp');
      return;
    }
  }
  window.pushDbg?.('[creek] no interpolation source on async-sim — plain pos passthrough');
})();

/** Refresh interpolated positions. Runs at the TOP of onFixed, BEFORE
 * pack.step posts the next batch — and since the harness runs fixed
 * callbacks before renderFrame, the renderer always draws the latest
 * completed sim frame blended toward the in-flight one. */
function updateInterpolatedPositions() {
  const s = pack.sim;
  switch (INTERP.mode) {
    case 'posExtrap': { // Lane B: hybrid lerp + gated velocity extrapolation
      const count = s.count | 0;
      const n3 = count * 3;
      const nowMs = performance.now();
      if (s.frame !== INTERP.lastFrame) {
        // new sim frame landed: refresh the τ origin + local arrival-rate EMA
        if (INTERP.lastFrameT > 0 && nowMs > INTERP.lastFrameT) {
          const iv = nowMs - INTERP.lastFrameT;
          INTERP.emaIntervalMs = INTERP.emaIntervalMs > 0
            ? INTERP.emaIntervalMs * 0.85 + iv * 0.15 : iv;
        }
        INTERP.lastFrameT = nowMs;
        INTERP.lastFrame = s.frame;
      }
      // τ = time since the last sim frame, clamped to 0.6·(1/simHz)
      const hz = INTERP.emaIntervalMs > 0 ? 1000 / INTERP.emaIntervalMs : 30;
      const tauMax = 0.6 / Math.max(5, hz);
      let tau = (nowMs - INTERP.lastFrameT) / 1000;
      if (!(tau >= 0)) tau = 0;
      else if (tau > tauMax) tau = tauMax;
      if (n3 > 0 && count === INTERP.lastCount) {
        try { s.posExtrap(INTERP.buf, tau); } catch { /* cosmetic: hold last buf */ }
      } else if (n3 > 0 && s.posCurr) {
        // spawn/drain teleport frame: raw passthrough (no valid prev/vel pair)
        INTERP.buf.set(s.posCurr.subarray(0, Math.min(n3, INTERP.buf.length)));
      }
      INTERP.lastCount = count;
      break;
    }
    case 'posLerp': // async-sim owns the fill entirely
      try { s.posLerp(); } catch { /* cosmetic */ }
      break;
    case 'posLerpTarget': // async-sim writes into INTERP.buf between frames
      break;
    case 'prevAlphaSwap': {
      const count = s.count | 0;
      const n3 = count * 3;
      const fresh = s.frame !== INTERP.lastFrame;
      INTERP.lastFrame = s.frame;
      const prev = s.posPrev;
      const alpha = Number(s.alpha);
      if (fresh && n3 > 0 && prev && Number.isFinite(alpha) && alpha >= 0 && alpha <= 1 &&
          count === INTERP.lastCount) {
        for (let i = 0; i < n3; i++) {
          INTERP.buf[i] = prev[i] + (INTERP.raw[i] - prev[i]) * alpha;
        }
      } else if (n3 > 0) {
        INTERP.buf.set(INTERP.raw.subarray(0, n3)); // passthrough fallback
      }
      INTERP.lastCount = count;
      break;
    }
    default: break; // null — sync sim or no source: nothing to do
  }
}

/* ---- effective sim Hz (HUD pacing metric) ----------------------------
 * Worker batches complete asynchronously at their own cadence (~35 Hz @26k);
 * counting completed frames per wall second makes pacing observable instead
 * of vibes. Sync sims have no .frame counter → report the display fps
 * (they step synchronously inside the fixed tick anyway). */
let simHz = 0, hzLastFrame = 0, hzLastT = performance.now();
setInterval(() => {
  const fr = typeof pack.sim.frame === 'number' ? pack.sim.frame : null;
  const now = performance.now();
  const el = (now - hzLastT) / 1000;
  if (fr != null) {
    if (el > 0) simHz = Math.max(0, fr - hzLastFrame) / el;
    hzLastFrame = fr;
  } else {
    simHz = harness.fps;
  }
  hzLastT = now;
  applyFoamBudget();
}, 1000);

/* outflow drain region: downstream end (x > +16.5) across the channel.
 * z-extent covers the meander over that reach plus a generous bank margin —
 * water that overspreads the banks near the outlet should still recycle
 * (observed: particles spread to z +10 at the outlet, past the old ±(meander+2)
 * window, pooling instead of recycling). */
const drainRegion = (() => {
  let zn = Infinity, zx = -Infinity;
  for (let x = 16.5; x <= 20; x += 0.05) {
    zn = Math.min(zn, channelZ(x));
    zx = Math.max(zx, channelZ(x));
  }
  const m = HALF_W + 6.0;
  return { min: [16.5, -3, Math.min(zn - m, -11)], max: [20, 4, Math.max(zx + m, 11)] }; // y band spans the whole bed depth
})();

/* ======================================================================
 * Controls: inflow emitter (pour), clear, render mode
 * ==================================================================== */
const EMIT_X = -18;
const ctrl = {
  pour: false,
  rate: 450,
  jetSpeed: 0.18, // inflow m/s along channel tangent — keep gentle (see emit loop)
  clear: () => pack.sim.reset(),
};
{
  const qp = new URLSearchParams(location.search);
  if (qp.get('pour') != null) {
    ctrl.pour = true;
    const r = parseFloat(qp.get('pour'));
    if (Number.isFinite(r)) ctrl.rate = r;
  }
}
const f = harness.gui.addFolder('🏞 Creek Flow');
f.add(ctrl, 'pour').name('inflow on/off');
f.add(ctrl, 'rate', 200, 900, 10).name('flow rate /s');
f.add(ctrl, 'jetSpeed', 0.05, 0.6, 0.01).name('inflow speed m/s');
const renderModeCtrl = { renderMode: 'screen' }; // matches the createWaterPack default
f.add(renderModeCtrl, 'renderMode', ['auto', 'metaballs', 'screen'])
  .name('render mode')
  .onChange((m) => Promise.resolve(pack.setRenderMode(m)).catch(
    (e) => window.pushDbg?.(`[creek] setRenderMode('${m}') failed: ${e?.message ?? e}`)));
f.add(ctrl, 'clear').name('clear');
f.close();

/* HUD + metrics */
let drainedTotal = 0;
let hudTick = 0;
const renderLabel = () => `${pack.screenState.active ? 'screen' : 'metaballs'}${pack.screenState.requested !== pack.screenState.active ? (pack.screenState.requested === 'screen' ? '*' : '') : ''}`;
harness.setHudProvider(() => [
  `<b>Creek Lab</b> — meandering channel, 2% grade, h=0.35`,
  `fps <b>${harness.fps.toFixed(0)}</b>  sim <b>${simHz.toFixed(0)}Hz</b> ${pack.sim.simMs.toFixed(1)}ms  particles <b>${pack.sim.count}</b>`,
  `render <b>${renderLabel()}</b>  interp <b>${INTERP.mode === 'posExtrap' ? 'hybrid extrap+lerp' : (INTERP.mode ?? 'off')}</b>`,
  `inflow <b>${ctrl.pour ? ctrl.rate : 0}</b>/s @ x=${EMIT_X}  drained <b>${drainedTotal}</b>`,
  `leaked: <b>${pack.sim.leakedTotal ?? 0}</b>  KE: ${(pack.sim.kineticEnergy ?? 0).toFixed(0)}  foam ×${foamScaleApplied.toFixed(2)}`,
]);

setupMetrics(() => ({
  scene: 'creek', particles: pack.sim.count, simMs: +pack.sim.simMs.toFixed(2),
  flowRate: ctrl.pour ? ctrl.rate : 0, drained: drainedTotal,
  leaked: pack.sim.leakedTotal ?? 0,
  kineticEnergy: +(pack.sim.kineticEnergy ?? 0).toFixed(1),
}));

setupAutoShots(harness.renderer, 4);

/* fixed-step loop: interpolate → emit → step colliders → drain → count recycle */
let emitAcc = 0;
harness.onFixed((dt) => {
  updateInterpolatedPositions(); // BEFORE pack.step posts the next batch
  if (ctrl.pour) {
    emitAcc += ctrl.rate * dt;
    while (emitAcc >= 1 && pack.sim.count < pack.sim.p.maxParticles) {
      const jx = (Math.random() - 0.5) * 0.8;
      const jz = (Math.random() - 0.5) * 0.8;
      const x = EMIT_X + jx, z = channelZ(EMIT_X) + jz;
      const bedY = terrainH(x, z);
      // Spawn AT THE WATER SURFACE, not at the bed: spawning deep inside the
      // pressurized column detonated the near-pressure term and launched new
      // particles upward like geysers. Entering from just above the surface
      // lets them settle gently into the flow.
      // Find local surface: highest particle within ±1 m of the spawn point.
      // Find local surface: highest particle within ±1 m of the spawn point,
      // capped at bedY + MAX_COLUMN — otherwise fountain particles (launched
      // skyward by an earlier detonation) get picked as "surface" and each new
      // spawn starts even higher: a positive-feedback geyser.
      let surfY = bedY; // fall back to bed when no water yet
      const s = pack.sim;
      const colCap = bedY + 1.5;
      for (let i = 0; i < s.count; i += 3) {
        const px = s.pos[i * 3], pz = s.pos[i * 3 + 2], py = s.pos[i * 3 + 1];
        if (px > x - 1 && px < x + 1 && pz > z - 1 && pz < z + 1 && py > surfY && py < colCap) surfY = py;
      }
      const tx = 1, tz = channelDz(x);
      const len = Math.hypot(tx, tz);
      const jet = ctrl.jetSpeed; // m/s along the tangent
      pack.sim.spawn(x, Math.max(bedY + 0.12, surfY + 0.05), z,
        (tx / len) * jet, -0.02, (tz / len) * jet);
      emitAcc--;
    }
  }
  pack.step(dt, [heightCollider]);
  try { foam?.update?.(dt, FOAM_FLAGS); } catch (e) { /* foam is cosmetic */ }
  // outflow recycle: remove anything that reached the downstream drain
  const before = pack.sim.count;
  pack.sim.drain(drainRegion);
  drainedTotal += before - pack.sim.count;
});

window.__dbg = { pack, harness, heights, drainRegion, channelZ, terrainH };
window.pushDbg?.('Creek Lab ready');
harness.start();
