// water-pack/index.js — public API of the water pack.
//
//   import { createWaterPack } from './water-pack/index.js';
//   const water = createWaterPack({ scene, bounds, gui });
//   water.sim.spawnBlock(...); water.step(dt);
//
// Exports: solver (WaterSim), surface (createWaterSurface), probe
// (createFillProbe/createRegionCensus), box3d adapter (createBox3DWater),
// the agent metrics pipeline (setupMetrics), optional render-lane loading
// (loadWaterLane), the screen-space render bridge (createScreenBridge), and
// the opt-in off-thread solver via createWaterPack({ workerSim: true }) —
// falls back to sync unless workerFallback:false (requires COOP/COEP).

export { WaterSim, DEFAULT_PARAMS } from './solver.js';
export { createWaterSurface } from './surface.js';
export { createFillProbe, createRegionCensus } from './probe.js';
export { createBox3DWater } from './box3d-adapter.js';
export { setupMetrics } from './metrics.js';

// Convenience: full three.js-side pack (sim + surface + optional gui).
import * as THREE from 'three';
import { WaterSim } from './solver.js';
import { createWaterSurface } from './surface.js';
import { maybeCreateAsyncSim } from './async-sim.js';

/* ======================================================================
 * Optional render lanes (screen-fluid / effects) — graceful fallback
 * ======================================================================
 * These modules are authored on a concurrent lane and may be missing or
 * broken at any point in time. import.meta.glob resolves to {} when the file
 * doesn't exist (build stays green) and bundles it once it does. Never import
 * them statically.
 */
const LANE_GLOBS = {
  'screen-fluid': import.meta.glob('./screen-fluid.js'),
  'effects': import.meta.glob('./effects.js'),
};

/** Load an optional lane module by name ('screen-fluid' | 'effects').
 * Resolves to the module namespace, or null when absent/unloadable. */
export async function loadWaterLane(name) {
  const loaders = LANE_GLOBS[name];
  if (!loaders) return null;
  const key = Object.keys(loaders)[0];
  if (!key) return null;
  try {
    return await loaders[key]();
  } catch (e) {
    window.pushDbg?.(`[water-pack] lane '${name}' failed to load: ${e?.message ?? e}`);
    return null;
  }
}

/* ======================================================================
 * Screen-render bridge
 * ======================================================================
 * Integration contract with src/water-pack/screen-fluid.js:
 *
 *   createScreenFluid(sim, renderer, scene, bounds, opts)
 *     → { renderWater(camera), update(camera), addGui(gui),
 *         state:{enabled,...}, dispose() }
 *
 * bounds is the sim world AABB passed as {min:[x,y,z], max:[x,y,z],
 * size:[x,y,z]}. opts receives `colorTarget` (the pack-owned WebGLRenderTarget
 * holding the scene color pass) and `depthTexture` (matching DepthTexture).
 *
 * Frame flow when screen mode is active (via harness.setCompositor):
 *   1. harness renders scene+camera into bridge.rt (offscreen, depth on)
 *   2. bridge blits rt.texture as a fullscreen backdrop quad
 *   3. bridge calls fluid.renderWater(camera) which composites water on top
 */
export function createScreenBridge(sim, {
  renderer = null, scene = null,
  bounds = null,          // world AABB {min:[], max:[]}
  surfaceGroup = null,    // hidden while screen mode composites
} = {}) {
  // compositor install point: anything exposing setCompositor(fn) — typically
  // the debug harness. Scenes attach it via bridge.setSink(harness).
  let sink = null;
  const state = { requested: 'metaballs', active: false, screenMs: null, error: null };
  let fluid = null;
  let rt = null, depthTex = null, quadScene = null, quadCam = null, quadMat = null;

  function ensureRT() {
    const w = renderer.domElement.width, h = renderer.domElement.height;
    if (!rt) {
      rt = new THREE.WebGLRenderTarget(w, h);
      depthTex = new THREE.DepthTexture(w, h);
      rt.depthTexture = depthTex;
      quadMat = new THREE.MeshBasicMaterial({ map: rt.texture, depthTest: false, depthWrite: false });
      quadScene = new THREE.Scene();
      quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), quadMat));
      quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    } else if (rt.width !== w || rt.height !== h) {
      rt.setSize(w, h);
      depthTex.dispose();
      depthTex = new THREE.DepthTexture(w, h);
      rt.depthTexture = depthTex;
      quadMat.map = rt.texture; // setSize allocates a fresh color texture
    }
    return rt;
  }

  function composite(rnd, sc, cam) {
    if (!cam) return; // no camera yet — skip this frame rather than throw
    const t0 = performance.now();
    const target = ensureRT();
    // Make sure camera matrices exist before any consumer reads them
    // (three populates projectionMatrix/matrixWorldInverse during render,
    // but the very first composite can run before that).
    cam.updateMatrixWorld();
    if (!cam.projectionMatrix?.elements) cam.updateProjectionMatrix();
    rnd.setRenderTarget(target);
    rnd.render(sc, cam); // includes clear (renderer.autoClear default true)
    rnd.setRenderTarget(null);
    rnd.render(quadScene, quadCam); // backdrop = scene color buffer
    fluid?.update?.(cam);
    fluid?.renderWater?.(cam);
    state.screenMs = performance.now() - t0;
  }

  async function activate() {
    if (fluid) return;
    if (!renderer) throw new Error('no renderer provided to createWaterPack/createScreenBridge');
    const mod = await loadWaterLane('screen-fluid');
    const ctor = mod?.createScreenFluid;
    if (typeof ctor !== 'function') throw new Error('screen-fluid.js not available');
    ensureRT();
    const bmin = bounds?.min ?? sim.bounds.min;
    const bmax = bounds?.max ?? sim.bounds.max;
    fluid = ctor(sim, renderer, scene,
      { min: [...bmin], max: [...bmax], size: [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]] },
      { colorTarget: rt, depthTexture: depthTex });
    fluid.addGui?.(sink?.gui);
  }

  /** Attach the compositor install point (the debug harness). */
  function setSink(s) { sink = s; }

  /** Switch render mode: 'auto'/'screen' (try screen fluid, fall back to
   * metaballs) or 'metaballs'. Returns a promise resolving to the effective
   * mode ('screen' | 'metaballs'). */
  async function setMode(mode) {
    state.requested = mode;
    if (mode === 'screen' || mode === 'auto') {
      try {
        await activate();
      } catch (e) {
        state.error = String(e?.message ?? e);
        window.pushDbg?.(`[water-pack] screen render unavailable (${state.error}) → metaballs`);
        state.active = false;
        if (surfaceGroup) surfaceGroup.visible = true;
        sink?.setCompositor?.(null);
        return 'metaballs';
      }
      state.error = null;
      state.active = true;
      if (fluid?.state) fluid.state.enabled = true;
      if (surfaceGroup) surfaceGroup.visible = false;
      sink?.setCompositor?.(composite);
      return 'screen';
    }
    state.active = false;
    if (fluid?.state) fluid.state.enabled = false;
    if (surfaceGroup) surfaceGroup.visible = true;
    sink?.setCompositor?.(null);
    state.screenMs = null;
    return 'metaballs';
  }

  function dispose() {
    sink?.setCompositor?.(null);
    try { fluid?.dispose?.(); } catch { /* best effort */ }
    fluid = null;
    depthTex?.dispose(); rt?.dispose(); quadMat?.dispose();
    rt = depthTex = quadMat = quadScene = quadCam = null;
    state.active = false;
  }

  return { state, setSink, setMode, composite, dispose, get fluid() { return fluid; } };
}

/* ====================================================================== */

export function createWaterPack({
  scene, bounds, surfaceBounds = null, params = {}, gui = null, waterline = null,
  surfaceOpts = {}, substeps = 1,
  // Lane C2 additions:
  renderer = null,       // required for renderMode auto/screen
  renderMode = 'auto',   // 'auto' tries screen-fluid.js, falls back to metaballs
  // Lane P2 additions (opt-in off-thread sim):
  workerSim = false,     // true → run the solver in a Web Worker via SABs
  workerFallback = true, // when workerSim and unsupported: true = silent sync fallback
}) {
  // ---- Sim selection (Lane P2) ------------------------------------------
  // workerSim=true routes the solver through the off-thread WaterSim worker
  // (sim-worker.mjs) with SharedArrayBuffer state views. The async wrapper is
  // call-site compatible with sync WaterSim (.pos/.vel/.nCount/.count/.p/
  // .bounds/.spawn/.spawnBlock/.drain/.step/.setParams/.leakedTotal/
  // .kineticEnergy/.simMs — see async-sim.js). Differences that remain:
  //   • state updates land one frame late (renderer reads last completed frame)
  //   • drain() is fire-and-forget, so `before - sim.count` deltas computed
  //     immediately after the call read as 0 until the next frame lands.
  let sim = null;
  let simIsAsync = false;
  if (workerSim) {
    const asyncSim = maybeCreateAsyncSim({ params });
    if (asyncSim) {
      sim = asyncSim;
      simIsAsync = true;
    } else if (!workerFallback) {
      throw new Error(
        '[water-pack] workerSim=true but crossOriginIsolated/SharedArrayBuffer unavailable ' +
        '(serve with COOP/COEP headers) and workerFallback=false — refusing sync fallback');
    } else {
      window.pushDbg?.('[water-pack] workerSim requested but unsupported → silent sync fallback');
    }
  }
  if (!sim) sim = new WaterSim(params);
  sim.bounds = { min: bounds.min, max: [bounds.min[0] + bounds.size[0], bounds.min[1] + bounds.size[1], bounds.min[2] + bounds.size[2]] };
  // On the async path this assignment forwards a setBounds message to the
  // worker; on the sync path it's the ordinary property write. Either way the
  // local value is readable immediately for surface/bridge construction.
  // surfaceBounds: tighter AABB for MarchingCubes (higher effective resolution
  // around the action). Defaults to `bounds`.
  const sb = surfaceBounds ?? bounds;
  const surface = createWaterSurface(sim, scene, sb, { waterline, ...surfaceOpts });
  if (gui) surface.addGui(gui);

  const bridge = createScreenBridge(sim, {
    renderer, scene,
    bounds: { min: sim.bounds.min, max: sim.bounds.max },
    surfaceGroup: surface.group,
  });

  const pack = {
    sim, surface,
    // substeps: N solver passes of dt/N. Halves the dt² pressure-displacement
    // term per pass — the fix for velocity jitter in small/deep containers
    // (bucket σy 0.10 → 0.04). Costs N× sim time; use 1 for big open scenes.
    //
    // ASYNC PATH (workerSim): posts ONE step message carrying sdt=dt/N and
    // steps=N (the worker loops the N passes off-thread), then IMMEDIATELY
    // calls surface.update() against the last COMPLETED frame's SAB data.
    // The renderer therefore runs one frame behind the solver — invisible at
    // render cadence, and the main thread pays only the postMessage cost
    // instead of N× solve time.
    step(dt, colliders) {
      if (simIsAsync) {
        sim.step(dt / substeps, colliders, substeps);
      } else {
        const sdt = dt / substeps;
        for (let i = 0; i < substeps; i++) sim.step(sdt, colliders);
      }
      surface.update();
      if (bridge.state.active) surface.group.visible = false; // surface.update() re-enables children
    },
    /** Attach the harness (or anything with setCompositor(fn)) so screen mode
     * can install its composite pass. Call once before/after setRenderMode. */
    attachCompositor(harnessLike) {
      bridge.setSink(harnessLike);
      if (bridge.state.active) bridge.setMode('screen'); // reinstall after late attach
    },
    get screenFluid() { return bridge.fluid; },
    screenState: bridge.state, // { requested, active, screenMs, error }
    setRenderMode(mode) { return bridge.setMode(mode); },
    /** 'worker' when the off-thread solver is active, else 'sync'. */
    get simMode() { return simIsAsync ? 'worker' : 'sync'; },
    get simIsAsync() { return simIsAsync; },
    dispose() {
      bridge.dispose();
      if (simIsAsync) { try { sim.dispose(); } catch { /* best effort */ } }
    },
  };

  // ---- Debug GUI: worker-sim indicator (Lane P2) --------------------------
  // The toggle is CREATION-TIME ONLY: switching mid-session would strand the
  // surface/probe/bridge references on the old sim object. So the checkbox
  // reflects the active mode and is disabled, with the why/how in its tooltip.
  if (gui) {
    const workerGui = { 'worker sim': simIsAsync };
    const tip = simIsAsync
      ? 'Worker/SAB sim ACTIVE (created with workerSim:true). Restart with the flag off to switch back.'
      : 'Sync sim. Set createWaterPack({ workerSim: true }) and reload to run the solver off-thread (requires COOP/COEP).';
    const ctrl = gui.add(workerGui, 'worker sim').disable().listen();
    ctrl.$input?.setAttribute('title', tip);
    ctrl.$name?.setAttribute?.('title', tip);
    ctrl.name('worker sim 🔒');
  }

  // Kick off initial mode (async, non-blocking). Falls back silently to
  // metaballs when the screen lane isn't available yet.
  if (renderMode && renderMode !== 'metaballs') {
    Promise.resolve(pack.setRenderMode(renderMode)).catch((e) =>
      window.pushDbg?.(`[water-pack] setRenderMode('${renderMode}') failed: ${e?.message ?? e}`));
  }

  return pack;
}
