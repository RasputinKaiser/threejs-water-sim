// scenes/water-harness.js — glue between the lab debug harness and the water
// pack: render hook, fixed-step update with Box3D in lockstep, a fill probe,
// a HUD line and a parameter GUI.

/**
 * Route the harness through the water: rendering via water.render(), and each
 * fixed tick through water.update() (+ Box3D stepped by exactly the fluid
 * time simulated, so bodies and water share one timeline).
 */
export function attachWater(harness, water, { b3 = null, world = null, before = null, after = null } = {}) {
  harness.setCompositor((renderer, scene, camera) => water.render(camera));
  harness.onFixed((dt) => {
    before?.(dt);
    const simDt = water.update(dt);
    if (world && simDt > 0) b3.b3World_Step(world, simDt, 4);
    after?.(dt, simDt);
  });
}

/**
 * Water level and surface flatness inside an axis-aligned region: the top
 * particle of every 2·spacing column, level = mean top + spacing/2 and
 * flatness = standard deviation of the tops (≈ 0 for still water).
 */
export function fillProbe(water, region) {
  const tops = new Map();
  return {
    region,
    measure() {
      const p = water.sim.positions, n = water.count, s = water.params.spacing, col = 2 * s;
      tops.clear();
      let k = 0;
      for (let i = 0; i < n; i++) {
        const x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
        if (x < region.min[0] || x > region.max[0] || z < region.min[2] || z > region.max[2] || y < region.min[1] || y > region.max[1]) continue;
        k++;
        const key = (Math.floor(x / col) + 32768) * 65536 + Math.floor(z / col) + 32768;
        const t = tops.get(key);
        if (t === undefined || y > t) tops.set(key, y);
      }
      if (!k) return { count: 0, level: NaN, flatness: NaN };
      let sum = 0, sum2 = 0;
      for (const t of tops.values()) { sum += t; sum2 += t * t; }
      const mean = sum / tops.size;
      return { count: k, level: mean + s / 2, flatness: Math.sqrt(Math.max(0, sum2 / tops.size - mean * mean)) };
    },
  };
}

/** Standard HUD line: fps, particles, solver mode/threads, step time. */
export function waterHudLine(harness, water) {
  const s = water.stats;
  const err = s.avgDensityError != null
    ? `ρ err ${(s.avgDensityError * 100).toFixed(2)}%/${((s.maxDensityError ?? 0) * 100).toFixed(1)}%  it ${(s.pressureIterations ?? 0).toFixed(0)}×${s.substeps ?? 1}`
    : `ρ err ${((s.maxDensityError ?? 0) * 100).toFixed(1)}%`;
  return `fps <b>${harness.fps.toFixed(0)}</b>  particles <b>${water.count}</b>  ` +
    `${water.params.solver} ${water.mode}${s.threads ? `×${s.threads}` : ''}  step <b>${(s.stepMs ?? 0).toFixed(1)}</b> ms  ${err}`;
}

/** Solver + look controls in a lil-gui folder. */
export function addWaterGui(gui, water, title = '💧 Water') {
  const f = gui.addFolder(title);
  const p = water.params, dfsph = p.solver === 'dfsph';
  const t = {
    viscosity: p.viscosity, vorticity: p.vorticity, cohesion: p.cohesion,
    friction: p.friction, iterations: p.iterations, densityTolerance: p.densityTolerance * 100,
  };
  f.add(t, 'viscosity', 0, 0.2, 0.005).onChange((v) => water.setParams({ viscosity: v }));
  f.add(t, 'vorticity', 0, 0.3, 0.005).onChange((v) => water.setParams({ vorticity: v }));
  if (dfsph) {
    f.add(t, 'friction', 0, 0.05, 0.001).name('bed drag C_f').onChange((v) => water.setParams({ friction: v }));
    f.add(t, 'densityTolerance', 0.02, 1, 0.01).name('density tol. %').onChange((v) => water.setParams({ densityTolerance: v / 100 }));
  } else {
    f.add(t, 'cohesion', 0, 1, 0.01).onChange((v) => water.setParams({ cohesion: v }));
    f.add(t, 'friction', 0, 1, 0.01).onChange((v) => water.setParams({ friction: v }));
    f.add(t, 'iterations', 1, 10, 1).onChange((v) => water.setParams({ iterations: v }));
  }
  const look = water.look;
  if (look) {
    const l = { scatter: look.scatter, roughness: look.roughness, refraction: look.refraction };
    f.add(l, 'scatter', 0, 1, 0.01).name('turbidity').onChange((v) => water.setLook({ scatter: v }));
    f.add(l, 'roughness', 0.01, 0.5, 0.01).onChange((v) => water.setLook({ roughness: v }));
    f.add(l, 'refraction', 0, 0.15, 0.005).onChange((v) => water.setLook({ refraction: v }));
  }
  f.close();
  return f;
}

/**
 * Agent-facing telemetry: with ?metrics=<label> in the URL, POST a JSON
 * snapshot every `interval` s to the shot server (:5185/metrics), which saves
 * it as .metrics/<label>.json. Silent when the server is not running.
 */
export function setupMetrics(getSnapshot, { interval = 2, port = '5185' } = {}) {
  const params = new URLSearchParams(location.search);
  if (!params.has('metrics')) return;
  const label = params.get('metrics') || 'run';
  setInterval(async () => {
    try {
      await fetch(`http://localhost:${port}/metrics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label, snapshot: getSnapshot() }),
      });
    } catch { /* shot server not running */ }
  }, interval * 1000);
}
