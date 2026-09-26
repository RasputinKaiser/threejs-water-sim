// test/gpu/check.js — runs in a browser (tools/gpu-check.mjs): the WebGPU
// backend against the CPU solver on the same scenarios.
import { createSimulation } from '../../src/water/sim.js';

const log = (s) => { document.getElementById('log').textContent += s + '\n'; console.log(s); };
const params = new URLSearchParams(location.search);
const scenarios = (params.get('s') ?? 'column,dam').split(',');

async function run(backend, scenario) {
  const ww = scenario === 'plunge';
  const sim = await createSimulation({ solver: 'dfsph', spacing: 0.1, maxParticles: 16384, maxDiffuse: ww ? 16384 : 0 },
    { backend, threads: 0, maxStepsPerFrame: 1 });
  let frames = 120;
  if (scenario === 'column') {
    sim.setColliders([{ type: 'container', position: [0, 1, 0], size: [0.6, 1, 0.6] }]);
    sim.fillBox([-0.6, 0, -0.6], [0.6, 1.2, 0.6], { jitter: 0 });
  } else if (scenario === 'dam') {
    sim.setColliders([{ type: 'container', position: [1, 1, 0], size: [2, 1, 0.6] }]);
    sim.fillBox([-1, 0, -0.6], [0, 1.2, 0.6], { jitter: 0 });
    frames = 90;
  } else if (scenario === 'buoy') {
    sim.setColliders([{ type: 'container', position: [0, 0.75, 0], size: [0.75, 0.75, 0.75] },
      { type: 'sphere', position: [0, 0.6, 0], radius: 0.25, dynamic: true, slot: 1, mass: 1e9 }]);
    sim.fillBox([-0.75, 0, -0.75], [0.75, 1.2, 0.75], { jitter: 0 });
    frames = 150;
  } else if (scenario === 'plunge') {
    // the whitewater test (test/whitewater.test.mjs): a block dropped into a pool
    sim.setColliders([{ type: 'container', position: [0, 1, 0], size: [0.8, 1, 0.8] }]);
    sim.fillBox([-0.8, 0, -0.8], [0.8, 0.5, 0.8], { jitter: 0 });
    frames = 120;
  }
  const peak = [0, 0, 0];
  let still = 0;
  const t0 = performance.now();
  for (let f = 0; f < frames; f++) {
    if (f === 90 && scenario === 'buoy') sim.takeImpulses();
    if (ww && f === 30) {
      still = sim.diffuse.count;
      sim.fillBox([-0.3, 1.2, -0.3], [0.3, 1.8, 0.3], { velocity: [0, -3, 0] });
    }
    if (sim.mode === 'gpu') await sim.stepNow(1); else sim.stepNow(1);
    if (ww) {
      const { count, data } = sim.diffuse, t = [0, 0, 0];
      for (let k = 0; k < count; k++) t[Math.floor(data[k * 4 + 3])]++;
      for (let q = 0; q < 3; q++) peak[q] = Math.max(peak[q], t[q]);
    }
  }
  const ms = (performance.now() - t0) / frames;
  const n = sim.count, p = sim.positions, v = sim.velocities;
  let mean = 0, top = -1e9, xmax = -1e9, ke = 0;
  for (let i = 0; i < n; i++) { mean += p[i * 3 + 1]; top = Math.max(top, p[i * 3 + 1]); xmax = Math.max(xmax, p[i * 3]); ke += v[i * 3] ** 2 + v[i * 3 + 1] ** 2 + v[i * 3 + 2] ** 2; }
  const out = { backend: sim.mode, n, mean: mean / n, top, xmax, rms: Math.sqrt(ke / n), ms, err: sim.stats.maxDensityError, avg: sim.stats.avgDensityError, it: sim.stats.pressureIterations, leaked: sim.stats.leaked };
  if (scenario === 'buoy') {
    const { impulses, time } = sim.takeImpulses();
    out.buoyancy = impulses[1 * 6 + 1] / time / (1000 * 9.81 * (4 / 3) * Math.PI * 0.25 ** 3);
  }
  if (ww) { out.still = still; out.peak = peak; out.final = sim.diffuse.count; }
  sim.dispose?.();
  return out;
}

const result = {};
try {
  for (const s of scenarios) {
    for (const backend of ['cpu', 'gpu']) {
      const r = await run(backend, s);
      result[`${s}/${backend}`] = r;
      log(`${s.padEnd(7)} ${r.backend.padEnd(7)} n=${r.n} mean y ${r.mean.toFixed(3)} top ${r.top.toFixed(3)} xmax ${r.xmax.toFixed(2)} rms ${r.rms.toFixed(3)} ` +
        `ρ err max ${((r.err ?? 0) * 100).toFixed(2)}% avg ${((r.avg ?? 0) * 100).toFixed(3)}% it ${r.it} leaked ${r.leaked} ${r.buoyancy != null ? `buoyancy ${r.buoyancy.toFixed(3)}×ρgV ` : ''}${r.peak ? `whitewater still ${r.still} peak spray/foam/bubble ${r.peak.join('/')} final ${r.final} ` : ''}${r.ms.toFixed(1)} ms/frame`);
    }
  }
} catch (e) { log('ERROR ' + (e.stack ?? e)); result.error = String(e.stack ?? e); }
window.__result = result;
