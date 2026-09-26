// src/water-pack/bench-report.mjs — perf headroom study for the CURRENT WaterSim.
// Profiles phase-by-phase cost (solver.js PHASES: predict, sort, pairs
// (incl. density + viscosity), relax, scatter, collide, derive) at h=0.35
// across particle counts [8000, 14000, 22000, 26000, 30000], using the
// solver's own per-step sim.phaseMs timings.
//
// Run: node src/water-pack/bench-report.mjs

import { WaterSim, PHASES } from './solver.js';

const H = 0.35;
const COUNTS = [8000, 14000, 22000, 26000, 30000];
const WARMUP = 10;
const TIMED = 60;
const DT = 1 / 60;
const COLLIDERS = [{ type: 'plane', o: [0, 0, 0], n: [0, 1, 0] }];

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : 0.5 * (s[mid - 1] + s[mid]);
}

function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }

// Lattice dims whose product ~= target count (slab-ish shapes like real scenes)
function latticeDims(target) {
  if (target <= 8000) return [20, 20, 20];
  if (target <= 14000) return [28, 25, 20];
  if (target <= 22000) return [41, 29, 18];
  if (target <= 26000) return [45, 31, 19];
  return [49, 33, 19]; // ~30700 requested
}

function profileCount(nTarget) {
  const [nx, ny, nz] = latticeDims(nTarget);
  const sim = new WaterSim({ h: H, maxParticles: 34000 });
  sim.spawnBlock(0, 0.3, 0, nx, ny, nz);
  const spawned = sim.count;
  for (let s = 0; s < WARMUP; s++) sim.step(DT, COLLIDERS);
  const ph = Object.fromEntries(PHASES.map((k) => [k, []]));
  const totals = [], kes = [];
  let npairs = 0, leaked = 0;
  for (let s = 0; s < TIMED; s++) {
    sim.step(DT, COLLIDERS);
    totals.push(sim.simMs);
    for (const k of PHASES) ph[k].push(sim.phaseMs[k]);
    npairs = sim._npairs;
    leaked = sim.leakedTotal ?? 0;
    kes.push(sim.kineticEnergy);
  }
  // NaN check
  let bad = false;
  for (let i = 0; i < sim.count * 3; i++) if (!Number.isFinite(sim.pos[i])) { bad = true; break; }

  const medTotal = median(totals);
  const rows = {};
  let sumPhases = 0;
  for (const k of Object.keys(ph)) {
    const m = median(ph[k]);
    rows[k] = { med: m, pct: (m / medTotal) * 100 };
    sumPhases += m;
  }
  return {
    target: nTarget, dims: `${nx}x${ny}x${nz}`, spawned, count: sim.count,
    medTotal, sumPhases, rows, npairs, leaked,
    keFirst: kes[0], keLast: kes[kes.length - 1],
    nan: bad,
  };
}

console.log(`=== WaterSim phase profile: h=${H}, dt=1/60, plane floor, ${TIMED} timed steps ===`);
console.log('');
const results = [];
for (const n of COUNTS) {
  const r = profileCount(n);
  results.push(r);
  console.log(`--- target ${n} (lattice ${r.dims}) : spawned=${r.spawned} finalCount=${r.count} pairs/step=${r.npairs} leaked=${r.leaked}`);
  console.log(`    TOTAL median ${r.medTotal.toFixed(2)} ms   (sum of phases ${r.sumPhases.toFixed(2)} ms)   KE ${r.keFirst.toFixed(0)}->${r.keLast.toFixed(0)}`);
  for (const [k, v] of Object.entries(r.rows)) {
    console.log(`    ${k.padEnd(8)} ${v.med.toFixed(3).padStart(8)} ms   ${v.pct.toFixed(1).padStart(5)}%`);
  }
  console.log(r.nan ? '    !! NaN/Inf in positions' : '    positions finite: OK');
  console.log('');
}

// scaling summary
console.log('scaling (median total ms vs particles):');
for (const r of results) {
  console.log(`  ${String(r.count).padStart(6)}  ${r.medTotal.toFixed(2).padStart(8)} ms   ${(r.medTotal / (r.count / 1000)).toFixed(3)} ms/k-particles   pairs/particle ${(r.npairs / r.count).toFixed(1)}`);
}
const badAny = results.some(r => r.nan || !Number.isFinite(r.medTotal));
if (badAny) { console.error('FAIL: NaN encountered'); process.exit(1); }
console.log('ALL COUNTS CLEAN');
