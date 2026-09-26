#!/usr/bin/env node
// tools/bench.mjs — solver throughput on this machine: a settled dam-break
// block of N particles in a tank, timed inline and on worker threads.
//
// Run:  node tools/bench.mjs [--particles N] [--steps N] [--threads a,b,…] [--solver dfsph|pbf]
// Prints ms/step, particle-steps per second and (inline) the per-phase split.

import { availableParallelism } from 'node:os';
import { createSimulation } from '../src/water/sim.js';
import { nodeWorkerFactory } from '../test/helpers/node-worker.mjs';

const args = process.argv.slice(2);
const argOf = (name, dflt) => {
  const i = args.indexOf(name);
  return i !== -1 && args[i + 1] !== undefined ? args[i + 1] : dflt;
};
const N = Number(argOf('--particles', 16000));
const STEPS = Number(argOf('--steps', 60));
const cores = availableParallelism();
const THREADS = argOf('--threads', `0,${Math.max(1, cores - 1)}`).split(',').map(Number);
const SOLVER = argOf('--solver', 'dfsph');
const SPACING = 0.1;

// a block of ~N particles against one wall of a tank twice its width
const side = Math.cbrt(N) * SPACING;
const tank = { type: 'container', position: [side, 2 * side, 0], size: [2 * side, 2 * side, side / 2 + 0.2] };

async function bench(threads) {
  const sim = await createSimulation({ spacing: SPACING, maxParticles: N + 1024, solver: SOLVER },
    { threads, workerFactory: nodeWorkerFactory, maxStepsPerFrame: 1 });
  sim.setColliders([tank]);
  sim.fillBox([-side + 0.01, 0, -side / 2], [0.01, side, side / 2], { seed: 1 });
  await sim.ready;
  const step = async () => {
    if (sim.mode === 'inline') { sim.stepNow(1); return; }
    await sim.stepNow(1);
  };
  for (let i = 0; i < 10; i++) await step(); // warm up JIT + first sort
  const t0 = performance.now();
  for (let i = 0; i < STEPS; i++) await step();
  const ms = (performance.now() - t0) / STEPS;
  const phases = sim.mode === 'inline' ? { ms: sim.solver.phaseMs, names: sim.solver.phaseNames } : null;
  const count = sim.count;
  sim.dispose();
  return { mode: sim.mode, threads, count, ms, phases };
}

console.log(`water solver bench — ${SOLVER}, ${cores} logical cores, spacing ${SPACING} m, ${STEPS} steps\n`);
for (const t of THREADS) {
  const r = await bench(t);
  const label = r.mode === 'inline' ? 'inline      ' : `${String(t).padStart(2)} threads  `;
  console.log(`${label} ${r.count} particles  ${r.ms.toFixed(2)} ms/step  ` +
    `${(r.count / r.ms / 1000).toFixed(2)} M particle-steps/s  (${(1000 / r.ms).toFixed(0)} steps/s)`);
  if (r.phases) {
    const parts = [];
    for (const [k, name] of Object.entries(r.phases.names)) if (name !== 'total' && r.phases.ms[k] > 0.005) parts.push(`${name} ${r.phases.ms[k].toFixed(2)}`);
    console.log(`             phases (ms, last step): ${parts.join(' · ')}`);
  }
}
