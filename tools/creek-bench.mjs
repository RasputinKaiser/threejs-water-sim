#!/usr/bin/env node
// tools/creek-bench.mjs — the Creek, headless: water enters through the
// submerged inlet, runs the 40 m meander past the boulders and leaves at the
// outlet. Reports cost and physical health as the flow develops:
//
//   ms/step      solver wall time per 1/60 s frame
//   in / out     inflow and outflow (L/s); equal once the flow is steady
//   depth, speed water depth and mean downstream speed at the gauge stations
//   ρ err        max / mean density error of the last step
//   vmax, leaked fastest particle, particles lost through the bounds
//
// Run: node tools/creek-bench.mjs [--seconds 30] [--spacing 0.15] [--threads 0]
//                                  [--solver dfsph|pbf] [--every 2] [--prefill 1]
// --prefill 1 (default) starts from the lab's pre-filled channel; 0 starts dry.

import { availableParallelism } from 'node:os';
import { createWater } from '../src/water/index.js';
import * as creek from '../src/scenes/creek-world.js';
import { nodeWorkerFactory } from '../test/helpers/node-worker.mjs';

const args = process.argv.slice(2);
const arg = (name, dflt) => {
  const i = args.indexOf(name);
  return i !== -1 && args[i + 1] !== undefined ? args[i + 1] : dflt;
};
const SECONDS = Number(arg('--seconds', 30));
const SPACING = Number(arg('--spacing', 0.15));
const THREADS = Number(arg('--threads', 0));
const EVERY = Number(arg('--every', 2));
const SOLVER = arg('--solver', undefined);
const PREFILL = arg('--prefill', '1') !== '0';
const DT = 1 / 60;

const water = await createWater({
  threads: THREADS, render: false, workerFactory: nodeWorkerFactory, maxStepsPerFrame: 1,
  params: {
    spacing: SPACING, maxParticles: 65536, friction: Number(arg('--friction', 0.02)),
    bounds: { min: [-20, -4, -12], max: [20, 6, 12] },
    ...(SOLVER ? { solver: SOLVER } : {}),
  },
});
water.addHeightfield(creek.heightfieldDesc(creek.buildHeights()));
for (const b of creek.boulderSpheres()) water.addCollider({ type: 'sphere', position: b.center, radius: b.radius });
const src = water.addSource(creek.inlet());
water.addDrain(creek.OUTLET);
if (PREFILL) water.spawn(creek.channelFill(SPACING));
const sim = water.sim;
const s = water.params.spacing, vol = s * s * s;

console.log(`creek bench — ${water.mode}${sim.threads ? ` ×${sim.threads}` : ''} (${availableParallelism()} cores), ` +
  `solver ${water.params.solver ?? 'pbf'}, spacing ${s} m, inflow ${(Math.PI * src.radius ** 2 * src.speed * 1000).toFixed(0)} L/s nominal`);
console.log('  t(s)  particles  ms/step   in L/s  out L/s   depth@stations (m)        speed@stations (m/s)     ρ err max/mean   vmax  leaked');

function gauges() {
  const p = sim.positions, v = sim.velocities, n = sim.count;
  return creek.STATIONS.map((sx) => {
    const cz = creek.channelZ(sx), tx = 1, tz = creek.channelDz(sx), tl = Math.hypot(tx, tz);
    let top = -Infinity, us = 0, k = 0;
    for (let i = 0; i < n; i++) {
      const x = p[i * 3], z = p[i * 3 + 2];
      if (Math.abs(x - sx) > 0.3 || Math.abs(z - cz) > 0.5) continue;
      top = Math.max(top, p[i * 3 + 1]);
      us += (v[i * 3] * tx + v[i * 3 + 2] * tz) / tl; k++;
    }
    const bed = creek.terrainH(sx, cz);
    return { depth: k ? Math.max(0, top + s / 2 - bed) : 0, speed: k ? us / k : 0 };
  });
}

let t = 0, spawned = 0, lastSpawned = 0, lastDrained = 0, msSum = 0, msN = 0;
const countBefore = () => sim.count;
for (let f = 1; t < SECONDS - 1e-9; f++) {
  const n0 = countBefore();
  const simDt = water.update(DT);
  if (sim.mode !== 'inline') {
    await new Promise((r) => setTimeout(r, 0));
    while (sim.busy) await new Promise((r) => setTimeout(r, 0));
  }
  t += simDt;
  msSum += sim.stepMs; msN++;
  if (Math.abs(t / EVERY - Math.round(t / EVERY)) < DT / 2 / EVERY && simDt > 0) {
    const st = water.stats;
    const drained = st.drained ?? 0;
    const g = gauges();
    let vmax = 0;
    for (let i = 0; i < sim.count; i++) vmax = Math.max(vmax, Math.hypot(sim.velocities[i * 3], sim.velocities[i * 3 + 1], sim.velocities[i * 3 + 2]));
    const outRate = (drained - lastDrained) * vol / EVERY * 1000;
    const inRate = (water.stats.spawned ?? NaN);
    lastDrained = drained;
    console.log(`${t.toFixed(1).padStart(6)}  ${String(sim.count).padStart(9)}  ${(msSum / msN).toFixed(2).padStart(7)}  ` +
      `${Number.isFinite(inRate) ? ((inRate - lastSpawned) * vol / EVERY * 1000).toFixed(0).padStart(7) : '      –'}  ${outRate.toFixed(0).padStart(7)}   ` +
      `${g.map((x) => x.depth.toFixed(2)).join(' ').padEnd(24)}  ${g.map((x) => x.speed.toFixed(2)).join(' ').padEnd(24)}  ` +
      `${((st.maxDensityError ?? 0) * 100).toFixed(1)}%/${((st.avgDensityError ?? NaN) * 100).toFixed(2)}%  ${vmax.toFixed(1).padStart(5)}  ${st.leaked ?? 0}`);
    if (Number.isFinite(inRate)) lastSpawned = inRate;
    msSum = 0; msN = 0;
  }
  void n0; void spawned;
}
if (sim.mode === 'inline') {
  const { phaseMs: ms, phaseNames: names } = sim.solver, parts = [];
  for (const [k, name] of Object.entries(names)) if (name !== 'total' && ms[k] > 0.005) parts.push(`${name} ${ms[k].toFixed(2)}`);
  console.log(`phases (ms, last step): ${parts.join(' · ')}`);
}
water.dispose();
