// water-pack/fuzz/fuzz-harness.mjs — invariant checker + scenario runner for
// property-based fuzzing of WaterSim (Lane A).
//
// checkInvariants(sim, label, ctx) returns an array of violation strings
// (empty = clean). Enforced properties mirror the guarantees documented in
// solver.js's header/DEFAULT_PARAMS:
//   1. all pos/vel entries finite        (max 5 violations reported)
//   2. count within [0, maxParticles]
//   3. no particle speed exceeds maxSpeed * 1.01   (squared compare, no sqrt)
//   4. KE explosion guard: kineticEnergy must never exceed the previous
//      finite KE by more than 100x in a single step
//
// runScenario(sim, steps, everyN, cb) drives `sim` at dt=1/60 against a plane
// floor, calling cb(stepIndex, sim) BEFORE each step (scenarios mutate there)
// and checking invariants every `everyN` steps. Returns accumulated violation
// strings prefixed with the step number they occurred on.

export const FUZZ_DT = 1 / 60;

// plane floor shared by every scenario
export const FLOOR = [{ type: 'plane', o: [0, 0, 0], n: [0, 1, 0] }];

export function checkInvariants(sim, label = '', ctx = {}) {
  const out = [];
  const tag = label ? `${label}: ` : '';
  const n = sim.count;

  // 1. finiteness of pos and vel (cap at 5 reports)
  let badPos = 0, badVel = 0;
  for (let i = 0; i < n * 3; i++) {
    if (!Number.isFinite(sim.pos[i])) { badPos++; if (badPos <= 5) out.push(`${tag}non-finite pos[${i}] = ${sim.pos[i]}`); }
    if (!Number.isFinite(sim.vel[i])) { badVel++; if (badVel <= 5) out.push(`${tag}non-finite vel[${i}] = ${sim.vel[i]}`); }
  }
  if (badPos > 5) out.push(`${tag}...and ${badPos - 5} more non-finite pos entries`);
  if (badVel > 5) out.push(`${tag}...and ${badVel - 5} more non-finite vel entries`);

  // 2. count sanity
  if (!Number.isInteger(n) || n < 0 || n > sim.p.maxParticles) {
    out.push(`${tag}count ${n} outside [0, ${sim.p.maxParticles}]`);
  }

  // 3. speed clamp: squared per-particle magnitude vs (maxV*1.01)^2
  const maxV = sim.p.maxSpeed;
  const lim2 = maxV * 1.01 * (maxV * 1.01);
  let speedViol = 0;
  for (let i = 0; i < n; i++) {
    const vx = sim.vel[i * 3], vy = sim.vel[i * 3 + 1], vz = sim.vel[i * 3 + 2];
    const sp2 = vx * vx + vy * vy + vz * vz;
    if (sp2 > lim2) {
      speedViol++;
      if (speedViol <= 5) {
        out.push(`${tag}particle ${i} speed² ${sp2.toFixed(1)} > (${maxV}×1.01)²=${lim2.toFixed(1)}`);
      }
    }
  }
  if (speedViol > 5) out.push(`${tag}...and ${speedViol - 5} more over-speed particles`);

  // 4. KE explosion guard (relative to last known finite KE)
  if (ctx.lastKE !== undefined && ctx.lastKE !== null && ctx.lastKE > 0 &&
      Number.isFinite(sim.kineticEnergy) && sim.kineticEnergy > ctx.lastKE * 100) {
    out.push(`${tag}KE explosion: ${sim.kineticEnergy.toFixed(1)} > 100 × lastKE ${ctx.lastKE.toFixed(1)}`);
  }

  return out;
}

export function runScenario(sim, steps, everyN, cb) {
  const violations = [];
  let lastKE;
  for (let s = 0; s < steps; s++) {
    if (cb) cb(s, sim);
    sim.step(FUZZ_DT, FLOOR);
    if ((s + 1) % everyN === 0 || s === steps - 1 || !Number.isFinite(sim.kineticEnergy)) {
      const vs = checkInvariants(sim, '', { lastKE });
      for (const v of vs) violations.push(`step=${s}: ${v}`);
    }
    // track only finite KE so invariant 4 compares against a sane baseline
    if (Number.isFinite(sim.kineticEnergy)) lastKE = sim.kineticEnergy;
  }
  return violations;
}
