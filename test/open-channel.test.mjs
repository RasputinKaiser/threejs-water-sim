// test/open-channel.test.mjs — flow down an inclined channel against open-
// channel hydraulics: with no bed drag the water accelerates at exactly g·S
// (the solver and the walls dissipate nothing), with quadratic bed drag C_f
// it follows u(t) = u∞·tanh(t·gS/u∞), u∞ = √(g·S·R/C_f).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createSimulation } from '../src/water/sim.js';

const S = 0.05, H = 0.3, L = 12, W = 0.6, g = 9.81, th = Math.atan(S), gS = g * Math.sin(th);

async function channel(friction) {
  const sim = await createSimulation({
    solver: 'dfsph', spacing: 0.1, maxParticles: 8000, maxDiffuse: 0, friction,
    gravity: [gS, -g * Math.cos(th), 0],   // tilted gravity: a flat bed on slope S
  }, { threads: 0 });
  sim.setColliders([
    { type: 'plane', position: [0, 0, 0] },
    { type: 'box', position: [0, 0.5, -W / 2 - 0.5], size: [L / 2 + 1, 1, 0.5] },
    { type: 'box', position: [0, 0.5, W / 2 + 0.5], size: [L / 2 + 1, 1, 0.5] },
    { type: 'box', position: [-L / 2 - 0.5, 0.5, 0], size: [0.5, 1, 1] },
    { type: 'box', position: [L / 2 + 0.5, 0.5, 0], size: [0.5, 1, 1] },
  ]);
  sim.fillBox([-L / 2, 0, -W / 2], [L / 2, H, W / 2], { jitter: 0 });
  return sim;
}

// mean downstream velocity of the middle section (end walls not felt yet)
function middleSpeed(sim) {
  const p = sim.positions, v = sim.velocities;
  let u = 0, k = 0;
  for (let i = 0; i < sim.count; i++) if (Math.abs(p[i * 3]) < 2) { u += v[i * 3]; k++; }
  return u / k;
}

test('frictionless channel: the water accelerates at g·S', async () => {
  const sim = await channel(0);
  for (let f = 0; f < 90; f++) sim.stepNow(1);
  const u = middleSpeed(sim), want = gS * 1.5;
  assert.ok(Math.abs(u - want) / want < 0.03, `u ${u.toFixed(3)} m/s after 1.5 s, g·S·t = ${want.toFixed(3)}`);
});

test('bed drag follows the uniform-flow approach u∞·tanh(t·gS/u∞)', async () => {
  const Cf = 0.02, R = H * W / (W + 2 * H), uInf = Math.sqrt(gS * R / Cf);
  const sim = await channel(Cf);
  for (let f = 0; f < 120; f++) sim.stepNow(1);
  const u = middleSpeed(sim), want = uInf * Math.tanh(2 * gS / uInf), free = gS * 2;
  assert.ok(Math.abs(u - want) / want < 0.05, `u ${u.toFixed(3)} m/s after 2 s, theory ${want.toFixed(3)} (frictionless ${free.toFixed(3)})`);
});
