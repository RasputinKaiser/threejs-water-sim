// water/core/whitewater.js — diffuse whitewater particles (spray, foam and
// bubbles) after Ihmsen et al. 2012, "Unified Spray, Foam and Bubbles for
// Particle-Based Fluids".
//
// Generation: every fluid particle has a potential, computed by the solver's
// force pass from data it already streams (solver.foamGen, particles/s):
//   trapped air  I_ta = Σ_j |v_ij| (1 − v̂_ij·x̂_ij) (1 − r_ij/h)   neighbors
//                converging on it (plunging jets, hydraulic jumps, wakes)
//   wave crest   I_wc = s_i · max(v_i·n̂_i, 0)   a free-surface particle
//                (s_i: surface-ness from the colour-field gradient) moving
//                outward along the surface normal (breaking crests, splashes)
//   energy       E_k = ½|v_i|²
//   rate         Φ(E_k)·(k_ta Φ(I_ta) + k_wc Φ(I_wc)),  Φ = clamped linear ramp
// Each step the expected number of new diffuse particles per fluid particle
// is drawn (stochastic rounding) in a small cylinder around it along its
// velocity.
//
// Dynamics, by the number of fluid neighbors n within h of a diffuse particle:
//   spray   n < ¼ full  ballistic under gravity with light air drag
//   foam    otherwise   carried by the local fluid velocity; ages and dies
//   bubble  n > ¾ full  buoyant, dragged toward the fluid velocity
// Diffuse particles never act on the fluid. Spray that flies into a solid and
// anything out of bounds dies; spray and bubbles age slowly so nothing lives
// forever.

import { colliderNear, colliderSDF } from './colliders.js';
import { H, U, hashCell } from './fluid-core.js';

export const DIFFUSE_STRIDE = 8; // x y z · vx vy vz · life · type
export const SPRAY = 0, FOAM = 1, BUBBLE = 2;

// Φ(I, lo, hi): 0 below lo, 1 above hi, linear between
const ramp = (x, lo, hi) => (x <= lo ? 0 : x >= hi ? 1 : (x - lo) / (hi - lo));

/** Generation rate (diffuse particles / s) from the three potentials. */
export function generationRate(dp, trappedAir, crest, speed2) {
  const ek = ramp(0.5 * speed2, dp.wwEnergy[0], dp.wwEnergy[1]);
  if (ek === 0) return 0;
  return ek * (dp.wwTrappedAir * ramp(trappedAir, dp.wwTrappedAirRange[0], dp.wwTrappedAirRange[1]) +
    dp.wwWaveCrest * ramp(crest, dp.wwCrestRange[0], dp.wwCrestRange[1]));
}

// small deterministic RNG (per step seed) for emission
function rng(seed) {
  let s = seed >>> 0 || 0x9e3779b9;
  return () => {
    s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

/**
 * Serial (coordinator): append new diffuse particles from solver.foamGen,
 * then drop dead ones. Returns the live count.
 */
export function emitWhitewater(solver, dt) {
  const dp = solver.p, hd = solver.header;
  const D = solver.D, d = solver.diffuse, gen = solver.foamGen;
  const p = solver.pos, v = solver.vel, n = hd[H.count];
  // compact: dead particles (life ≤ 0) swap-removed
  let m = hd[H.diffuse];
  for (let k = m - 1; k >= 0; k--) {
    if (d[k * DIFFUSE_STRIDE + 6] > 0) continue;
    m--;
    if (k !== m) d.copyWithin(k * DIFFUSE_STRIDE, m * DIFFUSE_STRIDE, (m + 1) * DIFFUSE_STRIDE);
  }
  const rand = rng(Math.imul(hd[H.frame] + 1, 2654435761));
  const rad = 0.5 * dp.spacing, [l0, l1] = dp.wwLifetime;
  for (let i = 0; i < n && m < D; i++) {
    const g = gen[i];
    if (!(g > 0)) continue;
    let k = Math.floor(g * dt + rand());
    if (k <= 0) continue;
    const i3 = i * 3;
    const vx = v[i3], vy = v[i3 + 1], vz = v[i3 + 2];
    const sp = Math.sqrt(vx * vx + vy * vy + vz * vz);
    // orthonormal frame around the velocity
    let ax = 0, ay = 1, az = 0;
    if (sp > 1e-6) { ax = vx / sp; ay = vy / sp; az = vz / sp; }
    let e1x = -ay, e1y = ax, e1z = 0;
    if (Math.abs(az) > 0.9) { e1x = 0; e1y = -az; e1z = ay; }
    const l = Math.hypot(e1x, e1y, e1z) || 1;
    e1x /= l; e1y /= l; e1z /= l;
    const e2x = ay * e1z - az * e1y, e2y = az * e1x - ax * e1z, e2z = ax * e1y - ay * e1x;
    for (; k > 0 && m < D; k--) {
      const r = rad * Math.sqrt(rand()), th = rand() * Math.PI * 2, hv = rand() * sp * dt;
      const c = Math.cos(th) * r, s = Math.sin(th) * r;
      const o = m * DIFFUSE_STRIDE;
      d[o] = p[i3] + c * e1x + s * e2x + hv * ax;
      d[o + 1] = p[i3 + 1] + c * e1y + s * e2y + hv * ay;
      d[o + 2] = p[i3 + 2] + c * e1z + s * e2z + hv * az;
      d[o + 3] = vx + (c * e1x + s * e2x) * 2;
      d[o + 4] = vy + (c * e1y + s * e2y) * 2;
      d[o + 5] = vz + (c * e1z + s * e2z) * 2;
      d[o + 6] = l0 + (l1 - l0) * rand();
      d[o + 7] = SPRAY;
      m++;
    }
  }
  hd[H.diffuse] = m;
  return m;
}

/**
 * Parallel over diffuse particles [i0, i1): classify by fluid neighbor
 * count on the solver's hash grid, advect, age, kill.
 */
export function advectWhitewater(solver, i0, i1, dt) {
  const dp = solver.p, u = solver.u;
  const d = solver.diffuse, p = solver.pos, v = solver.vel, start = solver.bucketStart;
  const h = dp.h, h2 = h * h, inv = 1 / h, mask = solver.header[H.tableMask];
  const sig = dp.kernelSigma;
  const gx = u[U.gx], gy = u[U.gy], gz = u[U.gz];
  const sprayMax = dp.wwSprayNeighbors, bubbleMin = dp.wwBubbleNeighbors;
  const kb = dp.wwBuoyancy, kd = dp.wwDrag, airDrag = Math.exp(-dp.wwAirDrag * dt);
  const rec = solver.colliders, hf = solver.heightfields, nc = solver.header[H.colliders], n = solver._n;
  const bOn = u[U.boundsOn] > 0;
  const x0 = u[U.bminX], y0 = u[U.bminY], z0 = u[U.bminZ], x1 = u[U.bmaxX], y1 = u[U.bmaxY], z1 = u[U.bmaxZ];
  for (let k = i0; k < i1; k++) {
    const o = k * DIFFUSE_STRIDE;
    let life = d[o + 6];
    if (life <= 0) continue;
    let x = d[o], y = d[o + 1], z = d[o + 2];
    let vx = d[o + 3], vy = d[o + 4], vz = d[o + 5];
    // fluid neighbors: count and kernel-weighted mean velocity
    const cx = Math.floor(x * inv), cy = Math.floor(y * inv), cz = Math.floor(z * inv);
    let cnt = 0, wsum = 0, fx = 0, fy = 0, fz = 0;
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
      const bk = hashCell(cx + a, cy + b, cz + c) & mask, be = start[bk + 1];
      for (let j = start[bk]; j < be; j++) {
        const j3 = j * 3;
        const dx = x - p[j3], dy = y - p[j3 + 1], dz = z - p[j3 + 2];
        const r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= h2) continue;
        cnt++;
        const q = Math.sqrt(r2) * inv;
        const w = q <= 0.5 ? sig * (6 * q * q * q - 6 * q * q + 1) : 2 * sig * (1 - q) * (1 - q) * (1 - q);
        wsum += w; fx += w * v[j3]; fy += w * v[j3 + 1]; fz += w * v[j3 + 2];
      }
    }
    if (wsum > 0) { fx /= wsum; fy /= wsum; fz /= wsum; }
    let type;
    if (cnt < sprayMax) {
      type = 0; // spray: ballistic
      vx = (vx + gx * dt) * airDrag; vy = (vy + gy * dt) * airDrag; vz = (vz + gz * dt) * airDrag;
      life -= 0.25 * dt;
    } else if (cnt > bubbleMin) {
      type = 2; // bubble: buoyant, dragged toward the fluid
      vx += -kb * gx * dt + kd * (fx - vx); vy += -kb * gy * dt + kd * (fy - vy); vz += -kb * gz * dt + kd * (fz - vz);
      life -= 0.25 * dt;
    } else {
      type = 1; // foam: rides the surface
      vx = fx; vy = fy; vz = fz;
      life -= dt;
    }
    x += vx * dt; y += vy * dt; z += vz * dt;
    if (bOn && (x < x0 || x > x1 || y < y0 || y > y1 || z < z0 || z > z1)) life = 0;
    else if (type === 0) {
      // only ballistic spray can fly into a solid (foam moves with the fluid,
      // which the boundaries already keep out; bubbles are under water)
      for (let c = 0; c < nc; c++) {
        if (!colliderNear(rec, c, x, y, z, 0)) continue;
        if (colliderSDF(rec, c, hf, x, y, z, n) < 0) { life = 0; break; }
      }
    }
    d[o] = x; d[o + 1] = y; d[o + 2] = z;
    d[o + 3] = vx; d[o + 4] = vy; d[o + 5] = vz;
    d[o + 6] = life; d[o + 7] = type;
  }
}

/** Pack [x, y, z, type + alpha] for rendering (alpha from remaining life). */
export function packWhitewater(d, m, out, fadeTime = 1) {
  for (let k = 0; k < m; k++) {
    const o = k * DIFFUSE_STRIDE, q = k * 4;
    out[q] = d[o]; out[q + 1] = d[o + 1]; out[q + 2] = d[o + 2];
    const a = Math.min(1, Math.max(0, d[o + 6] / fadeTime));
    out[q + 3] = d[o + 7] + 0.999 * a;
  }
  return m;
}
