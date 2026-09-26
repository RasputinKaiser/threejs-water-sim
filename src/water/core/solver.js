// water/core/solver.js — Position Based Fluids (Macklin & Müller 2013) on a
// spatial hash grid.
//
// One step (dt):
//   predict    v += g·dt, x* = x + v·dt, hash each particle's grid cell
//   sort       counting sort by hash bucket; particle arrays are reordered
//              so every bucket is a contiguous run (cache-friendly neighbor
//              loops). `id` / `framePrev` travel with the particles.
//   neighbors  full neighbor lists within h (27 cells, bucket-deduplicated)
//              + per-particle candidate colliders
//   solve ×N   λ_i = −C_i / (Σ|∇C|² + ε) with C_i = max(ρ_i/ρ0 − 1, 0),
//              Δx_i = Σ_j (λ_i + λ_j) ∇W_ij / ρ0, wall density from collider
//              SDFs, Δx scaled by the Jacobi relaxation factor ω, then
//              collision projection
//   friction   contact friction against touching colliders
//   velocity   v = (x − x_prev)/dt, speed clamp
//   vorticity  XSPH viscosity, vorticity confinement, optional cohesion
//   finalize   remove out-of-bounds / non-finite particles, stats
//
// Every per-particle phase is written in GATHER form (particle i writes only
// its own outputs), so any range split over threads gives bit-identical
// results. Phases take (i0, i1, tid); serial phases run on one thread.

import { wallFraction, wallFractionSlope } from './params.js';
import { COLLIDER_STRIDE, F, FLAG_DYNAMIC, colliderSDF, colliderVelocity } from './colliders.js';
import {
  FluidCore, H, U, PHASE as CORE_PHASE, TIMING, MAX_COLLIDER_CANDIDATES, addImpulse, hashCell,
} from './fluid-core.js';

export { H, U, allocateBuffers, hashCell, MAX_COLLIDER_CANDIDATES } from './fluid-core.js';

// Start-of-step overlap (fraction of the contact radius) resolved by the
// ordinary contact response; deeper overlaps are removed without velocity.
const DEPEN_SLOP = 0.25;

// Parallel phases (dispatch ids shared with worker threads).
export const PHASE = {
  predict: 1, ...CORE_PHASE, lambda: 5, delta: 6,
  apply: 7, frictionVelocity: 8, vorticity1: 9, vorticity2: 10,
};

export class PBFSolver extends FluidCore {
  static PHASES = PHASE;

  constructor(params = {}, buffers = null, opts = {}) {
    super(params.rho0 ? params : { ...params, solver: 'pbf' }, buffers, opts);
    const dp = this.p;
    // cohesion kernel normalization on the rest lattice
    this._cohNorm = 0;
    const s = dp.spacing, h = dp.h, R = Math.ceil(dp.kernelScale) + 1;
    for (let x = -R; x <= R; x++) for (let y = -R; y <= R; y++) for (let z = -R; z <= R; z++) {
      const r = s * Math.hypot(x, y, z);
      if (r > 0 && r < h) this._cohNorm += cohesionShape(r / h);
    }
    this._cohNorm = 1 / this._cohNorm;
  }

  // 1. gravity + prediction + hash key
  phasePredict(i0, i1) {
    const u = this.u, dt = u[U.dt];
    const gx = u[U.gx] * dt, gy = u[U.gy] * dt, gz = u[U.gz] * dt;
    const p = this.pos, pr = this.prev, v = this.vel, key = this.key;
    const inv = 1 / this.p.h, mask = this.header[H.tableMask];
    for (let i = i0; i < i1; i++) {
      const i3 = i * 3;
      const vx = v[i3] + gx, vy = v[i3 + 1] + gy, vz = v[i3 + 2] + gz;
      v[i3] = vx; v[i3 + 1] = vy; v[i3 + 2] = vz;
      const x = p[i3], y = p[i3 + 1], z = p[i3 + 2];
      pr[i3] = x; pr[i3 + 1] = y; pr[i3 + 2] = z;
      const nx = x + vx * dt, ny = y + vy * dt, nz = z + vz * dt;
      p[i3] = nx; p[i3 + 1] = ny; p[i3 + 2] = nz;
      key[i] = hashCell(Math.floor(nx * inv), Math.floor(ny * inv), Math.floor(nz * inv)) & mask;
    }
  }

  // 4a. density + λ
  phaseLambda(i0, i1) {
    const dp = this.p;
    const p = this.pos, nbr = this.nbr, cnt = this.nbrCount, M = this.M;
    const h = dp.h, h2 = dp.h2, K6 = dp.K6, KS = dp.KS, invRho0 = dp.invRho0, rho0 = dp.rho0;
    const W0 = dp.W0, eps = dp.epsilon;
    const wall = this.u[U.wallDensity] > 0;
    const rec = this.colliders, hf = this.heightfields, cand = this.cand, n = this._n;
    const K = MAX_COLLIDER_CANDIDATES;
    const dens = this.density, lam = this.lambda;
    for (let i = i0; i < i1; i++) {
      const i3 = i * 3;
      const xi = p[i3], yi = p[i3 + 1], zi = p[i3 + 2];
      let rho = W0, gx = 0, gy = 0, gz = 0, sum2 = 0;
      const base = i * M, e = base + cnt[i];
      for (let t = base; t < e; t++) {
        const j3 = nbr[t] * 3;
        const dx = xi - p[j3], dy = yi - p[j3 + 1], dz = zi - p[j3 + 2];
        const r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= h2) continue;
        const q = h2 - r2;
        rho += K6 * q * q * q;
        if (r2 > 1e-12) {
          const r = Math.sqrt(r2), hr = h - r;
          const g = -KS * hr * hr / r * invRho0; // ∇W/ρ0 = g · (x_i − x_j)
          const ax = g * dx, ay = g * dy, az = g * dz;
          gx += ax; gy += ay; gz += az;
          sum2 += ax * ax + ay * ay + az * az;
        }
      }
      if (wall) {
        const cb = i * K;
        for (let k = 0; k < K; k++) {
          const c = cand[cb + k];
          if (c < 0) break;
          let d = colliderSDF(rec, c, hf, xi, yi, zi, n);
          if (d >= h) continue;
          // a particle inside a solid is the collision phase's to resolve:
          // counted as if on the surface, it cannot blow up the pressure
          if (d < 0) d = 0;
          rho += rho0 * wallFraction(d, h);
          const s = wallFractionSlope(d, h); // ∂(ρ/ρ0)/∂d
          gx += s * n[0]; gy += s * n[1]; gz += s * n[2];
        }
      }
      dens[i] = rho;
      let C = rho * invRho0 - 1;
      if (C < 0) C = 0; // unilateral: free-surface particles are not pulled in
      lam[i] = -C / (gx * gx + gy * gy + gz * gz + sum2 + eps);
    }
  }

  // 4b. position correction Δx into tmpA (+ wall-pressure impulses)
  phaseDelta(i0, i1, tid = 0) {
    const dp = this.p;
    const p = this.pos, nbr = this.nbr, cnt = this.nbrCount, M = this.M;
    const h = dp.h, h2 = dp.h2, KS = dp.KS, invRho0 = dp.invRho0;
    const wall = this.u[U.wallDensity] > 0;
    const rec = this.colliders, hf = this.heightfields, cand = this.cand, n = this._n;
    const K = MAX_COLLIDER_CANDIDATES;
    const lam = this.lambda, out = this.tmpA;
    const imp = this.impulses, impBase = tid * this.maxColliders * 6;
    // the wall push reaches the particle scaled by ω in phaseApply, so its
    // reaction on the collider must be scaled the same (momentum conservation)
    const mOverDt = dp.particleMass / this.u[U.dt] * this.u[U.omega];
    for (let i = i0; i < i1; i++) {
      const i3 = i * 3;
      const xi = p[i3], yi = p[i3 + 1], zi = p[i3 + 2];
      const li = lam[i];
      let sx = 0, sy = 0, sz = 0;
      const base = i * M, e = base + cnt[i];
      for (let t = base; t < e; t++) {
        const j = nbr[t], j3 = j * 3;
        const dx = xi - p[j3], dy = yi - p[j3 + 1], dz = zi - p[j3 + 2];
        const r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= h2 || r2 <= 1e-12) continue;
        const r = Math.sqrt(r2), hr = h - r;
        const g = -KS * hr * hr / r * (li + lam[j]);
        sx += g * dx; sy += g * dy; sz += g * dz;
      }
      sx *= invRho0; sy *= invRho0; sz *= invRho0;
      if (wall && li < 0) {
        const cb = i * K;
        for (let k = 0; k < K; k++) {
          const c = cand[cb + k];
          if (c < 0) break;
          let d = colliderSDF(rec, c, hf, xi, yi, zi, n);
          if (d >= h) continue;
          if (d < 0) d = 0; // as in phaseLambda
          const o = c * COLLIDER_STRIDE;
          const s = li * wallFractionSlope(d, h) * rec[o + F.alpha]; // λ_i ∂C/∂d along n
          const wx = s * n[0], wy = s * n[1], wz = s * n[2];
          sx += wx; sy += wy; sz += wz;
          if (rec[o + F.flags] & FLAG_DYNAMIC) {
            addImpulse(imp, impBase + rec[o + F.slot] * 6, rec, o, xi, yi, zi,
              -wx * mOverDt, -wy * mOverDt, -wz * mOverDt);
          }
        }
      }
      out[i3] = sx; out[i3 + 1] = sy; out[i3 + 2] = sz;
    }
  }

  // 4c. apply Δx (× SOR factor ω), then collision projection. A particle
  // that entered a solid during this step leaves through the surface it
  // crossed (found on the prev→x segment), not the nearest face: the nearest
  // face of a box resting on the floor is its bottom, which would push the
  // particle into the floor and trap it there.
  phaseApply(i0, i1, tid = 0) {
    const dp = this.p;
    const p = this.pos, dx = this.tmpA, pr = this.prev, m = this._cv;
    const radStatic = dp.particleRadius, radDynamic = dp.dynamicRadius;
    const rec = this.colliders, hf = this.heightfields, cand = this.cand, n = this._n;
    const K = MAX_COLLIDER_CANDIDATES;
    const imp = this.impulses, impBase = tid * this.maxColliders * 6;
    const dt = this.u[U.dt], mOverDt = dp.particleMass / dt;
    const w = this.u[U.omega];
    for (let i = i0; i < i1; i++) {
      const i3 = i * 3;
      let x = p[i3] + w * dx[i3], y = p[i3 + 1] + w * dx[i3 + 1], z = p[i3 + 2] + w * dx[i3 + 2];
      const cb = i * K;
      for (let k = 0; k < K; k++) {
        const c = cand[cb + k];
        if (c < 0) break;
        let d = colliderSDF(rec, c, hf, x, y, z, n);
        const o = c * COLLIDER_STRIDE;
        const rad = rec[o + F.flags] & FLAG_DYNAMIC ? radDynamic : radStatic;
        if (d >= rad) continue;
        let sx = x, sy = y, sz = z; // surface-side reference point
        if (d < 0) {
          const d0 = colliderSDF(rec, c, hf, pr[i3], pr[i3 + 1], pr[i3 + 2], m);
          if (d0 > 0) {
            const t = d0 / (d0 - d); // linear estimate of the crossing
            sx = pr[i3] + (x - pr[i3]) * t; sy = pr[i3 + 1] + (y - pr[i3 + 1]) * t; sz = pr[i3 + 2] + (z - pr[i3 + 2]) * t;
            d = colliderSDF(rec, c, hf, sx, sy, sz, n);
          }
        }
        // target: the contact skin rad outside the surface. With a contact
        // share α < 1 (light dynamic body) only α of the skin depth is
        // enforced, but a particle is always returned to the surface itself —
        // softening the momentum exchange must never let fluid seep inside.
        const a = rec[o + F.alpha];
        const target = (d < 0 ? 0 : d) + a * (rad - (d < 0 ? 0 : d));
        const pen = target - d;
        const mx = sx + pen * n[0] - x, my = sy + pen * n[1] - y, mz = sz + pen * n[2] - z;
        x += mx; y += my; z += mz;
        // Depenetration is not a collision: the part of the push that only
        // undoes an overlap the particle already had at the start of the step
        // (spawned into a solid or its skin, a body created or teleported
        // into water) moves the step origin along, so it adds no velocity and
        // no reaction impulse. The start overlap is measured along this
        // contact's normal relative to the collider surface, which moved by
        // v·dt during the step — so a paddle sweeping into water still pushes
        // it. Like Box2D's linear slop, overlaps under `slop` keep the
        // ordinary response: where walls meet, particles end each step
        // slightly inside one wall's skin, and that response keeps them at rest.
        let sh = 0;
        if (pen > 0) {
          colliderVelocity(rec, c, x, y, z, m);
          const below = (x - pr[i3] - m[0] * dt) * n[0] + (y - pr[i3 + 1] - m[1] * dt) * n[1] +
            (z - pr[i3 + 2] - m[2] * dt) * n[2] - DEPEN_SLOP * rad;
          sh = below < pen ? below : pen;
          if (sh > 0) {
            pr[i3] += sh * n[0]; pr[i3 + 1] += sh * n[1]; pr[i3 + 2] += sh * n[2];
          } else sh = 0;
        }
        if (rec[o + F.flags] & FLAG_DYNAMIC) {
          addImpulse(imp, impBase + rec[o + F.slot] * 6, rec, o, x, y, z,
            -(mx - sh * n[0]) * mOverDt, -(my - sh * n[1]) * mOverDt, -(mz - sh * n[2]) * mOverDt);
        }
      }
      p[i3] = x; p[i3 + 1] = y; p[i3 + 2] = z;
    }
  }

  // 4d. contact friction: damp slip relative to each touching collider's
  // surface over the step (once per step, after the constraint iterations)
  phaseFriction(i0, i1, tid = 0) {
    const dp = this.p, u = this.u;
    const p = this.pos, pr = this.prev;
    const reach = dp.particleRadius + 0.1 * dp.spacing, dt = u[U.dt];
    const rec = this.colliders, hf = this.heightfields, cand = this.cand, n = this._n, cv = this._cv;
    const K = MAX_COLLIDER_CANDIDATES;
    const defFric = u[U.friction];
    const imp = this.impulses, impBase = tid * this.maxColliders * 6;
    const mOverDt = dp.particleMass / dt;
    for (let i = i0; i < i1; i++) {
      const cb = i * K;
      if (cand[cb] < 0) continue;
      const i3 = i * 3;
      let x = p[i3], y = p[i3 + 1], z = p[i3 + 2];
      for (let k = 0; k < K; k++) {
        const c = cand[cb + k];
        if (c < 0) break;
        const o = c * COLLIDER_STRIDE;
        const f = rec[o + F.friction] >= 0 ? rec[o + F.friction] : defFric;
        if (f <= 0) continue;
        const d = colliderSDF(rec, c, hf, x, y, z, n);
        if (d >= reach) continue;
        colliderVelocity(rec, c, x, y, z, cv);
        const rx = x - pr[i3] - cv[0] * dt, ry = y - pr[i3 + 1] - cv[1] * dt, rz = z - pr[i3 + 2] - cv[2] * dt;
        const rn = rx * n[0] + ry * n[1] + rz * n[2];
        const fa = f * rec[o + F.alpha];
        const mx = -fa * (rx - rn * n[0]), my = -fa * (ry - rn * n[1]), mz = -fa * (rz - rn * n[2]);
        x += mx; y += my; z += mz;
        if (rec[o + F.flags] & FLAG_DYNAMIC) {
          addImpulse(imp, impBase + rec[o + F.slot] * 6, rec, o, x, y, z,
            -mx * mOverDt, -my * mOverDt, -mz * mOverDt);
        }
      }
      p[i3] = x; p[i3 + 1] = y; p[i3 + 2] = z;
    }
  }

  // 5. velocity from positions, speed clamp
  phaseVelocity(i0, i1) {
    const u = this.u, invDt = 1 / u[U.dt], maxV = u[U.maxSpeed], maxV2 = maxV * maxV;
    const p = this.pos, pr = this.prev, v = this.vel;
    for (let i = i0; i < i1; i++) {
      const i3 = i * 3;
      let vx = (p[i3] - pr[i3]) * invDt, vy = (p[i3 + 1] - pr[i3 + 1]) * invDt, vz = (p[i3 + 2] - pr[i3 + 2]) * invDt;
      const s2 = vx * vx + vy * vy + vz * vz;
      if (s2 > maxV2) {
        const s = maxV / Math.sqrt(s2);
        vx *= s; vy *= s; vz *= s;
        p[i3] = pr[i3] + vx / invDt; p[i3 + 1] = pr[i3 + 1] + vy / invDt; p[i3 + 2] = pr[i3 + 2] + vz / invDt;
      }
      v[i3] = vx; v[i3 + 1] = vy; v[i3 + 2] = vz;
    }
  }

  // 6a. vorticity ω, XSPH and cohesion velocity changes (into tmpB)
  phaseVorticity1(i0, i1) {
    const dp = this.p, u = this.u;
    const p = this.pos, v = this.vel, nbr = this.nbr, cnt = this.nbrCount, M = this.M;
    const h = dp.h, h2 = dp.h2, K6 = dp.K6, KS = dp.KS;
    const dens = this.density, om = this.omega, out = this.tmpB;
    const xsph = u[U.viscosity];
    const coh = u[U.cohesion] * Math.hypot(u[U.gx], u[U.gy], u[U.gz]) * u[U.dt] * this._cohNorm;
    for (let i = i0; i < i1; i++) {
      const i3 = i * 3;
      const xi = p[i3], yi = p[i3 + 1], zi = p[i3 + 2];
      const vxi = v[i3], vyi = v[i3 + 1], vzi = v[i3 + 2];
      let wx = 0, wy = 0, wz = 0, ax = 0, ay = 0, az = 0, cx = 0, cy = 0, cz = 0;
      const base = i * M, e = base + cnt[i];
      for (let t = base; t < e; t++) {
        const j = nbr[t], j3 = j * 3;
        const dx = xi - p[j3], dy = yi - p[j3 + 1], dz = zi - p[j3 + 2];
        const r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= h2 || r2 <= 1e-12) continue;
        const invRhoJ = 1 / dens[j];
        const ux = v[j3] - vxi, uy = v[j3 + 1] - vyi, uz = v[j3 + 2] - vzi;
        const q = h2 - r2;
        const w = K6 * q * q * q * invRhoJ;
        ax += ux * w; ay += uy * w; az += uz * w;
        const r = Math.sqrt(r2), hr = h - r;
        const g = -KS * hr * hr / r * invRhoJ; // ∇W_ij/ρ_j = g·(x_i − x_j)
        const gx = g * dx, gy = g * dy, gz = g * dz;
        wx += uy * gz - uz * gy; wy += uz * gx - ux * gz; wz += ux * gy - uy * gx;
        if (coh !== 0) {
          const c = cohesionShape(r / h) / r;
          cx -= c * dx; cy -= c * dy; cz -= c * dz;
        }
      }
      om[i3] = wx; om[i3 + 1] = wy; om[i3 + 2] = wz;
      out[i3] = xsph * ax + coh * cx;
      out[i3 + 1] = xsph * ay + coh * cy;
      out[i3 + 2] = xsph * az + coh * cz;
    }
  }

  // 6b. vorticity confinement force; v += XSPH + confinement (in place: this
  // pass reads only ω and positions of neighbors, never their velocities)
  phaseVorticity2(i0, i1) {
    const dp = this.p;
    const p = this.pos, v = this.vel, nbr = this.nbr, cnt = this.nbrCount, M = this.M;
    const h = dp.h, h2 = dp.h2, KS = dp.KS;
    const dens = this.density, om = this.omega, dv = this.tmpB;
    const eps = this.u[U.vorticity] * h;
    for (let i = i0; i < i1; i++) {
      const i3 = i * 3;
      let vx = v[i3] + dv[i3], vy = v[i3 + 1] + dv[i3 + 1], vz = v[i3 + 2] + dv[i3 + 2];
      if (eps > 0) {
        const wxi = om[i3], wyi = om[i3 + 1], wzi = om[i3 + 2];
        const wi = Math.sqrt(wxi * wxi + wyi * wyi + wzi * wzi);
        if (wi > 1e-6) {
          const xi = p[i3], yi = p[i3 + 1], zi = p[i3 + 2];
          let ex = 0, ey = 0, ez = 0;
          const base = i * M, e = base + cnt[i];
          for (let t = base; t < e; t++) {
            const j = nbr[t], j3 = j * 3;
            const dx = xi - p[j3], dy = yi - p[j3 + 1], dz = zi - p[j3 + 2];
            const r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= h2 || r2 <= 1e-12) continue;
            const wj = Math.sqrt(om[j3] * om[j3] + om[j3 + 1] * om[j3 + 1] + om[j3 + 2] * om[j3 + 2]);
            const r = Math.sqrt(r2), hr = h - r;
            const g = -KS * hr * hr / r * (wj - wi) / dens[j];
            ex += g * dx; ey += g * dy; ez += g * dz;
          }
          const el = Math.sqrt(ex * ex + ey * ey + ez * ez);
          if (el > 1e-9) {
            const nx = ex / el, ny = ey / el, nz = ez / el;
            vx += eps * (ny * wzi - nz * wyi);
            vy += eps * (nz * wxi - nx * wzi);
            vz += eps * (nx * wyi - ny * wxi);
          }
        }
      }
      v[i3] = vx; v[i3 + 1] = vy; v[i3 + 2] = vz;
    }
  }

  /* ================= serial finalize ================= */

  phaseFinalize() {
    this.finalizeParticles();
    let maxErr = 0;
    const n = this.header[H.count], invRho0 = this.p.invRho0, dens = this.density;
    for (let i = 0; i < n; i++) { const e = dens[i] * invRho0 - 1; if (e > maxErr) maxErr = e; }
    this.u[U.maxDensityError] = maxErr;
  }


  /* ================= driver ================= */

  /** Run one parallel phase over particles [i0, i1) as thread `tid`. */
  runPhase(id, i0, i1, tid = 0) {
    switch (id) {
      case PHASE.predict: return this.phasePredict(i0, i1);
      case PHASE.gather: return this.phaseGather(i0, i1);
      case PHASE.scatterBack: return this.phaseScatterBack(i0, i1);
      case PHASE.neighbors: return this.phaseNeighbors(i0, i1, tid);
      case PHASE.lambda: return this.phaseLambda(i0, i1);
      case PHASE.delta: return this.phaseDelta(i0, i1, tid);
      case PHASE.apply: return this.phaseApply(i0, i1, tid);
      case PHASE.frictionVelocity:
        this.phaseFriction(i0, i1, tid);
        return this.phaseVelocity(i0, i1);
      case PHASE.vorticity1: return this.phaseVorticity1(i0, i1);
      case PHASE.vorticity2: return this.phaseVorticity2(i0, i1);
      default: throw new Error(`unknown phase ${id}`);
    }
  }

  /**
   * Advance one step. `parallel(id)` runs a phase over all particles — by
   * default on this thread; the thread pool passes a function that splits
   * the range across workers and waits at a barrier.
   */
  step(dt, parallel = null) {
    const T = this.phaseMs;
    T.fill(0);
    const t0 = performance.now();
    this.u[U.dt] = dt;
    this.header[H.overflow] = 0;
    const exec = parallel ?? ((id) => this.runPhase(id, 0, this.header[H.count], 0));
    const run = (id) => { const t = performance.now(); exec(id); T[id] += performance.now() - t; };
    this._advanceColliders(dt);
    if (this.header[H.count] > 0) {
      run(PHASE.predict);
      let t = performance.now();
      this.phaseSort();
      T[TIMING.sort] = performance.now() - t;
      run(PHASE.gather);
      run(PHASE.scatterBack);
      run(PHASE.neighbors);
      this._contactShares();
      const iters = this.u[U.iterations];
      for (let it = 0; it < iters; it++) {
        run(PHASE.lambda);
        run(PHASE.delta);
        run(PHASE.apply);
      }
      run(PHASE.frictionVelocity);
      run(PHASE.vorticity1);
      run(PHASE.vorticity2);
    }
    const tf = performance.now();
    this.phaseFinalize();
    const t1 = performance.now();
    T[TIMING.finalize] = t1 - tf;
    T[TIMING.total] = t1 - t0;
  }
}

// Cohesion profile: 0 at r=0 and r=h, peak near r≈0.6h (Akinci-like shape).
function cohesionShape(u) {
  if (u >= 1 || u <= 0) return 0;
  const a = (1 - u) * u;
  return a * a * a;
}

