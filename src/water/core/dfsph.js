// water/core/dfsph.js — Divergence-Free SPH (Bender & Koschier 2015, 2017) on
// the spatial hash grid, with SDF volume-map boundaries (Bender et al. 2019).
//
// One substep (dt from the CFL condition):
//   hash/sort  particles counting-sorted by grid-cell hash (fluid-core.js)
//   neighbors  neighbor lists within h + candidate colliders
//   density    ρ_i = Σ W_ij + ρ0 Ψ(d_i); every pair's ∇W_ij and W_ij are
//              cached, with each candidate collider's ρ0∇Ψ and velocity, and
//              β_i = 1 / (|Σ∇W_ij + ρ0∇Ψ|² + Σ|∇W_ij|²)
//   divergence iterate k_i = max(Dρ_i/Dt, 0)·β_i/dt,
//              v_i −= dt·[Σ (k_i + k_j)∇W_ij + k_i ρ0∇Ψ] until the mean
//              compression rate is below tolerance (makes the velocity
//              field divergence-free: no jitter, no bounce)
//   forces     gravity, XSPH viscosity, wall friction, vorticity confinement
//   pressure   same iteration with k_i = max(ρ*_i − ρ0, 0)·β_i/dt², where
//              ρ*_i = ρ_i + dt·Dρ_i/Dt is the density the velocities lead to
//              (unilateral: free-surface particles are never pulled in)
//   integrate  x += dt·v, then a safety projection out of solids
//
// Positions are fixed during both solves, so the kernel is evaluated once per
// substep and every solver iteration only streams the cached gradients.
// Units: particle mass 1, densities in Σ W (ρ0 = the lattice sum, 1/spacing³),
// velocities in m/s; the physical mass (waterDensity · spacing³) is used only
// for impulses exchanged with rigid bodies.
//
// Boundary pressure acts on dynamic bodies as the reaction of every velocity
// change it causes — buoyancy, drag and splashes come out of that.

import { COLLIDER_STRIDE, F, FLAG_DYNAMIC, SHAPE, colliderSDF, colliderVelocity, containerWalls } from './colliders.js';
import {
  FluidCore, H, U, PHASE as CORE_PHASE, TIMING, MAX_COLLIDER_CANDIDATES as K, BND_STRIDE, addImpulse,
} from './fluid-core.js';

export const PHASE = {
  ...CORE_PHASE, hash: 20, density: 21, divergenceResidual: 22, velocity: 23,
  forces: 24, confine: 25, densityResidual: 26, integrate: 27,
  warmDensity: 28, warmDivergence: 29, densityGate: 30, divergenceGate: 31,
};

// carried per-particle state (fluid-core carry slots): the total k each
// solve applied last step — the warm start of the next (Bender & Koschier:
// Jacobi-type pressure solves otherwise spend one iteration per particle
// layer rebuilding hydrostatic pressure from zero every step)
const C_DENSITY = 0, C_DIVERGENCE = 1;
// Warm start (params.warmStart, default ½), only where the particle is currently
// predicted to compress: increments are ≥ 0, so a full warm start ratchets up
// without bound (measured: a resting column blew up); with ½ the carried and
// freshly solved parts settle at equal shares. A projected solve on the total
// pressure, which could lower it again, needed Jacobi relaxation ω ≈ ½ to
// converge and ended up noisier.

// reduction slots (per thread)
const R_ERR = 0, R_MAX = 1;

export class DFSPHSolver extends FluidCore {
  static PHASES = PHASE;

  constructor(params = {}, buffers = null, opts = {}) {
    super(params.rho0 ? params : { ...params, solver: 'dfsph' }, buffers, opts);
    const b = this.buffers;
    this.pair = new Float32Array(b.pair);
    this.bnd = new Float32Array(b.bnd);
    this.beta = new Float32Array(b.beta);
    this.kappa = new Float32Array(b.kappa);
    this.bndF = this.p.bndF; this.bndDF = this.p.bndDF;
    this._wd = new Float64Array(5);
    this._wn = new Float64Array(15);
    this._ws = new Float64Array(5);
    this._g = new Float64Array(3);
  }

  // Ψ and ∇Ψ of a container as the union of its wall half-spaces:
  // Ψ = 1 − Π(1 − Ψ_w), ∇Ψ = Σ Ψ'_w n_w Π_{v≠w}(1 − Ψ_v). Writes ∇Ψ into
  // g[0..2] and returns Ψ (0 when no wall is within h).
  _containerVolume(c, x, y, z, g) {
    const dp = this.p, h = dp.h, Ft = this.bndF, dFt = this.bndDF, tInv = dp.bndInv;
    const wd = this._wd, wn = this._wn, ws = this._ws;
    containerWalls(this.colliders, c, x, y, z, wd, wn);
    let prod = 1;
    for (let w = 0; w < 5; w++) {
      let d = wd[w];
      if (d >= h) { wd[w] = 0; ws[w] = 0; continue; }
      if (d < 0) d = 0;
      const f = d * tInv, fi = f | 0, ft = f - fi;
      const psi = Ft[fi] + (Ft[fi + 1] - Ft[fi]) * ft;
      wd[w] = psi;
      ws[w] = dFt[fi] + (dFt[fi + 1] - dFt[fi]) * ft;
      prod *= 1 - psi;
    }
    g[0] = 0; g[1] = 0; g[2] = 0;
    if (prod === 1) return 0;
    for (let w = 0; w < 5; w++) {
      if (ws[w] === 0) continue;
      const k = ws[w] * (1 - wd[w] > 1e-9 ? prod / (1 - wd[w]) : 0);
      g[0] += k * wn[w * 3]; g[1] += k * wn[w * 3 + 1]; g[2] += k * wn[w * 3 + 2];
    }
    return 1 - prod;
  }

  /* ================= per-particle phases ================= */

  // density, kernel cache, boundary cache, β
  phaseDensity(i0, i1) {
    const dp = this.p;
    const p = this.pos, v = this.vel, nbr = this.nbr, cnt = this.nbrCount, M = this.M;
    const h = dp.h, invH = 1 / h, sig = dp.kernelSigma, sigH = sig * invH;
    const rho0 = dp.rho0, W0 = dp.W0;
    const pair = this.pair, bnd = this.bnd, dens = this.density, beta = this.beta;
    const rec = this.colliders, hf = this.heightfields, cand = this.cand, n = this._n, cv = this._cv;
    const Ft = this.bndF, dFt = this.bndDF, tInv = dp.bndInv, gv = this._g, id = this.id;
    const gMax = sigH * 2; // |dW/dr| at q = ⅓, the cubic kernel's steepest point
    for (let i = i0; i < i1; i++) {
      const i3 = i * 3;
      const xi = p[i3], yi = p[i3 + 1], zi = p[i3 + 2];
      let rho = W0, gx = 0, gy = 0, gz = 0, sum2 = 0;
      const base = i * M, e = base + cnt[i];
      for (let t = base; t < e; t++) {
        const j3 = nbr[t] * 3;
        const dx = xi - p[j3], dy = yi - p[j3 + 1], dz = zi - p[j3 + 2];
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz), q = r * invH;
        let W, dW;
        if (q <= 0.5) { W = sig * (6 * q * q * q - 6 * q * q + 1); dW = sigH * (18 * q * q - 12 * q); }
        else if (q < 1) { const a = 1 - q; W = 2 * sig * a * a * a; dW = -6 * sigH * a * a; }
        else { W = 0; dW = 0; }
        let ax, ay, az;
        if (r > 1e-6 * h) {
          const g = dW / r;
          ax = g * dx; ay = g * dy; az = g * dz;
        } else {
          // coincident pair (water spawned on water): a deterministic,
          // antisymmetric separation direction from the two ids at the
          // kernel's steepest gradient — otherwise both stay coincident forever
          const a = id[i], b = id[nbr[t]], lo = a < b ? a : b, hi = a < b ? b : a;
          const hsh = Math.imul(lo, 73856093) ^ Math.imul(hi, 19349663);
          const th = (hsh & 1023) * (Math.PI * 2 / 1024), ph = ((hsh >>> 10) & 1023) * (Math.PI / 1024);
          const sg = a < b ? gMax : -gMax;
          ax = sg * Math.sin(ph) * Math.cos(th); ay = sg * Math.cos(ph); az = sg * Math.sin(ph) * Math.sin(th);
        }
        const t4 = t * 4;
        pair[t4] = ax; pair[t4 + 1] = ay; pair[t4 + 2] = az; pair[t4 + 3] = W;
        rho += W;
        gx += ax; gy += ay; gz += az;
        sum2 += ax * ax + ay * ay + az * az;
      }
      // boundaries: Ψ(d) from the lattice table, ∇(ρ0Ψ) = ρ0 Ψ'(d) n
      const cb = i * K;
      for (let k = 0; k < K; k++) {
        const c = cand[cb + k];
        if (c < 0) break;
        const b8 = (cb + k) * BND_STRIDE;
        let psi, bx, by, bz;
        if (rec[c * COLLIDER_STRIDE] === SHAPE.container) {
          psi = this._containerVolume(c, xi, yi, zi, gv);
          bnd[b8 + 7] = 0;
          if (psi === 0) { bnd[b8] = 0; bnd[b8 + 1] = 0; bnd[b8 + 2] = 0; bnd[b8 + 6] = 0; continue; }
          bx = rho0 * gv[0]; by = rho0 * gv[1]; bz = rho0 * gv[2];
        } else {
          let d = colliderSDF(rec, c, hf, xi, yi, zi, n);
          bnd[b8 + 7] = d;
          if (d >= h) { bnd[b8] = 0; bnd[b8 + 1] = 0; bnd[b8 + 2] = 0; bnd[b8 + 6] = 0; continue; }
          if (d < 0) d = 0; // inside: the integrate projection resolves it
          const f = d * tInv, fi = f | 0, ft = f - fi;
          psi = Ft[fi] + (Ft[fi + 1] - Ft[fi]) * ft;
          const s = rho0 * (dFt[fi] + (dFt[fi + 1] - dFt[fi]) * ft);
          bx = s * n[0]; by = s * n[1]; bz = s * n[2];
        }
        rho += rho0 * psi;
        gx += bx; gy += by; gz += bz;
        colliderVelocity(rec, c, xi, yi, zi, cv);
        bnd[b8] = bx; bnd[b8 + 1] = by; bnd[b8 + 2] = bz;
        bnd[b8 + 3] = cv[0]; bnd[b8 + 4] = cv[1]; bnd[b8 + 5] = cv[2];
        bnd[b8 + 6] = psi;
      }
      dens[i] = rho;
      const den = gx * gx + gy * gy + gz * gz + sum2;
      beta[i] = den > 1e-9 ? 1 / den : 0;
      void v;
    }
  }

  // Dρ/Dt from the current velocities → this iteration's pressure increment
  // k_i = max(ρ*_i − ρ0, 0)·β_i/dt² (density) or max(Dρ_i/Dt, 0)·β_i/dt
  // (divergence); increments are ≥ 0 (no tension at the free surface) and
  // are summed into the carried total for the next step's warm start.
  // `accumulate` false: only classify (warm-start gating), nothing is summed.
  phaseResidual(i0, i1, tid, density, accumulate = true) {
    const dp = this.p, u = this.u, dt = u[U.stepDt];
    const v = this.vel, nbr = this.nbr, cnt = this.nbrCount, M = this.M;
    const pair = this.pair, bnd = this.bnd, dens = this.density, beta = this.beta, kap = this.kappa;
    const cand = this.cand, rho0 = dp.rho0, minN = dp.minNeighbors;
    const kDiv = 1 / dt, kDen = 1 / (dt * dt), maxOver = dp.maxOverdensity * rho0;
    const acc = this.carry[density ? C_DENSITY : C_DIVERGENCE];
    let err = 0, emax = 0;
    for (let i = i0; i < i1; i++) {
      const i3 = i * 3;
      const vx = v[i3], vy = v[i3 + 1], vz = v[i3 + 2];
      let drho = 0;
      const base = i * M, e = base + cnt[i];
      for (let t = base; t < e; t++) {
        const j3 = nbr[t] * 3, t4 = t * 4;
        drho += (vx - v[j3]) * pair[t4] + (vy - v[j3 + 1]) * pair[t4 + 1] + (vz - v[j3 + 2]) * pair[t4 + 2];
      }
      const cb = i * K;
      for (let k = 0; k < K; k++) {
        if (cand[cb + k] < 0) break;
        const b8 = (cb + k) * BND_STRIDE;
        drho += (vx - bnd[b8 + 3]) * bnd[b8] + (vy - bnd[b8 + 4]) * bnd[b8 + 1] + (vz - bnd[b8 + 5]) * bnd[b8 + 2];
      }
      let r, k;
      if (density) {
        // over-density already present (an invalid start: water spawned on
        // top of water, a body created inside it) is removed at most
        // maxOver·ρ0 per substep; compression the velocities cause is always
        // corrected in full
        let e = dens[i] - rho0;
        if (e > maxOver) e = maxOver;
        r = e + dt * drho;
        if (r < 0) r = 0;
        k = r * beta[i] * kDen;
      } else {
        // free-surface particles: too few neighbors for a meaningful Dρ/Dt
        r = cnt[i] < minN || drho < 0 ? 0 : drho;
        k = r * beta[i] * kDiv;
        r *= dt;
      }
      err += r;
      if (r > emax) emax = r;
      kap[i] = k;
      if (accumulate) acc[i] += k;
    }
    const red = this.reduce, o = tid * 8;
    red[o + R_ERR] += err;
    if (emax > red[o + R_MAX]) red[o + R_MAX] = emax;
  }

  // warm start: last step's total × factor for particles the gating residual
  // pass just found compressing (k > 0), 0 elsewhere; the velocity pass applies it
  phaseWarm(i0, i1, slot, factor) {
    const acc = this.carry[slot], kap = this.kappa;
    for (let i = i0; i < i1; i++) {
      const k = kap[i] > 0 ? acc[i] * factor : 0;
      kap[i] = k; acc[i] = k;
    }
  }

  // v_i −= dt·[Σ (k_i + k_j)∇W_ij + k_i ρ0∇Ψ]; boundary reactions → bodies
  phaseVelocity(i0, i1, tid) {
    const dp = this.p, dt = this.u[U.stepDt];
    const v = this.vel, nbr = this.nbr, cnt = this.nbrCount, M = this.M;
    const pair = this.pair, bnd = this.bnd, kap = this.kappa;
    const rec = this.colliders, cand = this.cand, p = this.pos;
    const imp = this.impulses, impBase = tid * this.maxColliders * 6, m = dp.particleMass;
    for (let i = i0; i < i1; i++) {
      const i3 = i * 3;
      const ki = kap[i];
      let ax = 0, ay = 0, az = 0;
      const base = i * M, e = base + cnt[i];
      for (let t = base; t < e; t++) {
        const s = ki + kap[nbr[t]];
        if (s === 0) continue;
        const t4 = t * 4;
        ax += s * pair[t4]; ay += s * pair[t4 + 1]; az += s * pair[t4 + 2];
      }
      let vx = v[i3] - dt * ax, vy = v[i3 + 1] - dt * ay, vz = v[i3 + 2] - dt * az;
      if (ki !== 0) {
        const cb = i * K;
        for (let k = 0; k < K; k++) {
          const c = cand[cb + k];
          if (c < 0) break;
          const b8 = (cb + k) * BND_STRIDE;
          // the constraint force Jᵀk: ∂ρ_i/∂x_i has the boundary term once, so
          // the wall pushes with k_i (not the mirrored 2k_i — that is not a
          // constraint force and does work on density-preserving motion:
          // measured, it pumped a resting slab up to 0.4 m/s)
          const f = -dt * ki;
          const dx = f * bnd[b8], dy = f * bnd[b8 + 1], dz = f * bnd[b8 + 2];
          if (dx === 0 && dy === 0 && dz === 0) continue;
          vx += dx; vy += dy; vz += dz;
          const o = c * COLLIDER_STRIDE;
          if (rec[o + F.flags] & FLAG_DYNAMIC) {
            addImpulse(imp, impBase + rec[o + F.slot] * 6, rec, o, p[i3], p[i3 + 1], p[i3 + 2], -m * dx, -m * dy, -m * dz);
          }
        }
      }
      v[i3] = vx; v[i3 + 1] = vy; v[i3 + 2] = vz;
    }
  }

  // non-pressure accelerations → tmpB (Δv) and vorticity ω
  phaseForces(i0, i1, tid) {
    const dp = this.p, u = this.u, dt = u[U.stepDt];
    const v = this.vel, nbr = this.nbr, cnt = this.nbrCount, M = this.M;
    const pair = this.pair, bnd = this.bnd, dens = this.density, om = this.omega, out = this.tmpB;
    const rec = this.colliders, cand = this.cand, p = this.pos;
    const imp = this.impulses, impBase = tid * this.maxColliders * 6, m = dp.particleMass;
    const gx = u[U.gx] * dt, gy = u[U.gy] * dt, gz = u[U.gz] * dt;
    // rates are specified per 1/60 s and made step-size independent
    const frames = dt * 60;
    const xsph = 1 - Math.pow(1 - Math.min(u[U.viscosity], 0.99), frames);
    const fric = u[U.friction], invS = 1 / dp.spacing;
    const wantOmega = u[U.vorticity] > 0;
    for (let i = i0; i < i1; i++) {
      const i3 = i * 3;
      const vxi = v[i3], vyi = v[i3 + 1], vzi = v[i3 + 2];
      let ax = 0, ay = 0, az = 0, wx = 0, wy = 0, wz = 0;
      const base = i * M, e = base + cnt[i];
      for (let t = base; t < e; t++) {
        const j = nbr[t], j3 = j * 3, t4 = t * 4;
        const invRho = 1 / dens[j];
        const ux = v[j3] - vxi, uy = v[j3 + 1] - vyi, uz = v[j3 + 2] - vzi;
        const w = pair[t4 + 3] * invRho;
        ax += ux * w; ay += uy * w; az += uz * w;
        if (wantOmega) {
          const qx = pair[t4] * invRho, qy = pair[t4 + 1] * invRho, qz = pair[t4 + 2] * invRho;
          wx += uy * qz - uz * qy; wy += uz * qx - ux * qz; wz += ux * qy - uy * qx;
        }
      }
      let dvx = gx + xsph * ax, dvy = gy + xsph * ay, dvz = gz + xsph * az;
      // wall shear: quadratic drag τ = ρ C_f |u_t| u_t on the slip relative to
      // the surface (C_f = `friction`, the bed's drag coefficient: ~0.003
      // smooth, ~0.01 gravel, ~0.03 boulders — Manning n² g / R^⅓), acting on a
      // particle's s² of wall area, weighted by its contact (2Ψ = 1 at rest
      // distance). Applied implicitly: u_t ← u_t / (1 + dt C_f |u_t| 2Ψ / s).
      if (fric > 0) {
        const cb = i * K;
        for (let k = 0; k < K; k++) {
          const c = cand[cb + k];
          if (c < 0) break;
          const b8 = (cb + k) * BND_STRIDE;
          const psi = bnd[b8 + 6];
          if (psi <= 0) continue;
          const o = c * COLLIDER_STRIDE;
          const cf = rec[o + F.friction] >= 0 ? rec[o + F.friction] : fric;
          if (cf <= 0) continue;
          const bx = bnd[b8], by = bnd[b8 + 1], bz = bnd[b8 + 2];
          const bl = Math.sqrt(bx * bx + by * by + bz * bz);
          if (bl === 0) continue;
          const nx = -bx / bl, ny = -by / bl, nz = -bz / bl;
          const rx = vxi - bnd[b8 + 3], ry = vyi - bnd[b8 + 4], rz = vzi - bnd[b8 + 5];
          const rn = rx * nx + ry * ny + rz * nz;
          const tx = rx - rn * nx, ty = ry - rn * ny, tz = rz - rn * nz;
          const ut = Math.sqrt(tx * tx + ty * ty + tz * tz);
          if (ut === 0) continue;
          const decay = 1 - 1 / (1 + dt * cf * ut * 2 * psi * invS);
          const fx = -decay * tx, fy = -decay * ty, fz = -decay * tz;
          dvx += fx; dvy += fy; dvz += fz;
          if (rec[o + F.flags] & FLAG_DYNAMIC) {
            addImpulse(imp, impBase + rec[o + F.slot] * 6, rec, o, p[i3], p[i3 + 1], p[i3 + 2], -m * fx, -m * fy, -m * fz);
          }
        }
      }
      out[i3] = dvx; out[i3 + 1] = dvy; out[i3 + 2] = dvz;
      om[i3] = wx; om[i3 + 1] = wy; om[i3 + 2] = wz;
    }
  }

  // v += Δv (+ vorticity confinement: reads only ω and the cache)
  phaseConfine(i0, i1) {
    const dp = this.p, u = this.u, dt = u[U.stepDt];
    const v = this.vel, nbr = this.nbr, cnt = this.nbrCount, M = this.M;
    const pair = this.pair, dens = this.density, om = this.omega, dv = this.tmpB;
    const eps = u[U.vorticity] * dp.h * dt * 60;
    for (let i = i0; i < i1; i++) {
      const i3 = i * 3;
      let vx = v[i3] + dv[i3], vy = v[i3 + 1] + dv[i3 + 1], vz = v[i3 + 2] + dv[i3 + 2];
      if (eps > 0) {
        const wxi = om[i3], wyi = om[i3 + 1], wzi = om[i3 + 2];
        const wi = Math.sqrt(wxi * wxi + wyi * wyi + wzi * wzi);
        if (wi > 1e-6) {
          let ex = 0, ey = 0, ez = 0;
          const base = i * M, e = base + cnt[i];
          for (let t = base; t < e; t++) {
            const j3 = nbr[t] * 3, t4 = t * 4;
            const wj = Math.sqrt(om[j3] * om[j3] + om[j3 + 1] * om[j3 + 1] + om[j3 + 2] * om[j3 + 2]);
            const s = (wj - wi) / dens[nbr[t]];
            ex += s * pair[t4]; ey += s * pair[t4 + 1]; ez += s * pair[t4 + 2];
          }
          const el = Math.sqrt(ex * ex + ey * ey + ez * ez);
          if (el > 1e-9) {
            // η = Σ (|ω_j| − |ω_i|) ∇_i W_ij points toward larger |ω|
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

  // x += dt·v, then keep particles out of solids: a particle closer than the
  // contact distance is moved out along the surface normal (leaving through
  // the surface it crossed this substep if it went in) and loses the velocity
  // component into the surface — inelastic, no launch.
  phaseIntegrate(i0, i1, tid) {
    const dp = this.p, u = this.u, dt = u[U.stepDt];
    const p = this.pos, v = this.vel;
    const maxV = u[U.maxSpeed], maxV2 = maxV * maxV;
    const rec = this.colliders, hf = this.heightfields, cand = this.cand, n = this._n, cv = this._cv;
    const imp = this.impulses, impBase = tid * this.maxColliders * 6, m = dp.particleMass;
    const rmin = dp.contactMin * dp.spacing;
    const wd = this._wd, wn = this._wn;
    for (let i = i0; i < i1; i++) {
      const i3 = i * 3;
      let vx = v[i3], vy = v[i3 + 1], vz = v[i3 + 2];
      const s2 = vx * vx + vy * vy + vz * vz;
      if (s2 > maxV2) { const s = maxV / Math.sqrt(s2); vx *= s; vy *= s; vz *= s; }
      const px = p[i3], py = p[i3 + 1], pz = p[i3 + 2];
      let x = px + vx * dt, y = py + vy * dt, z = pz + vz * dt;
      const cb = i * K;
      for (let k = 0; k < K; k++) {
        const c = cand[cb + k];
        if (c < 0) break;
        if (rec[c * COLLIDER_STRIDE] === SHAPE.container) {
          // each wall on its own (a corner needs both)
          containerWalls(rec, c, x, y, z, wd, wn);
          for (let w = 0; w < 5; w++) {
            const pen = rmin - wd[w];
            if (pen <= 0) continue;
            const nx = wn[w * 3], ny = wn[w * 3 + 1], nz = wn[w * 3 + 2];
            x += pen * nx; y += pen * ny; z += pen * nz;
            colliderVelocity(rec, c, x, y, z, cv);
            const vn = (vx - cv[0]) * nx + (vy - cv[1]) * ny + (vz - cv[2]) * nz;
            if (vn < 0) {
              vx -= vn * nx; vy -= vn * ny; vz -= vn * nz;
              const o = c * COLLIDER_STRIDE;
              if (rec[o + F.flags] & FLAG_DYNAMIC) {
                addImpulse(imp, impBase + rec[o + F.slot] * 6, rec, o, x, y, z, m * vn * nx, m * vn * ny, m * vn * nz);
              }
            }
          }
          continue;
        }
        let d = colliderSDF(rec, c, hf, x, y, z, n);
        if (d >= rmin) continue;
        let sx = x, sy = y, sz = z;
        if (d < 0) {
          const d0 = colliderSDF(rec, c, hf, px, py, pz, cv);
          if (d0 > 0) {
            const t = d0 / (d0 - d);
            sx = px + (x - px) * t; sy = py + (y - py) * t; sz = pz + (z - pz) * t;
            d = colliderSDF(rec, c, hf, sx, sy, sz, n);
          }
        }
        const pen = rmin - d;
        x = sx + pen * n[0]; y = sy + pen * n[1]; z = sz + pen * n[2];
        colliderVelocity(rec, c, x, y, z, cv);
        const vn = (vx - cv[0]) * n[0] + (vy - cv[1]) * n[1] + (vz - cv[2]) * n[2];
        if (vn < 0) {
          vx -= vn * n[0]; vy -= vn * n[1]; vz -= vn * n[2];
          const o = c * COLLIDER_STRIDE;
          if (rec[o + F.flags] & FLAG_DYNAMIC) {
            addImpulse(imp, impBase + rec[o + F.slot] * 6, rec, o, x, y, z, m * vn * n[0], m * vn * n[1], m * vn * n[2]);
          }
        }
      }
      p[i3] = x; p[i3 + 1] = y; p[i3 + 2] = z;
      v[i3] = vx; v[i3 + 1] = vy; v[i3 + 2] = vz;
    }
  }

  /* ================= driver ================= */

  runPhase(id, i0, i1, tid = 0) {
    switch (id) {
      case PHASE.hash: return this.phaseHash(i0, i1);
      case PHASE.gather: return this.phaseGather(i0, i1);
      case PHASE.scatterBack: return this.phaseScatterBack(i0, i1);
      case PHASE.neighbors: return this.phaseNeighbors(i0, i1, tid);
      case PHASE.density: return this.phaseDensity(i0, i1);
      case PHASE.divergenceResidual: return this.phaseResidual(i0, i1, tid, false);
      case PHASE.densityResidual: return this.phaseResidual(i0, i1, tid, true);
      case PHASE.divergenceGate: return this.phaseResidual(i0, i1, tid, false, false);
      case PHASE.densityGate: return this.phaseResidual(i0, i1, tid, true, false);
      case PHASE.velocity: return this.phaseVelocity(i0, i1, tid);
      case PHASE.forces: return this.phaseForces(i0, i1, tid);
      case PHASE.confine: return this.phaseConfine(i0, i1);
      case PHASE.integrate: return this.phaseIntegrate(i0, i1, tid);
      case PHASE.warmDensity: return this.phaseWarm(i0, i1, C_DENSITY, this.p.warmStart);
      case PHASE.warmDivergence: return this.phaseWarm(i0, i1, C_DIVERGENCE, this.p.warmStart);
      default: throw new Error(`unknown phase ${id}`);
    }
  }

  _maxReduce(k) {
    const r = this.reduce;
    let s = 0;
    for (let t = 0; t < this.threads; t++) { s = Math.max(s, r[t * 8 + k]); r[t * 8 + k] = 0; }
    return s;
  }

  /**
   * Advance by dt, in as many CFL substeps as the fastest particle needs
   * (≤ maxSubsteps). `parallel(id)` runs a phase over all particles — by
   * default on this thread; the thread pool splits the range across workers.
   */
  step(dt, parallel = null) {
    const T = this.phaseMs, u = this.u, dp = this.p;
    T.fill(0);
    const t0 = performance.now();
    this.header[H.overflow] = 0;
    const exec = parallel ?? ((id) => this.runPhase(id, 0, this.header[H.count], 0));
    const run = (id) => { const t = performance.now(); exec(id); T[id] += performance.now() - t; };
    // CFL: the fastest particle (plus what gravity adds over the step) moves
    // at most cfl · spacing per substep
    const g = Math.hypot(u[U.gx], u[U.gy], u[U.gz]);
    const vmax = u[U.maxVel] + g * dt;
    const maxSub = Math.max(1, u[U.maxSubsteps] | 0);
    const sub = Math.min(maxSub, Math.max(dp.minSubsteps, Math.ceil(dt * vmax / (u[U.cfl] * dp.spacing))));
    const sdt = dt / sub;
    let pIt = 0, dIt = 0, avgErr = 0, maxErr = 0;
    for (let s = 0; s < sub; s++) {
      u[U.dt] = sdt; u[U.stepDt] = sdt;
      this._advanceColliders(sdt);
      const n = this.header[H.count];
      if (n > 0) {
        run(PHASE.hash);
        let t = performance.now();
        this.phaseSort();
        T[TIMING.sort] += performance.now() - t;
        run(PHASE.gather);
        run(PHASE.scatterBack);
        run(PHASE.neighbors);
        this._contactShares();
        run(PHASE.density);
        // divergence-free solve
        const maxD = u[U.maxDivergenceIterations] | 0;
        if (maxD > 0) {
          run(PHASE.divergenceGate);
          this._sumReduce(R_ERR); this._maxReduce(R_MAX);
          run(PHASE.warmDivergence);
          run(PHASE.velocity);
          run(PHASE.divergenceResidual);
          let err = this._sumReduce(R_ERR) / n; this._maxReduce(R_MAX);
          const tol = u[U.divergenceTolerance] * dp.rho0;
          for (let it = 0; it < maxD && (err > tol || it < 1); it++) {
            run(PHASE.velocity);
            run(PHASE.divergenceResidual);
            err = this._sumReduce(R_ERR) / n; this._maxReduce(R_MAX);
            dIt++;
          }
        }
        run(PHASE.forces);
        run(PHASE.confine);
        // constant-density solve
        run(PHASE.densityGate);
        this._sumReduce(R_ERR); this._maxReduce(R_MAX);
        run(PHASE.warmDensity);
        run(PHASE.velocity);
        run(PHASE.densityResidual);
        let err = this._sumReduce(R_ERR) / n, emax = this._maxReduce(R_MAX);
        const tol = u[U.densityTolerance] * dp.rho0, maxIt = dp.maxIterations;
        const minIt = dp.minIterations;
        for (let it = 0; it < maxIt && (err > tol || it < minIt); it++) {
          run(PHASE.velocity);
          run(PHASE.densityResidual);
          err = this._sumReduce(R_ERR) / n; emax = this._maxReduce(R_MAX);
          pIt++;
        }
        avgErr = Math.max(avgErr, err * dp.invRho0);
        maxErr = Math.max(maxErr, emax * dp.invRho0);
        run(PHASE.integrate);
      }
      const tf = performance.now();
      this.finalizeParticles();
      T[TIMING.finalize] += performance.now() - tf;
    }
    u[U.dt] = dt;
    u[U.substeps] = sub;
    u[U.pressureIterations] = pIt / sub;
    u[U.divergenceIterations] = dIt / sub;
    u[U.avgDensityError] = avgErr;
    u[U.maxDensityError] = maxErr;
    T[TIMING.total] = performance.now() - t0;
  }
}
