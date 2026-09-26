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

import { deriveParams, wallFraction, wallFractionSlope } from './params.js';
import {
  COLLIDER_STRIDE, F, FLAG_DYNAMIC, colliderSDF, colliderVelocity,
} from './colliders.js';

export const MAX_COLLIDER_CANDIDATES = 4;

// Start-of-step overlap (fraction of the contact radius) resolved by the
// ordinary contact response; deeper overlaps are removed without velocity.
const DEPEN_SLOP = 0.25;

// Header (Int32) slots shared between threads.
export const H = {
  count: 0, colliders: 1, overflow: 2, removed: 3, nextId: 4, tableMask: 5,
  frame: 6, leaked: 7, quarantined: 8, drained: 9,
};
// Uniforms (Float64) — values that can change between steps.
export const U = {
  dt: 0, gx: 1, gy: 2, gz: 3, viscosity: 4, vorticity: 5, cohesion: 6,
  friction: 7, maxSpeed: 8, iterations: 9, wallDensity: 10,
  boundsOn: 11, bminX: 12, bminY: 13, bminZ: 14, bmaxX: 15, bmaxY: 16, bmaxZ: 17,
  kineticEnergy: 18, maxDensityError: 19, omega: 20,
};
const U_SIZE = 32;

// Parallel phases (dispatch ids shared with worker threads).
export const PHASE = {
  predict: 1, gather: 2, scatterBack: 3, neighbors: 4, lambda: 5, delta: 6,
  apply: 7, frictionVelocity: 8, vorticity1: 9, vorticity2: 10,
};
// Timing slots in solver.phaseMs (last step): parallel phases by id, plus
export const TIMING = { sort: 11, finalize: 12, total: 13 };
export const TIMING_NAMES = ['', 'predict', 'gather', 'scatterBack', 'neighbors', 'lambda', 'delta',
  'apply', 'frictionVelocity', 'vorticity1', 'vorticity2', 'sort', 'finalize', 'total'];

function nextPow2(n) { let p = 1; while (p < n) p <<= 1; return p; }

/**
 * Describe (and allocate) every buffer the solver needs. `alloc(bytes)`
 * returns an ArrayBuffer or SharedArrayBuffer. Thread-local scratch is not
 * part of this — each thread creates its own.
 */
export function allocateBuffers(dp, { alloc = (n) => new ArrayBuffer(n), threads = 1, maxColliders = 64 } = {}) {
  const N = dp.maxParticles;
  const M = dp.maxNeighbors;
  const T = nextPow2(Math.max(64, 2 * N));
  const f32 = (n) => alloc(n * 4), i32 = (n) => alloc(n * 4), f64 = (n) => alloc(n * 8);
  return {
    N, M, T, threads, maxColliders,
    header: i32(16),
    uniforms: f64(U_SIZE),
    pos: f32(N * 3), prev: f32(N * 3), vel: f32(N * 3), framePrev: f32(N * 3), id: i32(N),
    tmpA: f32(N * 3), tmpB: f32(N * 3), tmpC: f32(N * 3), tmpD: f32(N * 3), tmpI: i32(N),
    key: i32(N), perm: i32(N),
    bucketStart: i32(T + 1),
    nbrCount: i32(N), nbr: i32(N * M),
    candidates: i32(N * MAX_COLLIDER_CANDIDATES),
    density: f32(N), lambda: f32(N), omega: f32(N * 3),
    colliders: f32(maxColliders * COLLIDER_STRIDE),
    impulses: f64(threads * maxColliders * 6),
    contacts: f64(threads * maxColliders * 4),   // per thread, per slot: [count, Σvx, Σvy, Σvz] this step
    contactStats: f64(maxColliders * 4),         // per slot, last step: [count, mean vx, vy, vz]
  };
}

export class PBFSolver {
  /**
   * @param {object} params    user params (see params.js DEFAULTS) or derived params
   * @param {object} [buffers] from allocateBuffers (shared when threaded)
   * @param {object} [opts]    { tid, threads, init }
   */
  constructor(params = {}, buffers = null, { tid = 0, init = true, heightfields = [] } = {}) {
    const dp = params.rho0 ? params : deriveParams(params);
    this.p = dp;
    const b = buffers ?? allocateBuffers(dp);
    this.buffers = b;
    this.tid = tid;
    this.threads = b.threads;
    this.N = b.N; this.M = b.M; this.T = b.T;
    this.maxColliders = b.maxColliders;
    this.header = new Int32Array(b.header);
    this.u = new Float64Array(b.uniforms);
    this.pos = new Float32Array(b.pos); this.prev = new Float32Array(b.prev);
    this.vel = new Float32Array(b.vel); this.framePrev = new Float32Array(b.framePrev);
    this.id = new Int32Array(b.id);
    this.tmpA = new Float32Array(b.tmpA); this.tmpB = new Float32Array(b.tmpB);
    this.tmpC = new Float32Array(b.tmpC); this.tmpD = new Float32Array(b.tmpD);
    this.tmpI = new Int32Array(b.tmpI);
    this.key = new Int32Array(b.key); this.perm = new Int32Array(b.perm);
    this.bucketStart = new Int32Array(b.bucketStart);
    this.nbrCount = new Int32Array(b.nbrCount); this.nbr = new Int32Array(b.nbr);
    this.cand = new Int32Array(b.candidates);
    this.density = new Float32Array(b.density); this.lambda = new Float32Array(b.lambda);
    this.omega = new Float32Array(b.omega);
    this.colliders = new Float32Array(b.colliders);
    this.impulses = new Float64Array(b.impulses);
    this.contacts = new Float64Array(b.contacts);
    this.contactStats = new Float64Array(b.contactStats);
    this.heightfields = heightfields; // Float32Array per heightfield (packHeightfield layout)

    // thread-local scratch
    this._stamp = new Int32Array(this.T);
    this._stampVal = 0;
    this._buckets = new Int32Array(27);
    this._candIdx = new Int32Array(512);        // per-cell packed neighbor candidates
    this._candPos = new Float64Array(512 * 3);
    this._n = new Float64Array(3);
    this._cv = new Float64Array(3);
    this.phaseMs = new Float64Array(TIMING_NAMES.length); // wall time per phase, last step

    // cohesion kernel normalization on the rest lattice
    this._cohNorm = 0;
    {
      const s = dp.spacing, h = dp.h, R = Math.ceil(dp.kernelScale) + 1;
      for (let x = -R; x <= R; x++) for (let y = -R; y <= R; y++) for (let z = -R; z <= R; z++) {
        const r = s * Math.hypot(x, y, z);
        if (r > 0 && r < h) this._cohNorm += cohesionShape(r / h);
      }
      this._cohNorm = 1 / this._cohNorm;
    }

    if (init) {
      this.header[H.tableMask] = this.T - 1;
      this.setUniforms(dp);
    }
  }

  get count() { return this.header[H.count]; }

  /** Copy tunables into the shared uniform block (safe between steps). */
  setUniforms(p) {
    const u = this.u;
    const g = p.gravity ?? this.p.gravity;
    u[U.gx] = g[0]; u[U.gy] = g[1]; u[U.gz] = g[2];
    u[U.viscosity] = p.viscosity ?? this.p.viscosity;
    u[U.vorticity] = p.vorticity ?? this.p.vorticity;
    u[U.cohesion] = p.cohesion ?? this.p.cohesion;
    u[U.friction] = p.friction ?? this.p.friction;
    u[U.maxSpeed] = p.maxSpeed ?? this.p.maxSpeed;
    u[U.iterations] = p.iterations ?? this.p.iterations;
    u[U.omega] = p.sor ?? this.p.sor;
    u[U.wallDensity] = (p.wallDensity ?? this.p.wallDensity) ? 1 : 0;
    const bounds = p.bounds === undefined ? this.p.bounds : p.bounds;
    if (bounds) {
      u[U.boundsOn] = 1;
      u[U.bminX] = bounds.min[0]; u[U.bminY] = bounds.min[1]; u[U.bminZ] = bounds.min[2];
      u[U.bmaxX] = bounds.max[0]; u[U.bmaxY] = bounds.max[1]; u[U.bmaxZ] = bounds.max[2];
    } else if (p.bounds === null) {
      u[U.boundsOn] = 0;
    }
    for (const k of ['gravity', 'viscosity', 'vorticity', 'cohesion', 'friction', 'maxSpeed', 'iterations', 'sor', 'wallDensity', 'bounds']) {
      if (p[k] !== undefined && p !== this.p) this.p[k] = p[k];
    }
  }

  /* ================= serial helpers (coordinator only) ================= */

  /** Append one particle. Returns false when full or non-finite. */
  addParticle(x, y, z, vx = 0, vy = 0, vz = 0) {
    const n = this.header[H.count];
    if (n >= this.N) return false;
    if (!(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) &&
          Number.isFinite(vx) && Number.isFinite(vy) && Number.isFinite(vz))) return false;
    if (this.insideSolid(x, y, z)) return false;
    const i3 = n * 3;
    this.pos[i3] = x; this.pos[i3 + 1] = y; this.pos[i3 + 2] = z;
    this.prev[i3] = x; this.prev[i3 + 1] = y; this.prev[i3 + 2] = z;
    this.framePrev[i3] = x; this.framePrev[i3 + 1] = y; this.framePrev[i3 + 2] = z;
    this.vel[i3] = vx; this.vel[i3 + 1] = vy; this.vel[i3 + 2] = vz;
    this.id[n] = this.header[H.nextId]++;
    this.nbrCount[n] = 0;
    this.header[H.count] = n + 1;
    return true;
  }

  /** Is (x,y,z) strictly inside any collider? Spawns there are discarded. */
  insideSolid(x, y, z) {
    const rec = this.colliders, hf = this.heightfields, nc = this.header[H.colliders];
    for (let c = 0; c < nc; c++) if (colliderSDF(rec, c, hf, x, y, z, this._n) < 0) return true;
    return false;
  }

  /** Swap-remove particle i (order is rebuilt by the next sort anyway). */
  removeParticle(i) {
    const last = --this.header[H.count];
    if (i !== last) {
      const i3 = i * 3, l3 = last * 3;
      for (let k = 0; k < 3; k++) {
        this.pos[i3 + k] = this.pos[l3 + k];
        this.prev[i3 + k] = this.prev[l3 + k];
        this.vel[i3 + k] = this.vel[l3 + k];
        this.framePrev[i3 + k] = this.framePrev[l3 + k];
      }
      this.id[i] = this.id[last];
      this.nbrCount[i] = this.nbrCount[last];
    }
  }

  /** Remove every particle inside an axis-aligned box. Returns the count. */
  removeInBox(min, max) {
    let removed = 0;
    const p = this.pos;
    for (let i = this.header[H.count] - 1; i >= 0; i--) {
      const x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
      if (x >= min[0] && x <= max[0] && y >= min[1] && y <= max[1] && z >= min[2] && z <= max[2]) {
        this.removeParticle(i); removed++;
      }
    }
    this.header[H.drained] += removed;
    return removed;
  }

  /** Start of a render frame: interpolation origin for every live particle. */
  markFrame() {
    this.framePrev.set(this.pos.subarray(0, this.header[H.count] * 3));
  }

  /* ================= per-particle phases ================= */

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

  // 2a. counting sort by bucket (serial, O(n + table))
  phaseSort() {
    const n = this.header[H.count];
    const T = this.T, start = this.bucketStart, key = this.key, perm = this.perm;
    start.fill(0);
    for (let i = 0; i < n; i++) start[key[i] + 1]++;
    for (let b = 0; b < T; b++) start[b + 1] += start[b];
    // perm[slot] = old index, using start[] as the per-bucket cursor; the
    // cursors end at the next bucket's start, so shifting by one restores it
    for (let i = 0; i < n; i++) perm[start[key[i]]++] = i;
    for (let b = T; b > 0; b--) start[b] = start[b - 1];
    start[0] = 0;
  }

  // 2b. gather particle arrays into sorted order (into tmp buffers)
  phaseGather(i0, i1) {
    const perm = this.perm;
    const p = this.pos, pr = this.prev, v = this.vel, fp = this.framePrev, id = this.id;
    const A = this.tmpA, B = this.tmpB, C = this.tmpC, D = this.tmpD, I = this.tmpI;
    for (let k = i0; k < i1; k++) {
      const s = perm[k], s3 = s * 3, k3 = k * 3;
      A[k3] = p[s3]; A[k3 + 1] = p[s3 + 1]; A[k3 + 2] = p[s3 + 2];
      B[k3] = pr[s3]; B[k3 + 1] = pr[s3 + 1]; B[k3 + 2] = pr[s3 + 2];
      C[k3] = v[s3]; C[k3 + 1] = v[s3 + 1]; C[k3 + 2] = v[s3 + 2];
      D[k3] = fp[s3]; D[k3 + 1] = fp[s3 + 1]; D[k3 + 2] = fp[s3 + 2];
      I[k] = id[s];
    }
  }

  // 2c. copy the sorted tmp buffers back (and recompute keys in sorted order)
  phaseScatterBack(i0, i1) {
    const a = i0 * 3, b = i1 * 3;
    this.pos.set(this.tmpA.subarray(a, b), a);
    this.prev.set(this.tmpB.subarray(a, b), a);
    this.vel.set(this.tmpC.subarray(a, b), a);
    this.framePrev.set(this.tmpD.subarray(a, b), a);
    this.id.set(this.tmpI.subarray(i0, i1), i0);
  }

  // 3. neighbor lists + collider candidates. Particles are processed per grid
  // cell: the candidates from the 27 surrounding buckets are gathered once per
  // cell, pre-filtered to those within h of the cell's box (drops hash
  // collisions and far corners), packed contiguously, then tested per particle.
  phaseNeighbors(i0, i1, tid = 0) {
    const p = this.pos, h = this.p.h, h2 = h * h, inv = 1 / h;
    const M = this.M, nbr = this.nbr, cnt = this.nbrCount, start = this.bucketStart;
    const mask = this.header[H.tableMask];
    const stamp = this._stamp, buckets = this._buckets;
    let cIdx = this._candIdx, cPos = this._candPos;
    let overflow = 0;
    let i = i0;
    while (i < i1) {
      const i3 = i * 3;
      const cx = Math.floor(p[i3] * inv), cy = Math.floor(p[i3 + 1] * inv), cz = Math.floor(p[i3 + 2] * inv);
      // run of consecutive particles in the same cell
      let e = i + 1;
      while (e < i1) {
        const e3 = e * 3;
        if (Math.floor(p[e3] * inv) !== cx || Math.floor(p[e3 + 1] * inv) !== cy || Math.floor(p[e3 + 2] * inv) !== cz) break;
        e++;
      }
      // deduplicated buckets of the 27 neighbor cells
      if (++this._stampVal === 0x7fffffff) { stamp.fill(0); this._stampVal = 1; }
      const sv = this._stampVal;
      let nb = 0;
      for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const bk = hashCell(cx + dx, cy + dy, cz + dz) & mask;
        if (stamp[bk] !== sv) { stamp[bk] = sv; buckets[nb++] = bk; }
      }
      // gather candidates within h of this cell's box
      const x0 = cx * h, y0 = cy * h, z0 = cz * h, x1 = x0 + h, y1 = y0 + h, z1 = z0 + h;
      let nc = 0;
      for (let t = 0; t < nb; t++) {
        const bk = buckets[t], be = start[bk + 1];
        for (let j = start[bk]; j < be; j++) {
          const j3 = j * 3;
          const x = p[j3], y = p[j3 + 1], z = p[j3 + 2];
          const ex = x < x0 ? x0 - x : x > x1 ? x - x1 : 0;
          const ey = y < y0 ? y0 - y : y > y1 ? y - y1 : 0;
          const ez = z < z0 ? z0 - z : z > z1 ? z - z1 : 0;
          if (ex * ex + ey * ey + ez * ez >= h2) continue;
          if (nc >= cIdx.length) {
            const ni = new Int32Array(cIdx.length * 2); ni.set(cIdx); cIdx = this._candIdx = ni;
            const np = new Float64Array(cPos.length * 2); np.set(cPos); cPos = this._candPos = np;
          }
          cIdx[nc] = j; cPos[nc * 3] = x; cPos[nc * 3 + 1] = y; cPos[nc * 3 + 2] = z;
          nc++;
        }
      }
      // per-particle distance tests against the packed candidates
      for (let a = i; a < e; a++) {
        const a3 = a * 3;
        const xa = p[a3], ya = p[a3 + 1], za = p[a3 + 2];
        const base = a * M;
        let k = 0;
        for (let c = 0; c < nc; c++) {
          const c3 = c * 3;
          const dx = xa - cPos[c3], dy = ya - cPos[c3 + 1], dz = za - cPos[c3 + 2];
          const r2 = dx * dx + dy * dy + dz * dz;
          if (r2 < h2) {
            const j = cIdx[c];
            if (j === a) continue;
            if (k < M) nbr[base + k++] = j;
            else overflow++;
          }
        }
        cnt[a] = k;
      }
      i = e;
    }
    if (overflow) Atomics.add(this.header, H.overflow, overflow);
    this._collectCandidates(i0, i1, tid);
  }

  // Colliders within reach of each particle this step (≤ MAX_COLLIDER_CANDIDATES),
  // plus per-slot counts of particles touching dynamic colliders (contact shares).
  _collectCandidates(i0, i1, tid) {
    const contacts = this.contacts, conBase = tid * this.maxColliders * 4, v = this.vel;
    const touch = this.p.particleRadius + 0.25 * this.p.spacing;
    const nc = this.header[H.colliders];
    const cand = this.cand, K = MAX_COLLIDER_CANDIDATES;
    const p = this.pos, rec = this.colliders, hf = this.heightfields, n = this._n;
    const reach = this.p.h + this.p.maxSpeed * this.u[U.dt] * 0.25;
    for (let i = i0; i < i1; i++) {
      const i3 = i * 3, base = i * K;
      let k = 0;
      for (let c = 0; c < nc && k < K; c++) {
        const d = colliderSDF(rec, c, hf, p[i3], p[i3 + 1], p[i3 + 2], n);
        if (d < reach) {
          cand[base + k++] = c;
          const o = c * COLLIDER_STRIDE;
          if (d < touch && (rec[o + F.flags] & FLAG_DYNAMIC)) {
            const b = conBase + rec[o + F.slot] * 4;
            contacts[b]++; contacts[b + 1] += v[i3]; contacts[b + 2] += v[i3 + 1]; contacts[b + 3] += v[i3 + 2];
          }
        }
      }
      if (k < K) cand[base + k] = -1;
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
    const u = this.u, h = this.header;
    const p = this.pos, v = this.vel;
    // non-finite quarantine + bounds kill (backward: swap-remove safe)
    const bOn = u[U.boundsOn] > 0;
    const x0 = u[U.bminX], y0 = u[U.bminY], z0 = u[U.bminZ];
    const x1 = u[U.bmaxX], y1 = u[U.bmaxY], z1 = u[U.bmaxZ];
    for (let i = h[H.count] - 1; i >= 0; i--) {
      const i3 = i * 3;
      const x = p[i3], y = p[i3 + 1], z = p[i3 + 2];
      if (!(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) &&
            Number.isFinite(v[i3]) && Number.isFinite(v[i3 + 1]) && Number.isFinite(v[i3 + 2]))) {
        this.removeParticle(i); h[H.quarantined]++;
      } else if (bOn && (x < x0 || x > x1 || y < y0 || y > y1 || z < z0 || z > z1)) {
        this.removeParticle(i); h[H.leaked]++;
      }
    }
    let ke = 0, maxErr = 0;
    const n = h[H.count], invRho0 = this.p.invRho0, dens = this.density;
    for (let i = 0; i < n * 3; i++) ke += v[i] * v[i];
    for (let i = 0; i < n; i++) { const e = dens[i] * invRho0 - 1; if (e > maxErr) maxErr = e; }
    u[U.kineticEnergy] = 0.5 * this.p.particleMass * ke;
    u[U.maxDensityError] = maxErr;
    h[H.frame]++;
  }

  /** Reduce per-thread collider impulses into slot sums; returns a Float64Array view. */
  reduceImpulses(out) {
    const C = this.maxColliders * 6, imp = this.impulses;
    out.fill(0);
    for (let t = 0; t < this.threads; t++) {
      const b = t * C;
      for (let k = 0; k < C; k++) out[k] += imp[b + k];
    }
    imp.fill(0);
    return out;
  }

  // Mass-weighted contact for light dynamic bodies. The solver treats a
  // collider as immovable within a step, so a particle it touches gets the
  // collider's full velocity and the body receives the full reaction later:
  // for a light body that is an elastic-plus kick (a 10 kg crate dropped at
  // 2 m/s rebounded at 6 m/s). Splitting each correction as an inelastic
  // contact between the body (mass M) and the N particles it touched last
  // step gives particles the share α = M / (M + N·m). N is counted during
  // candidate collection of this step (serial, after the neighbor phase).
  _contactShares() {
    const rec = this.colliders, nc = this.header[H.colliders], m = this.p.particleMass;
    const C4 = this.maxColliders * 4, con = this.contacts, T = this.threads, cs = this.contactStats;
    cs.fill(0);
    for (let c = 0; c < nc; c++) {
      const o = c * COLLIDER_STRIDE;
      if (!(rec[o + F.flags] & FLAG_DYNAMIC)) { rec[o + F.alpha] = 1; continue; }
      const s4 = rec[o + F.slot] * 4;
      let N = 0, vx = 0, vy = 0, vz = 0;
      for (let t = 0; t < T; t++) {
        const b = t * C4 + s4;
        N += con[b]; vx += con[b + 1]; vy += con[b + 2]; vz += con[b + 3];
      }
      if (N > 0) { cs[s4] = N; cs[s4 + 1] = vx / N; cs[s4 + 2] = vy / N; cs[s4 + 3] = vz / N; }
      const M = rec[o + F.mass];
      rec[o + F.alpha] = M > 0 ? M / (M + Math.max(1, N) * m) : 1;
    }
    con.fill(0);
  }

  // Moving colliders (kinematic/dynamic bodies) are advanced along their
  // linear + angular velocity by dt before each step, so across a batch of k
  // steps a collider sweeps to where the rigid-body engine will have moved it
  // instead of staying frozen and then jumping into the fluid next batch.
  _advanceColliders(dt) {
    const rec = this.colliders, nc = this.header[H.colliders];
    for (let c = 0; c < nc; c++) {
      const o = c * COLLIDER_STRIDE;
      const vx = rec[o + F.vx], vy = rec[o + F.vy], vz = rec[o + F.vz];
      const wx = rec[o + F.wx], wy = rec[o + F.wy], wz = rec[o + F.wz];
      if (vx === 0 && vy === 0 && vz === 0 && wx === 0 && wy === 0 && wz === 0) continue;
      rec[o + F.px] += vx * dt; rec[o + F.py] += vy * dt; rec[o + F.pz] += vz * dt;
      if (wx !== 0 || wy !== 0 || wz !== 0) {
        // q ← normalize(q + ½·dt·(ω, 0)·q)
        const qx = rec[o + F.qx], qy = rec[o + F.qy], qz = rec[o + F.qz], qw = rec[o + F.qw];
        const h = 0.5 * dt;
        let nx = qx + h * (wx * qw + wy * qz - wz * qy);
        let ny = qy + h * (wy * qw + wz * qx - wx * qz);
        let nz = qz + h * (wz * qw + wx * qy - wy * qx);
        let nw = qw - h * (wx * qx + wy * qy + wz * qz);
        const l = 1 / Math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw);
        rec[o + F.qx] = nx * l; rec[o + F.qy] = ny * l; rec[o + F.qz] = nz * l; rec[o + F.qw] = nw * l;
      }
    }
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

// Spatial hash of an integer cell (Teschner et al. 2003 primes).
export function hashCell(x, y, z) {
  return (Math.imul(x, 92837111) ^ Math.imul(y, 689287499) ^ Math.imul(z, 283923481)) >>> 0;
}

// Cohesion profile: 0 at r=0 and r=h, peak near r≈0.6h (Akinci-like shape).
function cohesionShape(u) {
  if (u >= 1 || u <= 0) return 0;
  const a = (1 - u) * u;
  return a * a * a;
}

function addImpulse(imp, b, rec, o, x, y, z, jx, jy, jz) {
  imp[b] += jx; imp[b + 1] += jy; imp[b + 2] += jz;
  const rx = x - rec[o + F.px], ry = y - rec[o + F.py], rz = z - rec[o + F.pz];
  imp[b + 3] += ry * jz - rz * jy;
  imp[b + 4] += rz * jx - rx * jz;
  imp[b + 5] += rx * jy - ry * jx;
}
