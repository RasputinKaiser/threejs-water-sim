// water/core/fluid-core.js — what every particle solver shares: the buffers
// (shareable between worker threads), particle bookkeeping, the spatial hash
// grid (counting sort by hash bucket + per-cell neighbor search), collider
// candidates, finalisation and rigid-body impulse bookkeeping.
//
// Every per-particle phase is written in GATHER form (particle i writes only
// its own outputs), so any range split over threads gives bit-identical
// results. Phases take (i0, i1, tid); serial phases run on one thread.

import { deriveParams } from './params.js';
import { COLLIDER_STRIDE, F, FLAG_DYNAMIC, colliderSDF } from './colliders.js';

export const MAX_COLLIDER_CANDIDATES = 4;
// per particle, per candidate collider (DFSPH boundary cache):
// ∇Ψ xyz · boundary velocity xyz · volume fraction Ψ · signed distance
export const BND_STRIDE = 8;

// Header (Int32) slots shared between threads.
export const H = {
  count: 0, colliders: 1, overflow: 2, removed: 3, nextId: 4, tableMask: 5,
  frame: 6, leaked: 7, quarantined: 8, drained: 9,
};
// Uniforms (Float64) — values that can change between steps, plus stats.
export const U = {
  dt: 0, gx: 1, gy: 2, gz: 3, viscosity: 4, vorticity: 5, cohesion: 6,
  friction: 7, maxSpeed: 8, iterations: 9, wallDensity: 10,
  boundsOn: 11, bminX: 12, bminY: 13, bminZ: 14, bmaxX: 15, bmaxY: 16, bmaxZ: 17,
  kineticEnergy: 18, maxDensityError: 19, omega: 20,
  // DFSPH
  maxVel: 21, pressureIterations: 22, divergenceIterations: 23, substeps: 24,
  avgDensityError: 25, densityTolerance: 26, divergenceTolerance: 27,
  maxDivergenceIterations: 28, cfl: 29, maxSubsteps: 30, stepDt: 31,
};
const U_SIZE = 40;

// Phase ids shared by all solvers (dispatch ids for worker threads).
export const PHASE = { gather: 2, scatterBack: 3, neighbors: 4 };

// wall-time slots in solver.phaseMs beyond the phase ids
export const TIMING = { sort: 40, finalize: 41, total: 42 };
const TIMING_SLOTS = 44;

function nextPow2(n) { let p = 1; while (p < n) p <<= 1; return p; }

/**
 * Describe (and allocate) every buffer a solver needs. `alloc(bytes)`
 * returns an ArrayBuffer or SharedArrayBuffer. Thread-local scratch is not
 * part of this — each thread creates its own.
 */
export function allocateBuffers(dp, { alloc = (n) => new ArrayBuffer(n), threads = 1, maxColliders = 64 } = {}) {
  const N = dp.maxParticles;
  const M = dp.maxNeighbors;
  const K = MAX_COLLIDER_CANDIDATES;
  const T = nextPow2(Math.max(64, 2 * N));
  const f32 = (n) => alloc(n * 4), i32 = (n) => alloc(n * 4), f64 = (n) => alloc(n * 8);
  const dfsph = dp.solver === 'dfsph';
  return {
    N, M, T, threads, maxColliders, solver: dp.solver,
    header: i32(16),
    uniforms: f64(U_SIZE),
    pos: f32(N * 3), prev: f32(N * 3), vel: f32(N * 3), framePrev: f32(N * 3), id: i32(N),
    tmpA: f32(N * 3), tmpB: f32(N * 3), tmpC: f32(N * 3), tmpD: f32(N * 3), tmpI: i32(N),
    key: i32(N), perm: i32(N),
    bucketStart: i32(T + 1),
    nbrCount: i32(N), nbr: i32(N * M),
    candidates: i32(N * K),
    density: f32(N), lambda: f32(N), omega: f32(N * 3),
    colliders: f32(maxColliders * COLLIDER_STRIDE),
    impulses: f64(threads * maxColliders * 6),
    contacts: f64(threads * maxColliders * 4),   // per thread, per slot: [count, Σvx, Σvy, Σvz] this step
    contactStats: f64(maxColliders * 4),         // per slot, last step: [count, mean vx, vy, vz]
    // DFSPH: per-pair kernel cache [∇W xyz, W], boundary cache, solver scalars,
    // per-thread reductions
    pair: dfsph ? f32(N * M * 4) : null,
    bnd: dfsph ? f32(N * K * BND_STRIDE) : null,
    beta: dfsph ? f32(N) : null,
    kappa: dfsph ? f32(N) : null,
    // per-particle state that persists across steps and travels with the
    // particles through the sort (DFSPH warm start: pressure, divergence)
    carry: f32(N * (dfsph ? 2 : 0)), carryTmp: f32(N * (dfsph ? 2 : 0)),
    carryCount: dfsph ? 2 : 0,
    reduce: f64(threads * 8),
  };
}

export class FluidCore {
  /**
   * @param {object} params    user params (see params.js DEFAULTS) or derived params
   * @param {object} [buffers] from allocateBuffers (shared when threaded)
   * @param {object} [opts]    { tid, init, heightfields }
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
    this.reduce = new Float64Array(b.reduce);
    this.carryCount = b.carryCount;
    this.carry = [];
    this.carryTmp = [];
    for (let c = 0; c < b.carryCount; c++) {
      this.carry.push(new Float32Array(b.carry, c * this.N * 4, this.N));
      this.carryTmp.push(new Float32Array(b.carryTmp, c * this.N * 4, this.N));
    }
    this.heightfields = heightfields; // Float32Array per heightfield (packHeightfield layout)

    // thread-local scratch
    this._stamp = new Int32Array(this.T);
    this._stampVal = 0;
    this._buckets = new Int32Array(27);
    this._candIdx = new Int32Array(512);        // per-cell packed neighbor candidates
    this._candPos = new Float64Array(512 * 3);
    this._n = new Float64Array(3);
    this._cv = new Float64Array(3);
    this.phaseMs = new Float64Array(TIMING_SLOTS); // wall time per phase id, last step

    if (init) {
      this.header[H.tableMask] = this.T - 1;
      this.setUniforms(dp);
    }
  }

  get count() { return this.header[H.count]; }

  /** Names of the phaseMs slots in use (for profiling output). */
  get phaseNames() {
    const names = { [TIMING.sort]: 'sort', [TIMING.finalize]: 'finalize', [TIMING.total]: 'total' };
    for (const [k, v] of Object.entries(this.constructor.PHASES ?? PHASE)) names[v] = k;
    return names;
  }

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
    u[U.densityTolerance] = p.densityTolerance ?? this.p.densityTolerance;
    u[U.divergenceTolerance] = p.divergenceTolerance ?? this.p.divergenceTolerance;
    u[U.maxDivergenceIterations] = p.maxDivergenceIterations ?? this.p.maxDivergenceIterations;
    u[U.cfl] = p.cfl ?? this.p.cfl;
    u[U.maxSubsteps] = p.maxSubsteps ?? this.p.maxSubsteps;
    const bounds = p.bounds === undefined ? this.p.bounds : p.bounds;
    if (bounds) {
      u[U.boundsOn] = 1;
      u[U.bminX] = bounds.min[0]; u[U.bminY] = bounds.min[1]; u[U.bminZ] = bounds.min[2];
      u[U.bmaxX] = bounds.max[0]; u[U.bmaxY] = bounds.max[1]; u[U.bmaxZ] = bounds.max[2];
    } else if (p.bounds === null) {
      u[U.boundsOn] = 0;
    }
    for (const k of ['gravity', 'viscosity', 'vorticity', 'cohesion', 'friction', 'maxSpeed', 'iterations', 'sor',
      'wallDensity', 'bounds', 'densityTolerance', 'divergenceTolerance', 'maxDivergenceIterations', 'cfl', 'maxSubsteps',
      'minSubsteps', 'maxIterations', 'minIterations']) {
      if (p[k] !== undefined && p !== this.p) this.p[k] = p[k];
    }
  }

  /* ================= serial helpers (coordinator only) ================= */

  /** Append one particle. Returns false when full, non-finite or inside a solid. */
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
    for (let c = 0; c < this.carryCount; c++) this.carry[c][n] = 0;
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
      for (let c = 0; c < this.carryCount; c++) this.carry[c][i] = this.carry[c][last];
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

  /* ================= hash grid ================= */

  // hash key of each particle's grid cell (cell size h) at its current position
  phaseHash(i0, i1) {
    const p = this.pos, key = this.key;
    const inv = 1 / this.p.h, mask = this.header[H.tableMask];
    for (let i = i0; i < i1; i++) {
      const i3 = i * 3;
      key[i] = hashCell(Math.floor(p[i3] * inv), Math.floor(p[i3 + 1] * inv), Math.floor(p[i3 + 2] * inv)) & mask;
    }
  }

  // counting sort by bucket (serial, O(n + table))
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

  // gather particle arrays into sorted order (into tmp buffers)
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
    for (let c = 0; c < this.carryCount; c++) {
      const src = this.carry[c], dst = this.carryTmp[c];
      for (let k = i0; k < i1; k++) dst[k] = src[perm[k]];
    }
  }

  // copy the sorted tmp buffers back
  phaseScatterBack(i0, i1) {
    const a = i0 * 3, b = i1 * 3;
    this.pos.set(this.tmpA.subarray(a, b), a);
    this.prev.set(this.tmpB.subarray(a, b), a);
    this.vel.set(this.tmpC.subarray(a, b), a);
    this.framePrev.set(this.tmpD.subarray(a, b), a);
    this.id.set(this.tmpI.subarray(i0, i1), i0);
    for (let c = 0; c < this.carryCount; c++) this.carry[c].set(this.carryTmp[c].subarray(i0, i1), i0);
  }

  // Neighbor lists + collider candidates. Particles are processed per grid
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

  /* ================= serial finalize ================= */

  // remove out-of-bounds / non-finite particles; kinetic energy, max speed
  finalizeParticles() {
    const u = this.u, h = this.header;
    const p = this.pos, v = this.vel;
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
    let ke = 0, v2max = 0;
    const n = h[H.count];
    for (let i = 0; i < n; i++) {
      const i3 = i * 3, s2 = v[i3] * v[i3] + v[i3 + 1] * v[i3 + 1] + v[i3 + 2] * v[i3 + 2];
      ke += s2;
      if (s2 > v2max) v2max = s2;
    }
    u[U.kineticEnergy] = 0.5 * this.p.particleMass * ke;
    u[U.maxVel] = Math.sqrt(v2max);
    h[H.frame]++;
  }

  /** Reduce per-thread collider impulses into slot sums; returns `out`. */
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
  // contact between the body (mass M) and the N particles it touched this
  // step gives particles the share α = M / (M + N·m). N is counted during
  // candidate collection (serial, after the neighbor phase).
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
        const nx = qx + h * (wx * qw + wy * qz - wz * qy);
        const ny = qy + h * (wy * qw + wz * qx - wx * qz);
        const nz = qz + h * (wz * qw + wx * qy - wy * qx);
        const nw = qw - h * (wx * qx + wy * qy + wz * qz);
        const l = 1 / Math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw);
        rec[o + F.qx] = nx * l; rec[o + F.qy] = ny * l; rec[o + F.qz] = nz * l; rec[o + F.qw] = nw * l;
      }
    }
  }

  /** Sum of the per-thread reduction slot `k` (and clear it). */
  _sumReduce(k) {
    const r = this.reduce;
    let s = 0;
    for (let t = 0; t < this.threads; t++) { s += r[t * 8 + k]; r[t * 8 + k] = 0; }
    return s;
  }
}

// Spatial hash of an integer cell (Teschner et al. 2003 primes).
export function hashCell(x, y, z) {
  return (Math.imul(x, 92837111) ^ Math.imul(y, 689287499) ^ Math.imul(z, 283923481)) >>> 0;
}

export function addImpulse(imp, b, rec, o, x, y, z, jx, jy, jz) {
  imp[b] += jx; imp[b + 1] += jy; imp[b + 2] += jz;
  const rx = x - rec[o + F.px], ry = y - rec[o + F.py], rz = z - rec[o + F.pz];
  imp[b + 3] += ry * jz - rz * jy;
  imp[b + 4] += rz * jx - rx * jz;
  imp[b + 5] += rx * jy - ry * jx;
}

/** Health / cost figures of the last step (plain object, safe to post). */
export function solverStats(solver) {
  const u = solver.u, h = solver.header;
  return {
    kineticEnergy: u[U.kineticEnergy], maxDensityError: u[U.maxDensityError],
    avgDensityError: solver.p.solver === 'dfsph' ? u[U.avgDensityError] : undefined,
    pressureIterations: u[U.pressureIterations], divergenceIterations: u[U.divergenceIterations],
    substeps: u[U.substeps] || 1, maxSpeed: u[U.maxVel],
    overflow: h[H.overflow], leaked: h[H.leaked], quarantined: h[H.quarantined], drained: h[H.drained],
  };
}
