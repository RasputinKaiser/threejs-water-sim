// src/water-pack/solver-bench.mjs -- node benchmark comparing the ORIGINAL
// solver algorithm (OldWaterSim below, copied VERBATIM from
// `git show bbafd61:src/water-pack/solver.js`, only DEFAULT_PARAMS ->
// OLD_DEFAULT_PARAMS and `export class WaterSim` -> `class OldWaterSim`
// renamed) against the rewritten shared-pair-list WaterSim.
//
// Run: node src/water-pack/solver-bench.mjs

import { WaterSim } from './solver.js';

// ===================== BEGIN VERBATIM OLD SOLVER =====================
// water-pack/solver.js — Position-Based Fluid solver (Clavet et al. 2005
// double-density relaxation) tuned for game-scale realistic water: pours,
// splashes, and settles FLAT when filling containers of any size.
// Collider types: plane, box (axis-aligned), heightfield (terrain).
// Diagnostics: bounds-leak detection + kinetic energy per step.
//
// Perf notes: neighbor search uses a flat dense grid (cell heads + linked-next
// Int32Arrays — no Map, no GC). Viscosity + relaxation iterate pairs directly
// from the grid with zero allocations. ~6k particles ≈ few ms/step on M-series.

export const OLD_DEFAULT_PARAMS = {
  gravity: -9.81,
  // interaction radius (m). Particle spacing ≈ h*0.55
  h: 0.3,
  // ρ0 = Clavet kernel sum for the 0.55h cubic spawn lattice: 12q²+6q³ @ q=0.45
  // ≈ 2.98. This is dimensionless — identical at every h (it scales WITH h).
  // The old 8.2 was never reached by a proper lattice; scenes only "worked"
  // because poured particles ram-packed past it (over-compression as support),
  // and any block-spawned fill sat at P=0 and squashed/jiggled.
  restDensity: 3.0,       // ρ0
  stiffness: 10,          // k   (bulk) — 22 was "ridiculously stiff" (Ian, live test);
                          // 6-10 settles flat without the writhing-creature motion
  nearStiffness: 30,      // kNear (anti-clump) — was 90, same stiffness problem
  viscositySigma: 40,     // linear viscosity (high σ + impulse clamp = calm pools)
  viscosityBeta: 8,       // quadratic viscosity
  maxParticles: 9000,
  maxSpeed: 12,           // m/s clamp — PBF safety net against density spikes
  restitution: 0.1,       // water barely bounces
  contactFriction: 0.25,  // tangential damping on collider contact
  killLeaks: true,        // remove particles that escape the sim bounds
};

const GRID_DIM = 128; // hash grid dimension per axis (wraps via mask)

class OldWaterSim {
  constructor(params = {}) {
    this.p = { ...OLD_DEFAULT_PARAMS, ...params };
    this.count = 0;
    const cap = this.p.maxParticles;
    this.pos = new Float32Array(cap * 3);
    this.prev = new Float32Array(cap * 3);
    this.vel = new Float32Array(cap * 3);
    this.nCount = new Float32Array(cap); // neighbor count per particle (foam signal)

    // flat spatial grid: cellHead[cell] = first particle, next[i] = next in cell
    this.cellHead = new Int32Array(GRID_DIM * GRID_DIM * GRID_DIM).fill(-1);
    this.next = new Int32Array(cap);

    // pair scratch for the relaxation pass: j-list + precomputed dirs
    // MAX_PAIRS: 32 truncated neighbor lists caused asymmetric pressure kicks —
    // deep-pool particles have 40+ neighbors within h, and dropping an arbitrary
    // subset each step made the pool writhe ("water creature") forever.
    this.maxPairs = 64;
    this._pairJ = new Int32Array(cap * this.maxPairs);
    this._pairQ = new Float32Array(cap * this.maxPairs);
    this._pairNX = new Float32Array(cap * this.maxPairs);
    this._pairNY = new Float32Array(cap * this.maxPairs);
    this._pairNZ = new Float32Array(cap * this.maxPairs);
    this._pairLen = new Int32Array(cap);

    this.simMs = 0;
    this.gridDim = GRID_DIM;
  }

  get particleCount() { return this.count; }

  spawn(x, y, z, vx = 0, vy = 0, vz = 0) {
    if (this.count >= this.p.maxParticles) return;
    // overlap guard: spawning inside an existing particle detonates the near-pressure
    // term. Reject positions closer than 0.35h to any existing particle (grid-accelerated).
    {
      const minD2 = (this.h * 0.35) ** 2;
      const p = this.pos, inv = 1 / this.cellSize;
      const cx = Math.floor(x * inv), cy = Math.floor(y * inv), cz = Math.floor(z * inv);
      const D = this.gridDim, D2 = D * D, head = this.cellHead, next = this.next;
      for (let gx = -1; gx <= 1; gx++) for (let gy = -1; gy <= 1; gy++) for (let gz = -1; gz <= 1; gz++) {
        let j = head[((cx + gx) & (D - 1)) + ((cy + gy) & (D - 1)) * D + ((cz + gz) & (D - 1)) * D2];
        while (j !== -1) {
          const dx = p[j * 3] - x, dy = p[j * 3 + 1] - y, dz = p[j * 3 + 2] - z;
          if (dx * dx + dy * dy + dz * dz < minD2) return;
          j = next[j];
        }
      }
    }
    const i = this.count++;
    this.pos[i * 3] = x; this.pos[i * 3 + 1] = y; this.pos[i * 3 + 2] = z;
    this.prev[i * 3] = x; this.prev[i * 3 + 1] = y; this.prev[i * 3 + 2] = z;
    this.vel[i * 3] = vx; this.vel[i * 3 + 1] = vy; this.vel[i * 3 + 2] = vz;
  }

  spawnBlock(cx, cy, cz, nx, ny, nz, jitter = 0.02, v0 = 0) {
    // spawn an nx×ny×nz block of particles at spacing 0.55h centered at c
    const s = this.p.h * 0.55;
    for (let ix = 0; ix < nx; ix++)
      for (let iy = 0; iy < ny; iy++)
        for (let iz = 0; iz < nz; iz++) {
          const x = cx + (ix - nx / 2) * s + (Math.random() - 0.5) * jitter;
          const y = cy + iy * s + (Math.random() - 0.5) * jitter;
          const z = cz + (iz - nz / 2) * s + (Math.random() - 0.5) * jitter;
          this.spawn(x, y, z, 0, -v0, 0);
        }
  }

  removeParticle(i) {
    const last = --this.count;
    for (let k = 0; k < 3; k++) {
      this.pos[i * 3 + k] = this.pos[last * 3 + k];
      this.prev[i * 3 + k] = this.prev[last * 3 + k];
      this.vel[i * 3 + k] = this.vel[last * 3 + k];
    }
    this.nCount[i] = this.nCount[last];
  }

  // Drain: delete particles inside an axis-aligned region (e.g. a plughole).
  drain(region) {
    for (let i = this.count - 1; i >= 0; i--) {
      const x = this.pos[i * 3], y = this.pos[i * 3 + 1], z = this.pos[i * 3 + 2];
      if (x >= region.min[0] && x <= region.max[0] &&
          y >= region.min[1] && y <= region.max[1] &&
          z >= region.min[2] && z <= region.max[2]) {
        this.removeParticle(i);
      }
    }
  }

  reset() { this.count = 0; }

  step(dt, colliders) {
    const t0 = performance.now();
    const p = this.pos, v = this.vel, pr = this.prev;
    const n = this.count;
    const g = this.p.gravity;

    // 1. apply gravity + predict
    for (let i = 0; i < n; i++) {
      v[i * 3 + 1] += g * dt;
      pr[i * 3] = p[i * 3]; pr[i * 3 + 1] = p[i * 3 + 1]; pr[i * 3 + 2] = p[i * 3 + 2];
      p[i * 3] += v[i * 3] * dt; p[i * 3 + 1] += v[i * 3 + 1] * dt; p[i * 3 + 2] += v[i * 3 + 2] * dt;
    }

    // 2. neighbor search (flat grid, no alloc)
    this._buildGrid();

    // 3. viscosity impulses (pair-wise, before position solve)
    this._viscosity(dt);

    // 4. double density relaxation (gathers per-particle neighbor lists once)
    this._relax(dt);

    // 5. resolve collisions (position projection)
    this._collide(colliders, dt);

    // 6. derive velocities (clamped — PBF can spike on spawn overlap / deep penetration)
    const invDt = 1 / dt;
    const maxV = this.p.maxSpeed;
    for (let i = 0; i < n; i++) {
      let vx = (p[i * 3] - pr[i * 3]) * invDt;
      let vy = (p[i * 3 + 1] - pr[i * 3 + 1]) * invDt;
      let vz = (p[i * 3 + 2] - pr[i * 3 + 2]) * invDt;
      const sp2 = vx * vx + vy * vy + vz * vz;
      if (sp2 > maxV * maxV) {
        const s = maxV / Math.sqrt(sp2);
        vx *= s; vy *= s; vz *= s;
        // re-sync position to the clamped velocity so next frame stays consistent
        p[i * 3] = pr[i * 3] + vx * dt;
        p[i * 3 + 1] = pr[i * 3 + 1] + vy * dt;
        p[i * 3 + 2] = pr[i * 3 + 2] + vz * dt;
      }
      v[i * 3] = vx; v[i * 3 + 1] = vy; v[i * 3 + 2] = vz;
    }

    this.simMs = performance.now() - t0;

    // diagnostics (pack): out-of-bounds leak detection + kinetic energy
    if (this.bounds) {
      const b = this.bounds;
      let leaks = 0;
      for (let i = this.count - 1; i >= 0; i--) {
        const x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
        if (x < b.min[0] || x > b.max[0] || y < b.min[1] || z < b.min[2] || z > b.max[2]) {
          if (this.p.killLeaks) { this.leakedTotal = (this.leakedTotal ?? 0) + 1; this.removeParticle(i); continue; }
          leaks++;
        }
      }
      // particles currently out of bounds (cumulative removals in leakedTotal)
      this.leakedParticles = leaks;
    }
    {
      let ke = 0;
      for (let i = 0; i < this.count; i++) {
        ke += 0.5 * (v[i * 3] * v[i * 3] + v[i * 3 + 1] * v[i * 3 + 1] + v[i * 3 + 2] * v[i * 3 + 2]);
      }
      this.kineticEnergy = ke;
    }
  }

  // ---- flat dense grid ----
  _buildGrid() {
    const head = this.cellHead;
    head.fill(-1);
    const next = this.next;
    const inv = 1 / this.cellSize;
    const p = this.pos;
    const D = this.gridDim, D2 = D * D;
    for (let i = 0; i < this.count; i++) {
      const cx = (Math.floor(p[i * 3] * inv) & (D - 1));
      const cy = (Math.floor(p[i * 3 + 1] * inv) & (D - 1));
      const cz = (Math.floor(p[i * 3 + 2] * inv) & (D - 1));
      const cell = cx + cy * D + cz * D2;
      next[i] = head[cell];
      head[cell] = i;
    }
  }

  _viscosity(dt) {
    const p = this.pos, v = this.vel;
    const sigma = this.p.viscositySigma, beta = this.p.viscosityBeta;
    const h = this.h, h2 = h * h;
    const head = this.cellHead, next = this.next;
    const inv = 1 / this.cellSize;
    const D = this.gridDim, D2 = D * D;
    for (let i = 0; i < this.count; i++) {
      const xi = p[i * 3], yi = p[i * 3 + 1], zi = p[i * 3 + 2];
      const cx = Math.floor(xi * inv), cy = Math.floor(yi * inv), cz = Math.floor(zi * inv);
      const vix = v[i * 3], viy = v[i * 3 + 1], viz = v[i * 3 + 2];
      for (let gx = -1; gx <= 1; gx++) {
        const cellX = (cx + gx) & (D - 1);
        for (let gy = -1; gy <= 1; gy++) {
          const cellY = (cy + gy) & (D - 1);
          for (let gz = -1; gz <= 1; gz++) {
            const cellZ = (cz + gz) & (D - 1);
            let j = head[cellX + cellY * D + cellZ * D2];
            while (j !== -1) {
              if (j > i) { // each pair once
                const dx = p[j * 3] - xi, dy = p[j * 3 + 1] - yi, dz = p[j * 3 + 2] - zi;
                const r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < h2 && r2 > 1e-12) {
                  const r = Math.sqrt(r2);
                  const ux = dx / r, uy = dy / r, uz = dz / r;
                  const u = (vix - v[j * 3]) * ux + (viy - v[j * 3 + 1]) * uy + (viz - v[j * 3 + 2]) * uz;
                  if (u > 0) {
                    const q = r / h;
                    let I = dt * (1 - q) * (sigma * u + beta * u * u) * 0.5;
                    // stability guard: the impulse must never exceed the relative
                    // velocity it damps (each side gets I, total 2I). Without this
                    // the quadratic term overshoots at speed and REVERSES u —
                    // explicit-viscosity oscillation that keeps pools churning.
                    const Icap = u * 0.45;
                    if (I > Icap) I = Icap;
                    const Ix = I * ux, Iy = I * uy, Iz = I * uz;
                    v[i * 3] -= Ix; v[i * 3 + 1] -= Iy; v[i * 3 + 2] -= Iz;
                    v[j * 3] += Ix; v[j * 3 + 1] += Iy; v[j * 3 + 2] += Iz;
                  }
                }
              }
              j = next[j];
            }
          }
        }
      }
    }
  }

  _relax(dt) {
    const p = this.pos;
    const k = this.p.stiffness, kNear = this.p.nearStiffness;
    const rho0 = this.p.restDensity;
    const h = this.h, h2 = h * h;
    const dt2 = dt * dt;
    const head = this.cellHead, next = this.next;
    const inv = 1 / this.cellSize;
    const D = this.gridDim, D2 = D * D;
    const pJ = this._pairJ, pQ = this._pairQ, pNX = this._pairNX, pNY = this._pairNY, pNZ = this._pairNZ, pLen = this._pairLen;

    for (let i = 0; i < this.count; i++) {
      const xi = p[i * 3], yi = p[i * 3 + 1], zi = p[i * 3 + 2];
      const cx = Math.floor(xi * inv), cy = Math.floor(yi * inv), cz = Math.floor(zi * inv);
      let rho = 0, rhoNear = 0, cnt = 0;
      const base = i * this.maxPairs;
      for (let gx = -1; gx <= 1; gx++) {
        const cellX = (cx + gx) & (D - 1);
        for (let gy = -1; gy <= 1; gy++) {
          const cellY = (cy + gy) & (D - 1);
          for (let gz = -1; gz <= 1; gz++) {
            const cellZ = (cz + gz) & (D - 1);
            let j = head[cellX + cellY * D + cellZ * D2];
            while (j !== -1) {
              if (j !== i && cnt < this.maxPairs) {
                const dx = p[j * 3] - xi, dy = p[j * 3 + 1] - yi, dz = p[j * 3 + 2] - zi;
                const r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < h2 && r2 > 1e-12) {
                  const r = Math.sqrt(r2);
                  const q = 1 - r / h;
                  rho += q * q;
                  rhoNear += q * q * q;
                  pJ[base + cnt] = j;
                  pQ[base + cnt] = q;
                  const iny_ = 1 / r;
                  pNX[base + cnt] = dx * iny_; pNY[base + cnt] = dy * iny_; pNZ[base + cnt] = dz * iny_;
                  cnt++;
                }
              }
              j = next[j];
            }
          }
        }
      }
      pLen[i] = cnt;
      this.nCount[i] = cnt;

      // Negative pressure (rho < rho0 at the surface) ATTRACTS neighbors, which
      // makes clumps neck off and remerge — the "water creature / mitosis"
      // artifact. Clamping P to >= 0 keeps cohesion from viscosity and gravity
      // only: droplets stay round, pools stop writhing.
      const P = Math.max(0, k * (rho - rho0));
      const PNear = kNear * rhoNear;
      let dxAcc = 0, dyAcc = 0, dzAcc = 0;
      for (let t = 0; t < cnt; t++) {
        const Dmag = dt2 * (P * pQ[base + t] + PNear * pQ[base + t] * pQ[base + t]);
        const Dx = Dmag * pNX[base + t], Dy = Dmag * pNY[base + t], Dz = Dmag * pNZ[base + t];
        const j = pJ[base + t];
        p[j * 3] += Dx * 0.5; p[j * 3 + 1] += Dy * 0.5; p[j * 3 + 2] += Dz * 0.5;
        dxAcc -= Dx * 0.5; dyAcc -= Dy * 0.5; dzAcc -= Dz * 0.5;
      }
      p[i * 3] += dxAcc; p[i * 3 + 1] += dyAcc; p[i * 3 + 2] += dzAcc;
    }
  }

  _collide(colliders, dt = 1 / 60) {
    if (!colliders) return;
    const p = this.pos, v = this.vel;
    const r = 0.09; // particle radius for contact offset
    for (let i = 0; i < this.count; i++) {
      let px = p[i * 3], py = p[i * 3 + 1], pz = p[i * 3 + 2];
      for (const c of colliders) {
        let nx, ny, nz, depth;
        if (c.type === 'plane') {
          const d = (px - c.o[0]) * c.n[0] + (py - c.o[1]) * c.n[1] + (pz - c.o[2]) * c.n[2];
          depth = r - d;
          if (depth <= 0) continue;
          nx = c.n[0]; ny = c.n[1]; nz = c.n[2];
        } else if (c.type === 'box') {
          // identity-rotation boxes only (all current colliders are axis-aligned)
          const lx = px - c.c[0], ly = py - c.c[1], lz = pz - c.c[2];
          const ex = c.e[0] + r, ey = c.e[1] + r, ez = c.e[2] + r;
          const qx = Math.max(-ex, Math.min(ex, lx)) - lx;
          const qy = Math.max(-ey, Math.min(ey, ly)) - ly;
          const qz = Math.max(-ez, Math.min(ez, lz)) - lz;
          const dist2 = qx * qx + qy * qy + qz * qz;
          if (dist2 >= r * r) continue;
          if (dist2 > 1e-9) {
            const dist = Math.sqrt(dist2);
            nx = -qx / dist; ny = -qy / dist; nz = -qz / dist;
            depth = r - dist;
          } else {
            const pxs = ex - Math.abs(lx), pys = ey - Math.abs(ly), pzs = ez - Math.abs(lz);
            if (pxs <= pys && pxs <= pzs) { nx = Math.sign(lx) || 1; ny = 0; nz = 0; depth = pxs + r; }
            else if (pys <= pzs) { nx = 0; ny = Math.sign(ly) || 1; nz = 0; depth = pys + r; }
            else { nx = 0; ny = 0; nz = Math.sign(lz) || 1; depth = pzs + r; }
          }
        } else if (c.type === 'heightfield') {
          // c: { type:'heightfield', minX, minZ, nx, nz, dx, dz, heights:Float32Array }
          const fx = (px - c.minX) / c.dx, fz = (pz - c.minZ) / c.dz;
          if (fx < 0 || fx >= c.nx - 1 || fz < 0 || fz >= c.nz - 1) continue;
          const ix = Math.floor(fx), iz = Math.floor(fz);
          const tx = fx - ix, tz = fz - iz;
          const h00 = c.heights[iz * c.nx + ix],     h10 = c.heights[iz * c.nx + ix + 1];
          const h01 = c.heights[(iz + 1) * c.nx + ix], h11 = c.heights[(iz + 1) * c.nx + ix + 1];
          const ground = h00 * (1 - tx) * (1 - tz) + h10 * tx * (1 - tz) + h01 * (1 - tx) * tz + h11 * tx * tz;
          depth = (ground + r) - py;
          if (depth <= 0) continue;
          // normal from height gradient (central difference)
          const eps = 1;
          const ix0 = Math.max(0, ix - eps), ix1 = Math.min(c.nx - 1, ix + eps);
          const iz0 = Math.max(0, iz - eps), iz1 = Math.min(c.nz - 1, iz + eps);
          const sx = (c.heights[iz * c.nx + ix1] - c.heights[iz * c.nx + ix0]) / ((ix1 - ix0) * c.dx);
          const sz = (c.heights[iz1 * c.nx + ix] - c.heights[iz0 * c.nx + ix]) / ((iz1 - iz0) * c.dz);
          const invLen = 1 / Math.hypot(sx, 1, sz);
          nx = -sx * invLen; ny = invLen; nz = -sz * invLen;
        } else continue;

        px += nx * depth; py += ny * depth; pz += nz * depth;
        const vn = v[i * 3] * nx + v[i * 3 + 1] * ny + v[i * 3 + 2] * nz;
        if (vn < 0) {
          // water barely bounces
          const f = vn * this.p.restitution;
          v[i * 3] -= f * nx; v[i * 3 + 1] -= f * ny; v[i * 3 + 2] -= f * nz;
        }
        // Contact friction acts EVERY step while touching, not just on impact.
        // Previously it ran only inside `if (vn < 0)`, so resting water kept its
        // tangential slide forever (ice-skating across floors/terrain).
        // Proportional damping + a Coulomb-style constant decel so SLOW slides
        // actually come to a stop instead of asymptoting.
        const nDotV = v[i * 3] * nx + v[i * 3 + 1] * ny + v[i * 3 + 2] * nz;
        let tx = v[i * 3] - nDotV * nx, ty = v[i * 3 + 1] - nDotV * ny, tz = v[i * 3 + 2] - nDotV * nz;
        const tSp = Math.hypot(tx, ty, tz);
        if (tSp > 1e-6) {
          const cf = this.p.contactFriction;
          let nt = tSp * (1 - cf) - cf * Math.abs(this.p.gravity) * dt;
          if (nt < 0) nt = 0;
          const sc = nt / tSp;
          tx *= sc; ty *= sc; tz *= sc;
        }
        v[i * 3] = nDotV * nx + tx;
        v[i * 3 + 1] = nDotV * ny + ty;
        v[i * 3 + 2] = nDotV * nz + tz;
      }
      p[i * 3] = px; p[i * 3 + 1] = py; p[i * 3 + 2] = pz;
    }
  }

  get cellSize() { return this.p.h; }
  get h() { return this.p.h; }
  get h2() { return this.p.h * this.p.h; }
}

// ====================== END VERBATIM OLD SOLVER ======================


const H = 0.5;
const PARAMS = {
  h: H,
  maxParticles: 24000,
};
const COLLIDERS = [{ type: 'plane', o: [0, 0, 0], n: [0, 1, 0] }];

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : 0.5 * (s[mid - 1] + s[mid]);
}

function allFinite(sim) {
  for (let i = 0; i < sim.count * 3; i++) if (!Number.isFinite(sim.pos[i])) return false;
  return true;
}

function runSim(SimClass) {
  const sim = new SimClass(PARAMS);
  sim.bounds = { min: [-30, -5, -30], max: [30, 40, 30] };
  // settled-ish block: 60x24x15 lattice @ 0.55h spacing inside ~16.5 x 6.6 x 4.1 m region
  sim.spawnBlock(0, 0.3, 0, 60, 24, 15);
  const nSpawned = sim.count;
  // warmup (JIT) then 60 timed steps at dt=1/60
  for (let s = 0; s < 5; s++) sim.step(1 / 60, COLLIDERS);
  const times = [], kes = [];
  for (let s = 0; s < 60; s++) {
    sim.step(1 / 60, COLLIDERS);
    times.push(sim.simMs);
    kes.push(sim.kineticEnergy);
  }
  return { sim, times, kes, spawned: nSpawned };
}

console.log('spawning lattice + warming up both solvers...');
const t0 = performance.now();
const old_ = runSim(OldWaterSim);
const neu = runSim(WaterSim);

const oldMed = median(old_.times), newMed = median(neu.times);
const oldMean = old_.times.reduce((a, b) => a + b) / old_.times.length;
const newMean = neu.times.reduce((a, b) => a + b) / neu.times.length;

console.log('');
console.log('=== water-pack solver bench: ' + old_.spawned + ' particles, h=0.5, dt=1/60, plane floor ===');
console.log('spawned: old=' + old_.spawned + ' new=' + neu.spawned + ' (target 60*24*15=21600)');
console.log('');
console.log('per-step simMs over 60 steps:');
console.log('  old: median ' + oldMed.toFixed(3) + '  mean ' + oldMean.toFixed(3) + '  min ' + Math.min(...old_.times).toFixed(3) + '  max ' + Math.max(...old_.times).toFixed(3));
console.log('  new: median ' + newMed.toFixed(3) + '  mean ' + newMean.toFixed(3) + '  min ' + Math.min(...neu.times).toFixed(3) + '  max ' + Math.max(...neu.times).toFixed(3));
console.log('  speedup (median): ' + (oldMed / newMed).toFixed(2) + 'x   (mean): ' + (oldMean / newMean).toFixed(2) + 'x');
console.log('');
console.log('kinetic energy (first-timed / last / mean):');
console.log('  old: ' + old_.kes[0].toFixed(1) + ' / ' + old_.kes[59].toFixed(1) + ' / ' + (old_.kes.reduce((a,b)=>a+b)/60).toFixed(1));
console.log('  new: ' + neu.kes[0].toFixed(1) + ' / ' + neu.kes[59].toFixed(1) + ' / ' + (neu.kes.reduce((a,b)=>a+b)/60).toFixed(1));
console.log('leaked: old=' + (old_.sim.leakedTotal ?? 0) + ' new=' + (neu.sim.leakedTotal ?? 0));

if (!allFinite(old_.sim)) { console.error('FAIL: old sim positions contain NaN/Inf'); process.exit(1); }
if (!allFinite(neu.sim)) { console.error('FAIL: new sim positions contain NaN/Inf'); process.exit(1); }
console.log('positions finite after 60 steps: OK (both)');
console.log('pairs/step (new): ' + neu.sim._npairs + ', pair buffer cap: ' + neu.sim._pairCap);
const speedup = oldMed / newMed;
console.log(speedup >= 1.6 ? 'PASS: speedup ' + speedup.toFixed(2) + 'x >= 1.6x target'
                           : 'MISS: speedup ' + speedup.toFixed(2) + 'x < 1.6x target');
