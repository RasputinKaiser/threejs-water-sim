// pbf-water.js — Position-Based Fluid solver (Clavet et al. 2005 double-density
// relaxation) tuned for game-scale realistic water: pours, splashes, and settles
// FLAT when filling a container. Particles collide against an explicit list of
// box/plane colliders which mirror the Box3D static world.
//
// Perf notes: neighbor search uses a flat dense grid (cell heads + linked-next
// Int32Arrays — no Map, no GC). Viscosity + relaxation iterate pairs directly
// from the grid with zero allocations. ~6k particles ≈ few ms/step on M-series.

export const DEFAULT_PARAMS = {
  gravity: -9.81,
  // interaction radius (m). Particle spacing ≈ h*0.55
  h: 0.34,
  restDensity: 8.2,       // ρ0
  stiffness: 22,          // k   (bulk)
  nearStiffness: 90,      // kNear (anti-clump — this is what makes it settle flat)
  viscositySigma: 12,     // linear viscosity
  viscosityBeta: 4,       // quadratic viscosity
  maxParticles: 9000,
};

const GRID_DIM = 128; // hash grid dimension per axis (wraps via mask)

export class WaterSim {
  constructor(params = {}) {
    this.p = { ...DEFAULT_PARAMS, ...params };
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
    this._pairJ = new Int32Array(cap * 32);
    this._pairQ = new Float32Array(cap * 32);
    this._pairNX = new Float32Array(cap * 32);
    this._pairNY = new Float32Array(cap * 32);
    this._pairNZ = new Float32Array(cap * 32);
    this._pairLen = new Int32Array(cap);

    this.simMs = 0;
    this.gridDim = GRID_DIM;
  }

  get particleCount() { return this.count; }

  spawn(x, y, z, vx = 0, vy = 0, vz = 0) {
    if (this.count >= this.p.maxParticles) return;
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
    this._collide(colliders);

    // 6. derive velocities
    const invDt = 1 / dt;
    for (let i = 0; i < n; i++) {
      v[i * 3] = (p[i * 3] - pr[i * 3]) * invDt;
      v[i * 3 + 1] = (p[i * 3 + 1] - pr[i * 3 + 1]) * invDt;
      v[i * 3 + 2] = (p[i * 3 + 2] - pr[i * 3 + 2]) * invDt;
    }

    this.simMs = performance.now() - t0;
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
                    const I = dt * (1 - q) * (sigma * u + beta * u * u) * 0.5;
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
      const base = i * 32;
      for (let gx = -1; gx <= 1; gx++) {
        const cellX = (cx + gx) & (D - 1);
        for (let gy = -1; gy <= 1; gy++) {
          const cellY = (cy + gy) & (D - 1);
          for (let gz = -1; gz <= 1; gz++) {
            const cellZ = (cz + gz) & (D - 1);
            let j = head[cellX + cellY * D + cellZ * D2];
            while (j !== -1) {
              if (j !== i && cnt < 32) {
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

      const P = k * (rho - rho0);
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

  _collide(colliders) {
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
        } else continue;

        px += nx * depth; py += ny * depth; pz += nz * depth;
        const vn = v[i * 3] * nx + v[i * 3 + 1] * ny + v[i * 3 + 2] * nz;
        if (vn < 0) {
          const f = vn * 0.9; // slight restitution loss
          v[i * 3] -= f * nx; v[i * 3 + 1] -= f * ny; v[i * 3 + 2] -= f * nz;
        }
      }
      p[i * 3] = px; p[i * 3 + 1] = py; p[i * 3 + 2] = pz;
    }
  }

  get cellSize() { return this.p.h; }
  get h() { return this.p.h; }
  get h2() { return this.p.h * this.p.h; }
}
