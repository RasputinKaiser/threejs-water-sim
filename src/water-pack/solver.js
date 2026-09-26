// water-pack/solver.js — Position-Based Fluid solver (Clavet et al. 2005
// double-density relaxation) tuned for game-scale realistic water: pours,
// splashes, and settles FLAT when filling containers of any size.
// Collider types: plane, box (axis-aligned), heightfield (terrain).
// Diagnostics: bounds-leak detection, NaN quarantine, kinetic energy and
// per-phase timings (sim.phaseMs) every step.
//
// Step pipeline (see step()):
//   predict   gravity + explicit position prediction (particle order)
//   sort      counting-sort particles by grid cell, gather pos/vel into a
//             cell-ordered WORKING COPY (_sp/_sv)
//   pairs     ONE cell-pair walk over the working copy: builds the shared
//             pair list, scatters densities, counts neighbors and applies
//             the viscosity impulses (Gauss-Seidel over pairs)
//   relax     double-density relaxation off the stored pair list
//   scatter   working copy → particle order (incl. viscosity displacement)
//   collide   collider projection + restitution/friction
//   derive    v = Δx/dt (speed-clamped), leak kill, NaN quarantine, KE
//
// Why a cell-ordered working copy: neighbor loops then read contiguous runs
// of memory instead of chasing random particle indices, which is what
// dominated step time (perf-headroom.md). The copy is internal — pos/vel/
// nCount keep their particle order, so renderers, the async worker's
// double-buffered interpolation and the foam system see stable indices.

export const DEFAULT_PARAMS = {
  gravity: -9.81,
  // interaction radius (m). Particle spacing = h * spacingRatio (below)
  h: 0.3,
  // Lattice spacing as a fraction of h — used by spawnBlock. Default 0.55.
  // Raising it (e.g. 0.62) coarsens the fluid: pairs/particle scales
  // ~ ratio⁻³ (perf-headroom.md §P3). IMPORTANT: restDensity must be kept
  // consistent with it — see the exported helper restDensityForSpacing(ratio).
  spacingRatio: 0.55,
  // ρ0 = Clavet kernel sum for the 0.55h cubic spawn lattice: 12q²+6q³ @ q=0.45
  // ≈ 2.98. This is dimensionless — identical at every h (it scales WITH h).
  restDensity: 3.0,       // ρ0 (= restDensityForSpacing(0.55), rounded)
  stiffness: 10,          // k   (bulk) — 6-10 settles flat without writhing
  nearStiffness: 30,      // kNear (anti-clump)
  // Pairwise viscosity impulses (Clavet §4.1): I = dt·q·(σ·u + β·u²) for
  // approaching pairs, u = inward relative speed. These act for real now:
  // until the viscosity pass also displaced positions, step()'s v = Δx/dt
  // derive discarded every impulse, so σ/β had no effect at all. Values are
  // re-tuned for water that damps particle jitter but still sloshes: in a
  // 5×5 m tank σ=2/β=0.5 keeps ~95% of the slosh energy at t=2 s and halves
  // resting jitter (rms 0.118 → 0.051 m/s) vs σ=β=0; σ=40 (the old scene
  // value) kills ~86% of the slosh by t=2 s — syrup. Calm containers use ~4/1.
  viscositySigma: 2,      // linear viscosity
  viscosityBeta: 0.5,     // quadratic viscosity
  maxParticles: 9000,
  maxSpeed: 12,           // m/s clamp — PBF safety net against density spikes
  restitution: 0.1,       // water barely bounces
  contactFriction: 0.25,  // tangential damping on collider contact
  // Riverbed friction: used INSTEAD of contactFriction when colliding with a
  // HEIGHTFIELD surface. Defaults to the same value as contactFriction;
  // creek/river scenes set it low (~0.03-0.06) so water keeps moving down
  // gentle grades instead of stalling.
  bedFriction: 0.25,
  // Optional extra push for riverbed flow, applied ONLY while a particle is in
  // contact with a heightfield: acceleration = slopeAssist * |gravity| pointed
  // DOWN the local bed gradient (skipped where the gradient is ~0, so pools on
  // flat ground are untouched). 0 = off. On gentle creek grades (2-5%) the
  // every-step Coulomb stop term beats gravity's downslope component even at
  // low bedFriction; 0.5-1.0 keeps flow lively.
  slopeAssist: 0,
  killLeaks: true,        // remove particles that escape the sim bounds
  // Deep-pool packing trims (research/perf-headroom.md Priority 3). Both
  // default OFF; they only trim the LARGEST compression kicks.
  //   packMaxDisp: hard clamp (meters) on a single pair-relaxation
  //     displacement |Δx|. Infinity = off. Try 0.002–0.003 for deep pools.
  //   packDamp: 0..1 fractional damping of a pair's relaxation displacement
  //     ONLY when its average density pressure exceeds the rest level k·ρ0.
  packMaxDisp: Infinity,
  packDamp: 0,
};

// Rest density consistent with a spawn lattice at spacingRatio·h: the Clavet
// kernel sum ρ0 = 12q² + 6q³ with effective q = 1 − ratio (the nearest-
// neighbor shell at r = ratio·h gives q = 1 − r/h).
//   restDensityForSpacing(0.55) ≈ 2.977 (shipped restDensity rounds to 3.0)
//   restDensityForSpacing(0.62) ≈ 2.062
export function restDensityForSpacing(ratio) {
  const q = 1 - ratio;
  return 12 * q * q + 6 * q * q * q;
}

// Hash grid: GRID_DIM cells per axis, coordinates wrap via mask. Cells are
// h wide, so the grid repeats every GRID_DIM·h meters (45 m at h=0.35);
// wrapped aliases are rejected by the distance test.
const GRID_BITS = 7;
const GRID_DIM = 1 << GRID_BITS;
const GRID_MASK = GRID_DIM - 1;

// per-pair geometry layout in _pairData: [q, nx, ny, nz]
const PAIR_STRIDE = 4;

// Half-open 13-neighborhood: one of each ±cell pair (own cell is handled
// separately with b > a), listed row by row with x ascending. In cell-sorted
// slot order the cells x-1, x, x+1 of one row occupy adjacent slot runs, so
// _buildPairs merges them into one contiguous candidate range (≤5 ranges per
// cell instead of 14 runs). Every unordered pair within h is visited EXACTLY
// once across the whole walk.
const OFFS = [
  1, 0, 0,                           // own row: x+1
  -1, 1, 0,   0, 1, 0,   1, 1, 0,    // row (dy, dz) = (+1, 0)
  -1, -1, 1,  0, -1, 1,  1, -1, 1,   // row (-1, +1)
  -1, 0, 1,   0, 0, 1,   1, 0, 1,    // row ( 0, +1)
  -1, 1, 1,   0, 1, 1,   1, 1, 1,    // row (+1, +1)
];

// phase keys of sim.phaseMs, in step order
export const PHASES = ['predict', 'sort', 'pairs', 'relax', 'scatter', 'collide', 'derive'];

function _grow(oldArr, newArr) { newArr.set(oldArr); return newArr; }

export class WaterSim {
  constructor(params = {}) {
    this.p = { ...DEFAULT_PARAMS, ...params };
    this.count = 0;
    const cap = this.p.maxParticles;
    this.pos = new Float32Array(cap * 3);
    this.prev = new Float32Array(cap * 3);
    this.vel = new Float32Array(cap * 3);
    this.nCount = new Int32Array(cap); // neighbor count per particle (foam signal)

    // Linked-list grid over CURRENT positions (particle order). Built lazily
    // by ensureGrid() for the spawn overlap guard and external readers
    // (effects.computeCohesionField); step() no longer needs it.
    const D3 = GRID_DIM * GRID_DIM * GRID_DIM;
    this.cellHead = new Int32Array(D3).fill(-1);
    this.next = new Int32Array(cap);
    this._gridTouched = new Int32Array(cap); // cells whose head != -1
    this._nGridTouched = 0;
    this._gridValid = false;
    this._gridH = 0;

    // Cell sort scratch. Cell tables are indexed by cell id and only trusted
    // for cells stamped THIS step (_cellStamp === _stamp): a cell vacated
    // since an earlier step keeps stale start/count values.
    this._cellCnt = new Int32Array(D3);
    this._cellStart = new Int32Array(D3);
    this._cellStamp = new Int32Array(D3);
    this._stamp = 0;
    this._cells = new Int32Array(1 << 12); // occupied cell ids (growable)
    this._ncells = 0;
    this._key = new Int32Array(cap);       // cell id per particle
    this._perm = new Int32Array(cap);      // sorted slot → particle index

    // cell-ordered working copy + per-slot scratch
    this._sp = new Float32Array(cap * 3);
    this._sv = new Float32Array(cap * 3);
    this.rho = new Float32Array(cap);
    this.rhoNear = new Float32Array(cap);
    this.pressure = new Float32Array(cap);
    this.pressureNear = new Float32Array(cap);
    this._nc = new Int32Array(cap);

    // SHARED pair list, grouped by first endpoint a (slot order):
    // pairs of slot a are [_pairStart[a], _pairStart[a+1]); _pairB holds the
    // other endpoint, _pairData the geometry [q (=1-r/h), nx, ny, nz].
    // Growable, reused — zero steady-state allocation.
    this._pairCap = 1 << 18;
    this._npairs = 0;
    this._pairStart = new Int32Array(cap + 1);
    this._pairB = new Int32Array(this._pairCap);
    this._pairData = new Float32Array(this._pairCap * PAIR_STRIDE);
    this._runS = new Int32Array(14);
    this._runE = new Int32Array(14);

    this.simMs = 0;
    this.phaseMs = Object.fromEntries(PHASES.map((k) => [k, 0]));
    this.gridDim = GRID_DIM;
    this.kineticEnergy = 0;
  }

  get particleCount() { return this.count; }

  spawn(x, y, z, vx = 0, vy = 0, vz = 0) {
    if (this.count >= this.p.maxParticles) return false;
    if (!(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) &&
          Number.isFinite(vx) && Number.isFinite(vy) && Number.isFinite(vz))) return false;
    // Overlap guard: spawning inside an existing particle detonates the
    // near-pressure term. Reject positions closer than 0.35h to any particle,
    // including ones spawned earlier in the same frame.
    this.ensureGrid();
    const minD2 = (this.h * 0.35) ** 2;
    const p = this.pos, inv = 1 / this.cellSize;
    const cx = Math.floor(x * inv), cy = Math.floor(y * inv), cz = Math.floor(z * inv);
    const head = this.cellHead, next = this.next;
    for (let gz = -1; gz <= 1; gz++) for (let gy = -1; gy <= 1; gy++) for (let gx = -1; gx <= 1; gx++) {
      let j = head[((cx + gx) & GRID_MASK) | (((cy + gy) & GRID_MASK) << GRID_BITS) | (((cz + gz) & GRID_MASK) << (2 * GRID_BITS))];
      while (j !== -1) {
        const dx = p[j * 3] - x, dy = p[j * 3 + 1] - y, dz = p[j * 3 + 2] - z;
        if (dx * dx + dy * dy + dz * dz < minD2) return false;
        j = next[j];
      }
    }
    const i = this.count++;
    this.pos[i * 3] = x; this.pos[i * 3 + 1] = y; this.pos[i * 3 + 2] = z;
    this.prev[i * 3] = x; this.prev[i * 3 + 1] = y; this.prev[i * 3 + 2] = z;
    this.vel[i * 3] = vx; this.vel[i * 3 + 1] = vy; this.vel[i * 3 + 2] = vz;
    this.nCount[i] = 0;
    this._gridInsert(i, (cx & GRID_MASK) | ((cy & GRID_MASK) << GRID_BITS) | ((cz & GRID_MASK) << (2 * GRID_BITS)));
    return true;
  }

  spawnBlock(cx, cy, cz, nx, ny, nz, jitter = 0.02, v0 = 0) {
    // spawn an nx×ny×nz block of particles at spacing spacingRatio·h centered at c
    const s = this.p.h * this.p.spacingRatio;
    let added = 0;
    for (let ix = 0; ix < nx; ix++)
      for (let iy = 0; iy < ny; iy++)
        for (let iz = 0; iz < nz; iz++) {
          const x = cx + (ix - nx / 2) * s + (Math.random() - 0.5) * jitter;
          const y = cy + iy * s + (Math.random() - 0.5) * jitter;
          const z = cz + (iz - nz / 2) * s + (Math.random() - 0.5) * jitter;
          if (this.spawn(x, y, z, 0, -v0, 0)) added++;
        }
    return added;
  }

  removeParticle(i) {
    const last = --this.count;
    for (let k = 0; k < 3; k++) {
      this.pos[i * 3 + k] = this.pos[last * 3 + k];
      this.prev[i * 3 + k] = this.prev[last * 3 + k];
      this.vel[i * 3 + k] = this.vel[last * 3 + k];
    }
    this.nCount[i] = this.nCount[last];
    this._gridValid = false; // indices moved
  }

  // Drain: delete particles inside an axis-aligned region (e.g. a plughole).
  // Returns the number removed.
  drain(region) {
    const before = this.count;
    const [x0, y0, z0] = region.min, [x1, y1, z1] = region.max;
    for (let i = this.count - 1; i >= 0; i--) {
      const x = this.pos[i * 3], y = this.pos[i * 3 + 1], z = this.pos[i * 3 + 2];
      if (x >= x0 && x <= x1 && y >= y0 && y <= y1 && z >= z0 && z <= z1) this.removeParticle(i);
    }
    return before - this.count;
  }

  reset() { this.count = 0; this._gridValid = false; }

  /** Make cellHead/next reflect the current positions (lazy, O(count)). */
  ensureGrid() {
    if (this._gridValid && this._gridH === this.p.h) return;
    const head = this.cellHead, touched = this._gridTouched;
    for (let k = 0; k < this._nGridTouched; k++) head[touched[k]] = -1;
    this._nGridTouched = 0;
    const inv = 1 / this.cellSize, p = this.pos;
    for (let i = 0; i < this.count; i++) {
      this._gridInsert(i,
        (Math.floor(p[i * 3] * inv) & GRID_MASK) |
        ((Math.floor(p[i * 3 + 1] * inv) & GRID_MASK) << GRID_BITS) |
        ((Math.floor(p[i * 3 + 2] * inv) & GRID_MASK) << (2 * GRID_BITS)));
    }
    this._gridValid = true;
    this._gridH = this.p.h;
  }

  _gridInsert(i, cell) {
    const head = this.cellHead;
    if (head[cell] === -1) this._gridTouched[this._nGridTouched++] = cell;
    this.next[i] = head[cell];
    head[cell] = i;
  }

  step(dt, colliders) {
    const ph = this.phaseMs;
    const t0 = performance.now();
    this._gridValid = false;

    // Quarantine particles corrupted from outside (NaN/Inf pos or vel)
    // BEFORE they enter the pair walk — once paired, a NaN spreads through
    // densities and displacements to every neighbor.
    this._quarantine(true);

    const p = this.pos, v = this.vel, pr = this.prev;
    const n = this.count;
    const g = this.p.gravity;

    // 1. gravity + predict
    for (let i = 0; i < n; i++) {
      const i3 = i * 3;
      v[i3 + 1] += g * dt;
      pr[i3] = p[i3]; pr[i3 + 1] = p[i3 + 1]; pr[i3 + 2] = p[i3 + 2];
      p[i3] += v[i3] * dt; p[i3 + 1] += v[i3 + 1] * dt; p[i3 + 2] += v[i3 + 2] * dt;
    }
    let t = performance.now(); ph.predict = t - t0; let tp = t;

    // 2. cell sort + gather working copy
    this._sortByCell();
    t = performance.now(); ph.sort = t - tp; tp = t;

    // 3. pair walk: pair list + densities + neighbor counts + viscosity
    this._buildPairs(dt);
    t = performance.now(); ph.pairs = t - tp; tp = t;

    // 4. double density relaxation off the stored pair list
    this._relax(dt);
    t = performance.now(); ph.relax = t - tp; tp = t;

    // 5. working copy → particle order
    this._scatter(dt);
    t = performance.now(); ph.scatter = t - tp; tp = t;

    // 6. collisions (position projection + velocity response)
    this._collide(colliders, dt);
    t = performance.now(); ph.collide = t - tp; tp = t;

    // 7. derive velocities (clamped — PBF can spike on deep penetration)
    const invDt = 1 / dt;
    const maxV = this.p.maxSpeed, maxV2 = maxV * maxV;
    for (let i = 0; i < n; i++) {
      const i3 = i * 3;
      let vx = (p[i3] - pr[i3]) * invDt;
      let vy = (p[i3 + 1] - pr[i3 + 1]) * invDt;
      let vz = (p[i3 + 2] - pr[i3 + 2]) * invDt;
      const sp2 = vx * vx + vy * vy + vz * vz;
      if (sp2 > maxV2) {
        const s = maxV / Math.sqrt(sp2);
        vx *= s; vy *= s; vz *= s;
        // re-sync position to the clamped velocity so next frame stays consistent
        p[i3] = pr[i3] + vx * dt;
        p[i3 + 1] = pr[i3 + 1] + vy * dt;
        p[i3 + 2] = pr[i3 + 2] + vz * dt;
      }
      v[i3] = vx; v[i3 + 1] = vy; v[i3 + 2] = vz;
    }

    // diagnostics: out-of-bounds leak detection (open top: no max-y test)
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
    // A non-finite position must never reach a renderer.
    this._quarantine(false);

    let ke = 0;
    for (let i = 0, m3 = this.count * 3; i < m3; i++) ke += v[i] * v[i];
    this.kineticEnergy = 0.5 * ke;

    t = performance.now(); ph.derive = t - tp;
    this.simMs = t - t0;
  }

  // Remove particles with a non-finite position (and velocity, when
  // checkVel). One cheap well-predicted scan; the removal loop only runs if
  // the scan found something. Backward iteration is required: removeParticle
  // swaps the LAST element into slot i, which was then already checked.
  _quarantine(checkVel) {
    const p = this.pos, v = this.vel;
    const m3 = this.count * 3;
    let bad = false;
    for (let k = 0; k < m3; k++) {
      if (!Number.isFinite(p[k]) || (checkVel && !Number.isFinite(v[k]))) { bad = true; break; }
    }
    if (!bad) return;
    for (let i = this.count - 1; i >= 0; i--) {
      const i3 = i * 3;
      if (!Number.isFinite(p[i3]) || !Number.isFinite(p[i3 + 1]) || !Number.isFinite(p[i3 + 2]) ||
          (checkVel && (!Number.isFinite(v[i3]) || !Number.isFinite(v[i3 + 1]) || !Number.isFinite(v[i3 + 2])))) {
        this.quarantinedTotal = (this.quarantinedTotal ?? 0) + 1;
        this.removeParticle(i);
      }
    }
  }

  // Counting sort of particles by cell id; occupied cells visited in
  // ascending id order so x-neighbor runs are adjacent in memory. Gathers
  // pos/vel into the working copy (_sp/_sv).
  _sortByCell() {
    const n = this.count, p = this.pos, v = this.vel;
    const inv = 1 / this.cellSize;
    const key = this._key, perm = this._perm;
    const cnt = this._cellCnt, start = this._cellStart, stamp = this._cellStamp;
    let cells = this._cells, ncells = 0;
    if (this._stamp >= 0x7fffffff) { stamp.fill(0); this._stamp = 0; }
    const st = ++this._stamp;

    for (let i = 0; i < n; i++) {
      const c = (Math.floor(p[i * 3] * inv) & GRID_MASK) |
                ((Math.floor(p[i * 3 + 1] * inv) & GRID_MASK) << GRID_BITS) |
                ((Math.floor(p[i * 3 + 2] * inv) & GRID_MASK) << (2 * GRID_BITS));
      key[i] = c;
      if (stamp[c] !== st) {
        stamp[c] = st;
        cnt[c] = 0;
        if (ncells >= cells.length) cells = this._cells = _grow(cells, new Int32Array(cells.length << 1));
        cells[ncells++] = c;
      }
      cnt[c]++;
    }
    cells.subarray(0, ncells).sort();
    let acc = 0;
    for (let d = 0; d < ncells; d++) {
      const c = cells[d];
      start[c] = acc;
      acc += cnt[c];
    }
    // scatter ids (start[] as cursor), then rewind cursors to run bases
    for (let i = 0; i < n; i++) perm[start[key[i]]++] = i;
    for (let d = 0; d < ncells; d++) start[cells[d]] -= cnt[cells[d]];
    this._ncells = ncells;

    const sp = this._sp, sv = this._sv;
    for (let k = 0; k < n; k++) {
      const i3 = perm[k] * 3, k3 = k * 3;
      sp[k3] = p[i3]; sp[k3 + 1] = p[i3 + 1]; sp[k3 + 2] = p[i3 + 2];
      sv[k3] = v[i3]; sv[k3 + 1] = v[i3 + 1]; sv[k3 + 2] = v[i3 + 2];
    }
  }

  // ONE cell-pair walk over the working copy. For each occupied cell, the
  // candidate runs are the cell itself (b > a only) + the 13 half-space
  // neighbor cells; every unordered pair within h is recorded exactly once,
  // grouped by a. The same visit scatters Clavet densities, counts
  // neighbors and applies the pairwise viscosity impulse to _sv
  // (Gauss-Seidel: later pairs see earlier impulses).
  _buildPairs(dt) {
    const n = this.count;
    const sp = this._sp, sv = this._sv;
    const h = this.h, h2 = h * h, invH = 1 / h;
    const cnt = this._cellCnt, start = this._cellStart, stamp = this._cellStamp, st = this._stamp;
    const cells = this._cells, ncells = this._ncells;
    const rho = this.rho, rhoNear = this.rhoNear, nc = this._nc;
    rho.fill(0, 0, n); rhoNear.fill(0, 0, n); nc.fill(0, 0, n);
    const sigma = this.p.viscositySigma, beta = this.p.viscosityBeta;
    const viscous = sigma > 0 || beta > 0;
    const vdt = 0.5 * dt;

    const ps = this._pairStart;
    let cap = this._pairCap, pb = this._pairB, pd = this._pairData;
    let m = 0;
    const rs = this._runS, re = this._runE; // candidate runs of the current cell

    for (let d = 0; d < ncells; d++) {
      const c = cells[d];
      const cx = c & GRID_MASK, cy = (c >> GRID_BITS) & GRID_MASK, cz = c >> (2 * GRID_BITS);
      const s0 = start[c], e0 = s0 + cnt[c];
      // candidate ranges: range 0 = own cell (walked from b = a+1); each
      // further run is merged into the previous range when their slots touch
      let nr = 1;
      rs[0] = s0; re[0] = e0;
      for (let o = 0; o < 39; o += 3) {
        const ci = ((cx + OFFS[o]) & GRID_MASK) |
                   (((cy + OFFS[o + 1]) & GRID_MASK) << GRID_BITS) |
                   (((cz + OFFS[o + 2]) & GRID_MASK) << (2 * GRID_BITS));
        if (stamp[ci] !== st) continue; // empty this step (tables may be stale)
        const s = start[ci];
        if (s === re[nr - 1]) re[nr - 1] = s + cnt[ci];
        else { rs[nr] = s; re[nr] = s + cnt[ci]; nr++; }
      }

      for (let a = s0; a < e0; a++) {
        const a3 = a * 3;
        const xa = sp[a3], ya = sp[a3 + 1], za = sp[a3 + 2];
        let vax = sv[a3], vay = sv[a3 + 1], vaz = sv[a3 + 2];
        let ra = 0, rna = 0, nca = 0;
        ps[a] = m;
        for (let r = 0; r < nr; r++) {
          const kEnd = re[r];
          for (let b = r === 0 ? a + 1 : rs[r]; b < kEnd; b++) {
            const b3 = b * 3;
            const dx = sp[b3] - xa, dy = sp[b3 + 1] - ya, dz = sp[b3 + 2] - za;
            const r2 = dx * dx + dy * dy + dz * dz;
            if (!(r2 < h2 && r2 > 1e-12)) continue; // also rejects NaN
            if (m >= cap) {
              cap <<= 1;
              this._pairCap = cap;
              pb = this._pairB = _grow(pb, new Int32Array(cap));
              pd = this._pairData = _grow(pd, new Float32Array(cap * PAIR_STRIDE));
            }
            const dist = Math.sqrt(r2), invR = 1 / dist;
            const q = 1 - dist * invH;
            const ux = dx * invR, uy = dy * invR, uz = dz * invR;
            const m4 = m * PAIR_STRIDE;
            pb[m] = b;
            pd[m4] = q; pd[m4 + 1] = ux; pd[m4 + 2] = uy; pd[m4 + 3] = uz;
            m++;
            const q2 = q * q, q3 = q2 * q;
            ra += q2; rna += q3; nca++;
            rho[b] += q2; rhoNear[b] += q3; nc[b]++;
            if (viscous) {
              // inward relative speed along a→b
              const u = (vax - sv[b3]) * ux + (vay - sv[b3 + 1]) * uy + (vaz - sv[b3 + 2]) * uz;
              if (u > 0) {
                let I = vdt * q * (sigma * u + beta * u * u);
                // stability guard: never exceed the relative speed being
                // damped (each side gets I, total 2I) — otherwise the
                // quadratic term overshoots at speed and REVERSES u.
                const Icap = u * 0.45;
                if (I > Icap) I = Icap;
                const Ix = I * ux, Iy = I * uy, Iz = I * uz;
                vax -= Ix; vay -= Iy; vaz -= Iz;
                sv[b3] += Ix; sv[b3 + 1] += Iy; sv[b3 + 2] += Iz;
              }
            }
          }
        }
        rho[a] += ra; rhoNear[a] += rna; nc[a] += nca;
        sv[a3] = vax; sv[a3 + 1] = vay; sv[a3 + 2] = vaz;
      }
    }
    ps[n] = m;
    this._npairs = m;
  }

  // Double-density relaxation: per-slot pressures (negative pressure clamped),
  // then symmetric pair displacements — ±half to each endpoint, so momentum
  // is conserved exactly.
  _relax(dt) {
    const sp = this._sp;
    const k = this.p.stiffness, kNear = this.p.nearStiffness;
    const rho0 = this.p.restDensity;
    const n = this.count;
    const dt2 = dt * dt;
    const rho = this.rho, rhoNear = this.rhoNear;
    const P = this.pressure, PN = this.pressureNear;

    // Negative pressure (rho < rho0 at the surface) ATTRACTS neighbors, which
    // makes clumps neck off and remerge — the "water creature / mitosis"
    // artifact. Clamping P to >= 0 keeps cohesion from viscosity and gravity
    // only: droplets stay round, pools stop writhing.
    for (let i = 0; i < n; i++) {
      const pi = k * (rho[i] - rho0);
      P[i] = pi > 0 ? pi : 0;
      PN[i] = kNear * rhoNear[i];
    }

    const ps = this._pairStart, pb = this._pairB, pd = this._pairData;
    const packCap = this.p.packMaxDisp;
    const packDamp = this.p.packDamp;
    const trimPairs = packDamp > 0 || packCap < Infinity;
    // rest pressure level k·ρ0: a pair whose AVERAGE density pressure exceeds
    // this has local ρ > 2ρ0 — deep-pool ram-packing, exactly what we trim
    const pressThresh = k * rho0;
    const hdt2 = 0.5 * dt2;
    for (let a = 0; a < n; a++) {
      const tEnd = ps[a + 1];
      let t = ps[a];
      if (t === tEnd) continue;
      const Pa = P[a], PNa = PN[a];
      let ax = 0, ay = 0, az = 0;
      for (; t < tEnd; t++) {
        const b = pb[t], t4 = t * PAIR_STRIDE;
        const Pav = 0.5 * (Pa + P[b]);
        const PNav = 0.5 * (PNa + PN[b]);
        const q = pd[t4];
        let D = hdt2 * (Pav * q + PNav * q * q); // half-displacement per endpoint
        if (trimPairs) {
          if (packDamp > 0 && Pav > pressThresh) D *= 1 - packDamp;
          if (D > 0.5 * packCap) D = 0.5 * packCap;
        }
        const Dx = D * pd[t4 + 1], Dy = D * pd[t4 + 2], Dz = D * pd[t4 + 3];
        const b3 = b * 3;
        sp[b3] += Dx; sp[b3 + 1] += Dy; sp[b3 + 2] += Dz;
        ax -= Dx; ay -= Dy; az -= Dz;
      }
      const a3 = a * 3;
      sp[a3] += ax; sp[a3 + 1] += ay; sp[a3 + 2] += az;
    }
  }

  // Working copy → particle order. The viscosity pass changed velocities
  // AFTER positions were predicted, so the same Δv is applied to position
  // here (Δx = Δv·dt) — without it the v = Δx/dt derive discards viscosity.
  _scatter(dt) {
    const n = this.count, perm = this._perm;
    const p = this.pos, v = this.vel, ncOut = this.nCount;
    const sp = this._sp, sv = this._sv, nc = this._nc;
    for (let k = 0; k < n; k++) {
      const i = perm[k], i3 = i * 3, k3 = k * 3;
      const vx = sv[k3], vy = sv[k3 + 1], vz = sv[k3 + 2];
      p[i3] = sp[k3] + (vx - v[i3]) * dt;
      p[i3 + 1] = sp[k3 + 1] + (vy - v[i3 + 1]) * dt;
      p[i3 + 2] = sp[k3 + 2] + (vz - v[i3 + 2]) * dt;
      v[i3] = vx; v[i3 + 1] = vy; v[i3 + 2] = vz;
      ncOut[i] = nc[k];
    }
  }

  _collide(colliders, dt = 1 / 60) {
    if (!colliders || colliders.length === 0) return;
    const p = this.pos, v = this.vel, pr = this.prev;
    const r = 0.09; // particle radius for contact offset
    const restitution = this.p.restitution;
    const contactF = this.p.contactFriction, bedF = this.p.bedFriction;
    const gAbsDt = Math.abs(this.p.gravity) * dt;
    const assist = this.p.slopeAssist * Math.abs(this.p.gravity) * dt;
    for (let i = 0; i < this.count; i++) {
      const i3 = i * 3;
      let px = p[i3], py = p[i3 + 1], pz = p[i3 + 2];
      // remember incoming velocity so any collider response below can be made
      // permanent: step()'s derive pass recomputes v = (p - prev)/dt, which
      // would otherwise discard friction/restitution edits made here.
      // Shifting prev by -Δv*dt keeps the response in the derived velocity.
      const v0x = v[i3], v0y = v[i3 + 1], v0z = v[i3 + 2];
      for (let ci = 0; ci < colliders.length; ci++) {
        const c = colliders[ci];
        let nx, ny, nz, depth;
        let isBed = false, gsx = 0, gsz = 0; // heightfield gradient (bed features)
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
          if (!(fx >= 0 && fx < c.nx - 1 && fz >= 0 && fz < c.nz - 1)) continue;
          const ix = Math.floor(fx), iz = Math.floor(fz);
          const tx = fx - ix, tz = fz - iz;
          const hs = c.heights, cnx = c.nx;
          const h00 = hs[iz * cnx + ix], h10 = hs[iz * cnx + ix + 1];
          const h01 = hs[(iz + 1) * cnx + ix], h11 = hs[(iz + 1) * cnx + ix + 1];
          const ground = h00 * (1 - tx) * (1 - tz) + h10 * tx * (1 - tz) + h01 * (1 - tx) * tz + h11 * tx * tz;
          depth = (ground + r) - py;
          if (depth <= 0) continue;
          // normal from height gradient (central difference)
          const ix0 = Math.max(0, ix - 1), ix1 = Math.min(cnx - 1, ix + 1);
          const iz0 = Math.max(0, iz - 1), iz1 = Math.min(c.nz - 1, iz + 1);
          const sx = (hs[iz * cnx + ix1] - hs[iz * cnx + ix0]) / ((ix1 - ix0) * c.dx);
          const sz = (hs[iz1 * cnx + ix] - hs[iz0 * cnx + ix]) / ((iz1 - iz0) * c.dz);
          const invLen = 1 / Math.sqrt(sx * sx + 1 + sz * sz);
          nx = -sx * invLen; ny = invLen; nz = -sz * invLen;
          // riverbed contact: bedFriction / slopeAssist below
          isBed = true; gsx = sx; gsz = sz;
        } else continue;

        px += nx * depth; py += ny * depth; pz += nz * depth;
        let vx = v[i3], vy = v[i3 + 1], vz = v[i3 + 2];
        let vn = vx * nx + vy * ny + vz * nz;
        if (vn < 0) {
          // water barely bounces (the position projection above already
          // removed the penetration; this Δv becomes the small rebound)
          const f = vn * restitution;
          vx -= f * nx; vy -= f * ny; vz -= f * nz;
          vn -= f;
        }
        // Contact friction acts EVERY step while touching: proportional
        // damping + a Coulomb-style constant decel so SLOW slides actually
        // stop instead of asymptoting.
        let tx = vx - vn * nx, ty = vy - vn * ny, tz = vz - vn * nz;
        const tSp = Math.sqrt(tx * tx + ty * ty + tz * tz);
        if (tSp > 1e-6) {
          const cf = isBed ? bedF : contactF;
          let nt = tSp * (1 - cf) - cf * gAbsDt;
          if (nt < 0) nt = 0;
          const sc = nt / tSp;
          tx *= sc; ty *= sc; tz *= sc;
        }
        vx = vn * nx + tx; vy = vn * ny + ty; vz = vn * nz + tz;
        // slope assist: extra push down the riverbed gradient, only on
        // heightfield contacts with a nonzero gradient
        if (isBed && assist > 0) {
          const gradMag = Math.sqrt(gsx * gsx + gsz * gsz);
          if (gradMag > 1e-6) {
            vx -= assist * gsx / gradMag;
            vz -= assist * gsz / gradMag;
          }
        }
        v[i3] = vx; v[i3 + 1] = vy; v[i3 + 2] = vz;
      }
      p[i3] = px; p[i3 + 1] = py; p[i3 + 2] = pz;
      // persist the collider velocity response through the derive pass
      const dvx = v[i3] - v0x, dvy = v[i3 + 1] - v0y, dvz = v[i3 + 2] - v0z;
      if (dvx !== 0 || dvy !== 0 || dvz !== 0) {
        pr[i3] -= dvx * dt; pr[i3 + 1] -= dvy * dt; pr[i3 + 2] -= dvz * dt;
      }
    }
  }

  get cellSize() { return this.p.h; }
  get h() { return this.p.h; }
  get h2() { return this.p.h * this.p.h; }
}
