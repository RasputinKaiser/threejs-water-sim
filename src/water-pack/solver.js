// water-pack/solver.js — Position-Based Fluid solver (Clavet et al. 2005
// double-density relaxation) tuned for game-scale realistic water: pours,
// splashes, and settles FLAT when filling containers of any size.
// Collider types: plane, box (axis-aligned), heightfield (terrain).
// Diagnostics: bounds-leak detection + kinetic energy per step.
//
// Perf notes (shared pair-list architecture): ONE flat-grid walk per step
// builds a single unordered pair list (half-open 13-neighborhood trick — own
// cell takes j>i, each of 13 half-space neighbor cells takes all j). That one
// list feeds BOTH the viscosity impulses AND the double-density relaxation
// (classic 3-pass: scatter densities during build → per-particle pressures →
// symmetric pair displacement). All scratch buffers are growable and reused —
// zero allocations in steady-state step(). ~22k particles ≈ few ms/step.

export const DEFAULT_PARAMS = {
  gravity: -9.81,
  // interaction radius (m). Particle spacing = h * spacingRatio (below)
  h: 0.3,
  // Lattice spacing as a fraction of h — used by spawnBlock and any code that
  // historically assumed the hardcoded spacing = 0.55h. Default 0.55 IS the
  // exact historical value, so existing scenes are unchanged. Raising it
  // (e.g. 0.62) coarsens the fluid: pairs/particle scales ~ ratio⁻³, so the
  // dominant pair-build phase shrinks accordingly (perf-headroom.md §P3).
  // IMPORTANT: restDensity must be kept consistent with it — see the exported
  // helper restDensityForSpacing(ratio). ρ0(0.55) ≈ 2.98; the shipped default
  // below rounds that to 3.0.
  spacingRatio: 0.55,
  // ρ0 = Clavet kernel sum for the 0.55h cubic spawn lattice: 12q²+6q³ @ q=0.45
  // ≈ 2.98. This is dimensionless — identical at every h (it scales WITH h).
  // The old 8.2 was never reached by a proper lattice; scenes only "worked"
  // because poured particles ram-packed past it (over-compression as support),
  // and any block-spawned fill sat at P=0 and squashed/jiggled.
  restDensity: 3.0,       // ρ0 (= restDensityForSpacing(0.55), rounded)
  stiffness: 10,          // k   (bulk) — 22 was "ridiculously stiff" (Ian, live test);
                          // 6-10 settles flat without the writhing-creature motion
  nearStiffness: 30,      // kNear (anti-clump) — was 90, same stiffness problem
  viscositySigma: 40,     // linear viscosity (high σ + impulse clamp = calm pools)
  viscosityBeta: 8,       // quadratic viscosity
  maxParticles: 9000,
  maxSpeed: 12,           // m/s clamp — PBF safety net against density spikes
  restitution: 0.1,       // water barely bounces
  contactFriction: 0.25,  // tangential damping on collider contact
  // Riverbed friction: used INSTEAD of contactFriction when colliding with a
  // HEIGHTFIELD surface. Defaults to the same value as contactFriction so
  // existing scenes behave identically; creek/river scenes set it low
  // (~0.03-0.06) so water keeps moving down gentle grades instead of stalling.
  bedFriction: 0.25,
  // Optional extra push for riverbed flow, applied ONLY while a particle is in
  // contact with a heightfield: acceleration = slopeAssist * |gravity| pointed
  // DOWN the local bed gradient (skipped where the gradient is ~0, so pools on
  // flat ground are untouched). 0 = off (existing scenes unchanged). Needed
  // because on gentle creek grades (2-5%) the every-step Coulomb stop term
  // beats gravity's downslope component even at low bedFriction; 0.5-1.0 keeps
  // flow lively.
  slopeAssist: 0,
  killLeaks: true,        // remove particles that escape the sim bounds
  // Deep-pool packing trims (research/perf-headroom.md Priority 3): as a pool
  // deepens it ram-packs, inflating pairs/particle 6.9 → 17.5 (measured) and
  // dominating step cost at 22k+ particles. Both knobs default OFF so existing
  // scenes stay bit-identical; they only trim the LARGEST compression kicks.
  //   packMaxDisp: hard clamp (meters) on a single pair-relaxation displacement
  //     |Δx| in _relax(). Infinity = off. Typical useful scale ~ dt²·k·ρ0·q
  //     (~0.004 m at h=0.35 defaults); try 0.002–0.003 for deep pools.
  //   packDamp: 0..1 fractional damping applied to a pair's relaxation
  //     displacement ONLY when its average density pressure exceeds the rest
  //     level k·ρ0 (i.e. the pair is genuinely over-compressed — sprays and
  //     surface particles at P≈0 are untouched). 0 = off; start at 0.5.
  packMaxDisp: Infinity,
  packDamp: 0,
  // Lane C pair-build traversal strategy for _buildPairs. 'runs' (default):
  // counting-sorted contiguous per-cell slices — kills next[] pointer-chasing.
  // 'direct': legacy linked-list traversal. 'pair': legacy traversal with a
  // deferred pairC[] density scatter (measured +54% slower @26k; kept only as
  // a bench fallback). Env override WATER_DENSITY_SCATTER for A/B benching.
  densityScatter: 'runs',
};

// Rest density consistent with a spawn lattice at spacingRatio·h, following
// the params-comment convention above: evaluate the Clavet kernel sum
// ρ0 = 12q² + 6q³ with effective q = 1 − ratio (the nearest-neighbor shell at
// r = ratio·h gives q = 1 − r/h).
//   restDensityForSpacing(0.55) ≈ 2.977 (shipped restDensity rounds to 3.0)
//   restDensityForSpacing(0.62) ≈ 2.062
// Scenes that raise spacingRatio should set restDensity to this value so the
// pressure baseline matches the coarser lattice.
export function restDensityForSpacing(ratio) {
  const q = 1 - ratio;
  return 12 * q * q + 6 * q * q * q;
}

const GRID_DIM = 128; // hash grid dimension per axis (wraps via mask)

// interleaved per-pair geometry layout in _pairData: [q, nx, ny, nz]
const PAIR_STRIDE = 4;

// Half-open 13-neighborhood: one of each ±cell pair (+ own cell handled
// separately with j>i). Guarantees every unordered pair within h is visited
// EXACTLY once across the whole grid walk.
const OFFS = [
  1, 0, 0,  0, 1, 0,  0, 0, 1,
  1, 1, 0,  1, -1, 0,  1, 0, 1,  1, 0, -1,  0, 1, 1,  0, 1, -1,
  1, 1, 1,  1, 1, -1,  1, -1, 1,  -1, 1, 1,
];

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

    // flat spatial grid: cellHead[cell] = first particle, next[i] = next in cell
    this.cellHead = new Int32Array(GRID_DIM * GRID_DIM * GRID_DIM).fill(-1);
    this.next = new Int32Array(cap);

    // per-particle density/pressure scratch (reused every step)
    this.rho = new Float32Array(cap);
    this.rhoNear = new Float32Array(cap);
    this.pressure = new Float32Array(cap);
    this.pressureNear = new Float32Array(cap);

    // Lane C pair-build acceleration scratch: contiguous per-cell RUNS.
    // The legacy walk chased next[] linked lists (~14 cell visits × ~6
    // candidates per particle — hundreds of thousands of dependent random
    // Int32 loads per step). Instead _buildPairs counting-sorts particle
    // indices by grid cell into _sorted (three sequential O(n) passes +
    // dirty-cell bookkeeping), then scans each cell's slice contiguously.
    // Density scatter stays the immediate per-pair form ('direct' semantics):
    // deferred/per-cell accumulator alternatives measured SLOWER (see
    // _buildPairs comments) — the scatter was never the bottleneck.
    const D3 = GRID_DIM * GRID_DIM * GRID_DIM;
    this._cellCnt = new Int32Array(D3);      // particles per cell (this step)
    this._cellStart = new Int32Array(D3);    // run offset into _sorted
    this._cellStamp = new Int32Array(D3);    // first-touch dedupe stamp
    this._cellDirty = new Int32Array(1 << 16);
    this._sorted = new Int32Array(1 << 15);  // growable particle ids, cell-grouped
    this._stamp = 0;
    // strategy resolved once (param > env > default 'runs') — no per-pair branch
    let ds = this.p.densityScatter;
    if ((ds == null || ds === DEFAULT_PARAMS.densityScatter) &&
        typeof process !== 'undefined' && process.env && process.env.WATER_DENSITY_SCATTER) {
      ds = process.env.WATER_DENSITY_SCATTER;
    }
    this.densityScatter = ds || 'runs';

    // SHARED pair list (growable, reused — zero steady-state allocation):
    // ONE interleaved Int32Array of endpoints [i, j] at stride 2 plus ONE
    // interleaved Float32Array of per-pair geometry [q (=1-r/h), nx, ny, nz]
    // at stride 4 (PAIR_STRIDE). Interleaving keeps the sequential per-pair
    // reads in _viscosity/_relax on two cache streams instead of six separate
    // SoA arrays. The old _pairR (r) was written but never read by any
    // consumer — dropped.
    // Initial capacity covers ~12 neighbors/particle; doubles on demand.
    this._pairCap = 1 << 18;
    this._npairs = 0;
    this._pairEnds = new Int32Array(this._pairCap * 2);
    this._pairData = new Float32Array(this._pairCap * PAIR_STRIDE);
    this._pairC = new Float32Array(this._pairCap);     // deferred-density variant
    this._pairCN = new Float32Array(this._pairCap);    // (densityScatter='pair')

    this.simMs = 0;
    this.gridDim = GRID_DIM;
    // overlap-guard support: cellHead is empty until the first step() builds
    // the grid; without this flag spawn()'s duplicate rejection silently
    // no-ops for any spawning done BEFORE the first step (fuzz S3 finding).
    this._gridBuilt = false;
  }

  get particleCount() { return this.count; }

  spawn(x, y, z, vx = 0, vy = 0, vz = 0) {
    if (this.count >= this.p.maxParticles) return;
    // overlap guard: spawning inside an existing particle detonates the near-pressure
    // term. Reject positions closer than 0.35h to any existing particle (grid-accelerated).
    {
      // lazy grid build: the rejection walk needs cellHead populated; before
      // the first step() it is all -1 and the guard would accept duplicates.
      // Stays in pre-step mode (_gridBuilt false) until step() runs; each
      // accepted spawn below inserts itself so consecutive spawns see each other.
      if (!this._gridBuilt && this.count > 0) this._buildGrid();
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
    if (!this._gridBuilt) {
      // pre-first-step: keep the guard's grid current by inserting the new
      // particle into its cell's linked list (O(1)) so the NEXT spawn in this
      // batch is checked against it
      const D = this.gridDim, D2 = D * D, inv = 1 / this.cellSize;
      const cx = Math.floor(x * inv) & (D - 1);
      const cy = Math.floor(y * inv) & (D - 1);
      const cz = Math.floor(z * inv) & (D - 1);
      const cell = cx + cy * D + cz * D2;
      this.next[i] = this.cellHead[cell];
      this.cellHead[cell] = i;
    }
  }

  spawnBlock(cx, cy, cz, nx, ny, nz, jitter = 0.02, v0 = 0) {
    // spawn an nx×ny×nz block of particles at spacing spacingRatio·h centered at c
    const s = this.p.h * this.p.spacingRatio;
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
    this._gridBuilt = true;

    // 3. ONE shared pair list: builds pairs, scatters densities, counts neighbors
    this._buildPairs();

    // 4. viscosity impulses off the shared pair list (before position solve)
    this._viscosity(dt);

    // 5. double density relaxation off the same pair list (symmetric, momentum-conserving)
    this._relax(dt);

    // 6. resolve collisions (position projection)
    this._collide(colliders, dt);

    // 7. derive velocities (clamped — PBF can spike on spawn overlap / deep penetration)
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
    // NaN quarantine: one non-finite position must never persist. NaN compares
    // false in every relational branch below/above (pairs r2<h2, speed clamp,
    // bounds leak check), so an infected particle silently opts out of ALL
    // physics forever, poisoning kineticEnergy and any renderer reading pos.
    // One cheap well-predicted scan every step; the removal loop only runs if
    // the scan actually found something. Backward iteration is required:
    // removeParticle() swaps the LAST element into slot i, and backward order
    // guarantees that element was already checked.
    {
      let badPos = false;
      const m3 = this.count * 3;
      for (let k = 0; k < m3; k++) {
        if (!Number.isFinite(p[k])) { badPos = true; break; }
      }
      if (badPos) {
        for (let i = this.count - 1; i >= 0; i--) {
          if (!Number.isFinite(p[i * 3]) || !Number.isFinite(p[i * 3 + 1]) ||
              !Number.isFinite(p[i * 3 + 2])) {
            this.removeParticle(i);
          }
        }
      }
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

  // ---- ONE shared pair-list build ----
  // Walks each particle's own cell (pairs j>i only) + the 13 half-space
  // neighbor cells (all j). Every unordered pair within h lands in the list
  // exactly once. Simultaneously scatter-adds rho/rhoNear (Clavet kernels)
  // and fills nCount — so viscosity AND relaxation never touch the grid.
  // Traversal strategy is selected once (see densityScatter param):
  //   'runs'   — counting-sorted contiguous per-cell slices (default; kills
  //              next[] pointer-chasing)
  //   'direct' — legacy linked-list traversal (bench fallback)
  //   'pair'   — legacy traversal + deferred pairC[] density append
  // Measured out: per-grid-cell density ACCUMULATORS ('cell') — slower than
  // legacy even before being ruled out physically (a cell total handed to
  // every member gives particles density from non-neighbors), and deferred
  // pairC[] scatter (+54% pairs ms @26k): the scattered rho[j] writes were
  // never the bottleneck; list traversal was.
  _buildPairs() {
    if (this.densityScatter === 'pair') this._buildPairsDeferred();
    else if (this.densityScatter === 'direct') this._buildPairsDirect();
    else if (this.densityScatter === 'probe-none') this._buildPairsRuns(true);
    else this._buildPairsRuns(false);
  }

  // Shared pair walk over CONTIGUOUS PER-CELL RUNS (default). Same pair set
  // and immediate density scatter as the legacy path; only candidate
  // enumeration differs: each cell's particle ids come from a contiguous
  // slice of _sorted instead of a next[] linked list. Pair ORDER within the
  // list differs from legacy (summation-order float drift only).
  //
  // Run construction (three sequential O(n) passes, dirty-cell bookkeeping):
  //   1. count particles per cell (_cellCnt), stamp-deduped into _cellDirty
  //   2. prefix-sum occupied cells -> run base offsets (_cellStart)
  //   3. second particle pass scatters ids into _sorted (start[] as cursor;
  //      cnt[] is preserved so [start[c], start[c]+cnt[c]) is the run)
  _buildPairsRuns(skipDensity = false) {
    const p = this.pos;
    const n = this.count;
    const h = this.h, h2 = h * h;
    const inv = 1 / this.cellSize;
    const D = this.gridDim, D2 = D * D;

    let cap = this._pairCap;
    let pe = this._pairEnds, pd = this._pairData;
    let m = 0, m4 = 0;

    const rho = this.rho, rhoNear = this.rhoNear, nc = this.nCount;
    rho.fill(0, 0, n); rhoNear.fill(0, 0, n); nc.fill(0, 0, n);

    // ---- build per-cell runs ----
    let stamp = this._stamp;
    if (stamp >= 2147483647) { this._cellStamp.fill(0); stamp = 0; }
    this._stamp = ++stamp;
    const cellStamp = this._cellStamp, cellCnt = this._cellCnt, cellStart = this._cellStart;
    let dirty = this._cellDirty;
    let ndirty = 0;
    let sorted = this._sorted;
    if (sorted.length < n) sorted = this._sorted = new Int32Array(Math.max(n, sorted.length << 1));

    // pass 1: count per cell
    for (let i = 0; i < n; i++) {
      const c = (Math.floor(p[i * 3] * inv) & (D - 1)) +
                (Math.floor(p[i * 3 + 1] * inv) & (D - 1)) * D +
                (Math.floor(p[i * 3 + 2] * inv) & (D - 1)) * D2;
      if (cellStamp[c] !== stamp) {
        cellStamp[c] = stamp;
        cellCnt[c] = 0;
        if (ndirty >= dirty.length) dirty = this._cellDirty = _grow(dirty, new Int32Array(dirty.length << 1));
        dirty[ndirty++] = c;
      }
      cellCnt[c]++;
    }
    // pass 2: prefix sums over occupied cells only
    let acc = 0;
    for (let d = 0; d < ndirty; d++) {
      const c = dirty[d];
      cellStart[c] = acc;
      acc += cellCnt[c];
    }
    // pass 3: scatter ids (start[] becomes the cursor; cnt[] keeps run length)
    for (let i = 0; i < n; i++) {
      const c = (Math.floor(p[i * 3] * inv) & (D - 1)) +
                (Math.floor(p[i * 3 + 1] * inv) & (D - 1)) * D +
                (Math.floor(p[i * 3 + 2] * inv) & (D - 1)) * D2;
      sorted[cellStart[c]++] = i;
    }
    // rewind cursors back to run bases
    for (let d = 0; d < ndirty; d++) {
      const c = dirty[d];
      cellStart[c] -= cellCnt[c];
    }

    // ---- pair walk over runs ----
    for (let i = 0; i < n; i++) {
      const xi = p[i * 3], yi = p[i * 3 + 1], zi = p[i * 3 + 2];
      const cx = Math.floor(xi * inv) & (D - 1);
      const cy = Math.floor(yi * inv) & (D - 1);
      const cz = Math.floor(zi * inv) & (D - 1);

      // -- own cell: strictly j > i --
      {
        const cb = cx + cy * D + cz * D2;
        const s = cellStart[cb], e = s + cellCnt[cb];
        for (let k = s; k < e; k++) {
          const j = sorted[k];
          if (j <= i) continue;
          const dx = p[j * 3] - xi, dy = p[j * 3 + 1] - yi, dz = p[j * 3 + 2] - zi;
          const r2 = dx * dx + dy * dy + dz * dz;
          if (r2 >= h2 || r2 <= 1e-12) continue;
          if (m >= cap) {
            cap <<= 1;
            this._pairCap = cap;
            pe = this._pairEnds = _grow(pe, new Int32Array(cap * 2));
            pd = this._pairData = _grow(pd, new Float32Array(cap * PAIR_STRIDE));
          }
          const r = Math.sqrt(r2), invR = 1 / r, q = 1 - r / h;
          pe[m * 2] = i; pe[m * 2 + 1] = j;
          pd[m4] = q;
          pd[m4 + 1] = dx * invR; pd[m4 + 2] = dy * invR; pd[m4 + 3] = dz * invR;
          m++; m4 += PAIR_STRIDE;
          const q2 = q * q;
          if (!skipDensity) {
            rho[i] += q2; rho[j] += q2;
            rhoNear[i] += q2 * q; rhoNear[j] += q2 * q;
            nc[i]++; nc[j]++;
          }
        }
      }

      // -- 13 half-space neighbor cells: ALL j --
      for (let o = 0; o < OFFS.length; o += 3) {
        const ci = ((cx + OFFS[o]) & (D - 1)) + ((cy + OFFS[o + 1]) & (D - 1)) * D + ((cz + OFFS[o + 2]) & (D - 1)) * D2;
        const s = cellStart[ci], e = s + cellCnt[ci];
        for (let k = s; k < e; k++) {
          const j = sorted[k];
          const dx = p[j * 3] - xi, dy = p[j * 3 + 1] - yi, dz = p[j * 3 + 2] - zi;
          const r2 = dx * dx + dy * dy + dz * dz;
          if (r2 >= h2 || r2 <= 1e-12) continue;
          if (m >= cap) {
            cap <<= 1;
            this._pairCap = cap;
            pe = this._pairEnds = _grow(pe, new Int32Array(cap * 2));
            pd = this._pairData = _grow(pd, new Float32Array(cap * PAIR_STRIDE));
          }
          const r = Math.sqrt(r2), invR = 1 / r, q = 1 - r / h;
          pe[m * 2] = i; pe[m * 2 + 1] = j;
          pd[m4] = q;
          pd[m4 + 1] = dx * invR; pd[m4 + 2] = dy * invR; pd[m4 + 3] = dz * invR;
          m++; m4 += PAIR_STRIDE;
          const q2 = q * q;
          if (!skipDensity) {
            rho[i] += q2; rho[j] += q2;
            rhoNear[i] += q2 * q; rhoNear[j] += q2 * q;
            nc[i]++; nc[j]++;
          }
        }
      }
    }
    this._npairs = m;
  }

  // Bench variant 'pair': append per-pair contributions sequentially during
  // the walk (rho[i] stays inline/local), then ONE scattered pass over the
  // pair list reads pairC[] sequentially but still writes rho[j] randomly.
  _buildPairsDeferred() {
    const p = this.pos;
    const n = this.count;
    const h = this.h, h2 = h * h;
    const head = this.cellHead, next = this.next;
    const inv = 1 / this.cellSize;
    const D = this.gridDim, D2 = D * D;

    let cap = this._pairCap;
    let pe = this._pairEnds, pd = this._pairData;
    let pc = this._pairC, pcn = this._pairCN;
    let m = 0, m4 = 0;

    const rho = this.rho, rhoNear = this.rhoNear, nc = this.nCount;
    rho.fill(0, 0, n); rhoNear.fill(0, 0, n); nc.fill(0, 0, n);

    for (let i = 0; i < n; i++) {
      const xi = p[i * 3], yi = p[i * 3 + 1], zi = p[i * 3 + 2];
      const cx = Math.floor(xi * inv) & (D - 1);
      const cy = Math.floor(yi * inv) & (D - 1);
      const cz = Math.floor(zi * inv) & (D - 1);

      // -- own cell: strictly j > i --
      {
        let j = head[cx + cy * D + cz * D2];
        while (j !== -1) {
          if (j > i) {
            const dx = p[j * 3] - xi, dy = p[j * 3 + 1] - yi, dz = p[j * 3 + 2] - zi;
            const r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < h2 && r2 > 1e-12) {
              if (m >= cap) {
                cap <<= 1;
                this._pairCap = cap;
                pe = this._pairEnds = _grow(pe, new Int32Array(cap * 2));
                pd = this._pairData = _grow(pd, new Float32Array(cap * PAIR_STRIDE));
                pc = this._pairC = _grow(pc, new Float32Array(cap));
                pcn = this._pairCN = _grow(pcn, new Float32Array(cap));
              }
              const r = Math.sqrt(r2), invR = 1 / r, q = 1 - r / h;
              const q2 = q * q, q3 = q2 * q;
              pe[m * 2] = i; pe[m * 2 + 1] = j;
              pd[m4] = q;
              pd[m4 + 1] = dx * invR; pd[m4 + 2] = dy * invR; pd[m4 + 3] = dz * invR;
              rho[i] += q2; rhoNear[i] += q3; nc[i]++;
              pc[m] = q2; pcn[m] = q3;
              m++; m4 += PAIR_STRIDE;
            }
          }
          j = next[j];
        }
      }

      // -- 13 half-space neighbor cells: ALL j --
      for (let o = 0; o < OFFS.length; o += 3) {
        const cellX = (cx + OFFS[o]) & (D - 1);
        const cellY = (cy + OFFS[o + 1]) & (D - 1);
        const cellZ = (cz + OFFS[o + 2]) & (D - 1);
        let j = head[cellX + cellY * D + cellZ * D2];
        while (j !== -1) {
          const dx = p[j * 3] - xi, dy = p[j * 3 + 1] - yi, dz = p[j * 3 + 2] - zi;
          const r2 = dx * dx + dy * dy + dz * dz;
          if (r2 < h2 && r2 > 1e-12) {
            if (m >= cap) {
              cap <<= 1;
              this._pairCap = cap;
              pe = this._pairEnds = _grow(pe, new Int32Array(cap * 2));
              pd = this._pairData = _grow(pd, new Float32Array(cap * PAIR_STRIDE));
              pc = this._pairC = _grow(pc, new Float32Array(cap));
              pcn = this._pairCN = _grow(pcn, new Float32Array(cap));
            }
            const r = Math.sqrt(r2), invR = 1 / r, q = 1 - r / h;
            pe[m * 2] = i; pe[m * 2 + 1] = j;
            pd[m4] = q;
            pd[m4 + 1] = dx * invR; pd[m4 + 2] = dy * invR; pd[m4 + 3] = dz * invR;
            m++; m4 += PAIR_STRIDE;
            const q2 = q * q, q3 = q2 * q;
            rho[i] += q2; rhoNear[i] += q3; nc[i]++;
            pc[m] = q2; pcn[m] = q3;
          }
          j = next[j];
        }
      }
    }

    // deferred scatter: sequential pairC reads, random rho[j] writes
    for (let t = 0, t2 = 0; t < m; t++, t2 += 2) {
      const j = pe[t2 + 1];
      rho[j] += pc[t]; rhoNear[j] += pcn[t]; nc[j]++;
    }
    this._npairs = m;
  }

  // Legacy 'direct' variant (pre-Lane-C verbatim): immediate per-pair
  // rho[j]/rhoNear[j]/nc[j] scatter. Kept as bench fallback.
  _buildPairsDirect() {
    const p = this.pos;
    const n = this.count;
    const h = this.h, h2 = h * h;
    const head = this.cellHead, next = this.next;
    const inv = 1 / this.cellSize;
    const D = this.gridDim, D2 = D * D;

    let cap = this._pairCap;
    let pe = this._pairEnds, pd = this._pairData;
    let m = 0, m4 = 0;

    const rho = this.rho, rhoNear = this.rhoNear, nc = this.nCount;
    rho.fill(0, 0, n); rhoNear.fill(0, 0, n); nc.fill(0, 0, n);

    for (let i = 0; i < n; i++) {
      const xi = p[i * 3], yi = p[i * 3 + 1], zi = p[i * 3 + 2];
      const cx = Math.floor(xi * inv) & (D - 1);
      const cy = Math.floor(yi * inv) & (D - 1);
      const cz = Math.floor(zi * inv) & (D - 1);

      // -- own cell: strictly j > i --
      {
        let j = head[cx + cy * D + cz * D2];
        while (j !== -1) {
          if (j > i) {
            const dx = p[j * 3] - xi, dy = p[j * 3 + 1] - yi, dz = p[j * 3 + 2] - zi;
            const r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < h2 && r2 > 1e-12) {
              if (m >= cap) {
                cap <<= 1;
                this._pairCap = cap;
                pe = this._pairEnds = _grow(pe, new Int32Array(cap * 2));
                pd = this._pairData = _grow(pd, new Float32Array(cap * PAIR_STRIDE));
              }
              const r = Math.sqrt(r2), invR = 1 / r, q = 1 - r / h;
              pe[m * 2] = i; pe[m * 2 + 1] = j;
              pd[m4] = q;
              pd[m4 + 1] = dx * invR; pd[m4 + 2] = dy * invR; pd[m4 + 3] = dz * invR;
              m++; m4 += PAIR_STRIDE;
              const q2 = q * q;
              rho[i] += q2; rho[j] += q2;
              rhoNear[i] += q2 * q; rhoNear[j] += q2 * q;
              nc[i]++; nc[j]++;
            }
          }
          j = next[j];
        }
      }

      // -- 13 half-space neighbor cells: ALL j --
      for (let o = 0; o < OFFS.length; o += 3) {
        const cellX = (cx + OFFS[o]) & (D - 1);
        const cellY = (cy + OFFS[o + 1]) & (D - 1);
        const cellZ = (cz + OFFS[o + 2]) & (D - 1);
        let j = head[cellX + cellY * D + cellZ * D2];
        while (j !== -1) {
          const dx = p[j * 3] - xi, dy = p[j * 3 + 1] - yi, dz = p[j * 3 + 2] - zi;
          const r2 = dx * dx + dy * dy + dz * dz;
          if (r2 < h2 && r2 > 1e-12) {
            if (m >= cap) {
              cap <<= 1;
              this._pairCap = cap;
              pe = this._pairEnds = _grow(pe, new Int32Array(cap * 2));
              pd = this._pairData = _grow(pd, new Float32Array(cap * PAIR_STRIDE));
            }
            const r = Math.sqrt(r2), invR = 1 / r, q = 1 - r / h;
            pe[m * 2] = i; pe[m * 2 + 1] = j;
            pd[m4] = q;
            pd[m4 + 1] = dx * invR; pd[m4 + 2] = dy * invR; pd[m4 + 3] = dz * invR;
            m++; m4 += PAIR_STRIDE;
            const q2 = q * q;
            rho[i] += q2; rho[j] += q2;
            rhoNear[i] += q2 * q; rhoNear[j] += q2 * q;
            nc[i]++; nc[j]++;
          }
          j = next[j];
        }
      }
    }
    this._npairs = m;
  }

  // viscosity impulses, straight off the shared pair list (contiguous indexed
  // loops instead of pointer-chasing next[] links)
  _viscosity(dt) {
    const v = this.vel;
    const sigma = this.p.viscositySigma, beta = this.p.viscosityBeta;
    const pe = this._pairEnds, pd = this._pairData;
    const m = this._npairs;
    for (let t = 0, t4 = 0; t < m; t++, t4 += PAIR_STRIDE) {
      const t2 = t * 2;
      const i = pe[t2], j = pe[t2 + 1];
      const i3 = i * 3, j3 = j * 3;
      const ux = pd[t4 + 1], uy = pd[t4 + 2], uz = pd[t4 + 3];
      const u = (v[i3] - v[j3]) * ux + (v[i3 + 1] - v[j3 + 1]) * uy + (v[i3 + 2] - v[j3 + 2]) * uz;
      if (u > 0) {
        const q = pd[t4]; // = 1 - r/h
        let I = dt * q * (sigma * u + beta * u * u) * 0.5;
        // stability guard: the impulse must never exceed the relative
        // velocity it damps (each side gets I, total 2I). Without this
        // the quadratic term overshoots at speed and REVERSES u —
        // explicit-viscosity oscillation that keeps pools churning.
        const Icap = u * 0.45;
        if (I > Icap) I = Icap;
        const Ix = I * ux, Iy = I * uy, Iz = I * uz;
        v[i3] -= Ix; v[i3 + 1] -= Iy; v[i3 + 2] -= Iz;
        v[j3] += Ix; v[j3 + 1] += Iy; v[j3 + 2] += Iz;
      }
    }
  }

  // double-density relaxation, pass 1 (per-particle pressures incl. negative-
  // pressure clamp) + pass 2 (symmetric pair displacements, momentum-conserving)
  _relax(dt) {
    const p = this.pos;
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
      P[i] = Math.max(0, k * (rho[i] - rho0));
      PN[i] = kNear * rhoNear[i];
    }

    const pe = this._pairEnds, pd = this._pairData;
    const m = this._npairs;
    // packing trims (both default OFF → this branch is never taken and the
    // loop below is arithmetically identical to the pre-trim solver)
    const packCap = this.p.packMaxDisp;
    const packDamp = this.p.packDamp;
    const trimPairs = packDamp > 0 || packCap < Infinity;
    // rest pressure level k·ρ0: a pair whose AVERAGE density pressure exceeds
    // this has local ρ > 2ρ0 — deep-pool ram-packing, exactly what we trim
    const pressThresh = k * rho0;
    for (let t = 0, t4 = 0; t < m; t++, t4 += PAIR_STRIDE) {
      const t2 = t * 2;
      const i = pe[t2], j = pe[t2 + 1];
      // averaged pair pressure — symmetric by construction; ±half displacement
      // to each endpoint conserves momentum exactly
      const Pav = 0.5 * (P[i] + P[j]);
      const PNav = 0.5 * (PN[i] + PN[j]);
      const q = pd[t4];
      let Dmag = dt2 * (Pav * q + PNav * q * q);
      if (trimPairs) {
        if (packDamp > 0 && Pav > pressThresh) Dmag *= 1 - packDamp;
        if (Dmag > packCap) Dmag = packCap;
      }
      const ux = pd[t4 + 1], uy = pd[t4 + 2], uz = pd[t4 + 3];
      const Dx = Dmag * ux * 0.5, Dy = Dmag * uy * 0.5, Dz = Dmag * uz * 0.5;
      const i3 = i * 3, j3 = j * 3;
      p[j3] += Dx; p[j3 + 1] += Dy; p[j3 + 2] += Dz;
      p[i3] -= Dx; p[i3 + 1] -= Dy; p[i3 + 2] -= Dz;
    }
  }

  _collide(colliders, dt = 1 / 60) {
    if (!colliders) return;
    const p = this.pos, v = this.vel, pr = this.prev;
    const r = 0.09; // particle radius for contact offset
    for (let i = 0; i < this.count; i++) {
      let px = p[i * 3], py = p[i * 3 + 1], pz = p[i * 3 + 2];
      // remember incoming velocity so any collider response below can be made
      // permanent: step()'s derive pass recomputes v = (p - prev)/dt, which
      // used to silently DISCARD friction/restitution edits made here.
      // Shifting prev by -Δv*dt keeps the response in the derived velocity.
      const v0x = v[i * 3], v0y = v[i * 3 + 1], v0z = v[i * 3 + 2];
      for (const c of colliders) {
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
          // remember this is a riverbed contact + its gradient for bedFriction /
          // slopeAssist below (plane/box contacts keep plain contactFriction)
          isBed = true; gsx = sx; gsz = sz;
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
          // heightfield contacts use the separate bedFriction so creek beds can
          // be slippery while walls/props stay grippy; default equals
          // contactFriction so existing scenes are bit-identical
          const cf = isBed ? this.p.bedFriction : this.p.contactFriction;
          let nt = tSp * (1 - cf) - cf * Math.abs(this.p.gravity) * dt;
          if (nt < 0) nt = 0;
          const sc = nt / tSp;
          tx *= sc; ty *= sc; tz *= sc;
        }
        let vx = nDotV * nx + tx;
        let vz = nDotV * nz + tz;
        // slope assist: extra push along the downhill (-gradient) direction of
        // the riverbed, magnitude slopeAssist*|gravity|, ONLY on heightfield
        // contacts with a nonzero gradient. Keeps gentle creeks lively when
        // gravity's downslope share alone loses to the Coulomb stop term.
        if (isBed && this.p.slopeAssist > 0) {
          const gradMag = Math.hypot(gsx, gsz);
          if (gradMag > 1e-6) {
            const a = this.p.slopeAssist * Math.abs(this.p.gravity) * dt;
            vx += a * (-gsx / gradMag);
            vz += a * (-gsz / gradMag);
          }
        }
        v[i * 3] = vx;
        v[i * 3 + 1] = nDotV * ny + ty;
        v[i * 3 + 2] = vz;
      }
      p[i * 3] = px; p[i * 3 + 1] = py; p[i * 3 + 2] = pz;
      // persist any collider velocity response (friction/restitution) through
      // the derive pass — without this it was discarded every step
      const dvx = v[i * 3] - v0x, dvy = v[i * 3 + 1] - v0y, dvz = v[i * 3 + 2] - v0z;
      if (dvx !== 0 || dvy !== 0 || dvz !== 0) {
        pr[i * 3] -= dvx * dt; pr[i * 3 + 1] -= dvy * dt; pr[i * 3 + 2] -= dvz * dt;
      }
    }
  }

  get cellSize() { return this.p.h; }
  get h() { return this.p.h; }
  get h2() { return this.p.h * this.p.h; }
}
