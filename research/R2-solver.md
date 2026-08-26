# R2 — Water Solver Rewrite: What Shipped + Remaining Optimization Backlog

Batch R2 deliverable for `src/water-pack/`. Covers the shared-pair-list solver
rewrite that landed in `src/water-pack/solver.js` and the prioritized list of
what we would do next to make typed-array JS go faster.

All benchmark numbers below come from `src/water-pack/solver-bench.mjs`
(`node src/water-pack/solver-bench.mjs`) on the dev machine (M-series Mac,
Node). The bench embeds the pre-rewrite solver verbatim as `OldWaterSim`
(copied from `bbafd61:src/water-pack/solver.js`, only renamed) so both sides of
the A/B are literally the same code except the algorithm under test.

---

## 1. What was implemented in this batch

### 1.1 The problem with the old structure

The old solver walked the 27-cell neighborhood **twice per step** — once in
`_viscosity()` and once in `_relax()` — each time re-deriving pairs from the
grid via `cellHead`/`next[]` linked-list chasing. Worse, `_relax` stored a
**per-particle neighbor list truncated at `maxPairs = 64`**: deep-pool
particles have 40+ neighbors within h, and when the cap bit, an *arbitrary*
subset of each particle's neighbors was silently dropped every step. Because
each particle then pushed against a different neighbor set than its partner,
pressure kicks became asymmetric — momentum noise that never cancelled, which
is what made deep pools writhe forever ("water creature"). The truncation also
cost memory up front (`cap × maxPairs` slots preallocated whether used or not).

### 1.2 The new architecture: one shared pair list

`WaterSim.step(dt, colliders)` now runs:

1. **Predict** — gravity + `p += v·dt`, saving `prev`.
2. **`_buildGrid()`** — unchanged flat dense grid (`128³` cells, wrap via
   `& (D-1)` mask), `head.fill(-1)` + push-front insert. No Map, no GC.
3. **`_buildPairs()`** — ONE grid walk builds a single flat unordered pair
   list using the **half-open 13-cell scheme**: for particle i, its own cell
   contributes only pairs with `j > i`; each of the 13 half-space neighbor
   offsets (`OFFS` table: 6 face + 12 edge/corner directions, exactly one of
   each ±pair) contributes all `j`. Every unordered pair within h is visited
   **exactly once**, no dedup pass, no symmetric double-count.
   During the same walk it:
   - scatter-adds Clavet kernels into `rho[i]/rho[j] += q²` and
     `rhoNear[i]/rhoNear[j] += q³` (q = 1 − r/h),
   - counts `nCount[i]` (foam signal),
   - caches pair geometry: `pairI/pairJ` indices plus packed
     `q`, `r`, and unit normal `(nx, ny, nz)` — so downstream passes never
     recompute a sqrt or a distance.
   Buffers grow by doubling (`_grow` copies old contents) and are reused;
   steady-state `step()` performs zero allocations. At 21.6k particles the
   list holds ~438k pairs inside a 2^18-slot buffer (~7× headroom).
4. **`_viscosity(dt)`** — contiguous loop over the shared pair list. Same
   Clavet impulse `I = dt·q·(σu + βu²)·½` with the anti-overshoot clamp
   `I ≤ 0.45·u` (prevents explicit-viscosity oscillation), applied
   symmetrically ±I along the cached normal.
5. **`_relax(dt)`** — classic 3-pass double-density relaxation off the same
   list: (a) densities already scattered during build; (b) per-particle
   pressures `P = max(0, k(ρ−ρ0))`, `P_near = kNear·ρ_near` (negative-pressure
   clamp kills the mitosis artifact); (c) per-pair displacement using the
   **averaged** pressures `½(P_i+P_j)`, split ±half to each endpoint —
   symmetric by construction, conserves momentum exactly. This replaces the
   old accumulate-on-i scheme where j-particles received pushes from lists
   that didn't agree.
6. Colliders → velocity derivation (clamped) → diagnostics, all unchanged.

### 1.3 Why `maxPairs` truncation is gone (correctness, not just speed)

The pair list has no per-particle cap at all — every qualifying pair enters
exactly once and both passes see identical geometry. Asymmetric kicks vanish
structurally rather than being tuned away. Memory went from
`cap × 64 × 5` preallocated slots to ~438k actual pairs × 7 arrays, grown on
demand — smaller in practice *and* unbounded-safe.

### 1.4 Bench methodology

- Scenario: settled-ish 60×24×15 lattice = **21,600 particles** spawned at
  0.55h spacing over an infinite plane floor, `h = 0.5`, `dt = 1/60`.
- 5 warmup steps (JIT), then 60 timed steps; report median/mean/min/max of
  `simMs`, plus kinetic energy trace, leak count, NaN check on positions,
  pairs/step, and a PASS gate at ≥1.6x median speedup.
- Both solvers run identical spawn/scenario; old algorithm is verbatim source
  from commit `bbafd61`.

### 1.5 Numbers

| metric | OldWaterSim | WaterSim |
|---|---|---|
| median ms/step | 88.45 | 44.64 |
| mean ms/step | 83.97 | 45.01 |
| min / max | 52.74 / 134.80 | 26.96 / 64.60 |
| KE last step | 356,443 | 329,970 |

Speedup this run: **1.98x median, 1.87x mean**. An earlier run of the same
script measured 1.74x median — treat **~1.7–2.0x as the honest range**; single
run-to-run spread on this machine is roughly ±0.2x, so quote medians, not best
cases. Positions finite after 60 steps on both sides; 0 leaks; ~438k pairs/step
against a 524,288-slot buffer.

---

## 2. Remaining optimization backlog

Prioritized by expected gain-per-effort for typed-array JS. Estimates marked
as such — none are measured yet.

### (a) Counting sort → contiguous cell sweeps — **do first**

**Problem:** `_buildPairs` still walks `next[]` linked lists inside each cell.
That's pointer chasing through a scattered Int32Array — cache-hostile, and it
defeats the hardware prefetcher precisely in the hottest loop.

**Fix:** replace head/next insertion with a two-pass **counting sort by cell**
(Green, GDC 2010, §"uniform grid" — sort indices by cell id, keep
`cellStart[]` prefix sums):

```
// pass 0: count particles per cell into count[cell]
for i: count[cellOf(i)]++
// exclusive-scan count -> cellStart[]
// pass 1: scatter particle indices in sorted order
pos-in-cell cursor per cell: sorted[idx++] = i
// sweep: for each occupied cell c, particles are
// sorted[cellStart[c] .. cellStart[c+1]) — CONTIGUOUS
```

The pair build then iterates dense index ranges instead of following `next[]`,
and the own-cell/13-offset logic stays byte-for-byte the same otherwise. Bonus:
sorted order also makes the density scatter-adds land near their neighbors in
memory.

- **Effort:** small-medium — one afternoon; touches `_buildGrid` +
  `_buildPairs` iteration only, no API change.
- **Expected gain (estimate):** 20–40% off the pair-build phase. Pair build is
  the largest single phase, but viscosity/relax loops over the existing list
  don't benefit, so overall maybe 10–25% end-to-end. Estimate, not measured.
- **Risk:** low. Pure data-layout change; bench gates catch regressions.

### (b) WASM + SIMD port of the pair loops

**Honest assessment:** realistic **2–4x** over the JS hot loops, not the 10x+
marketing numbers you see for float-heavy C vs. naive JS — modern JITs already
keep typed-array scalar loops within ~2–3x of scalar C, and the win here comes
mostly from SIMD width (4-wide f32) plus predictable aliasing.

- **What ports well:** `_buildPairs` inner distance/kernel math,
  `_viscosity` and `_relax` pair sweeps — straight-line f32 arithmetic over
  SoA buffers, no allocation, no branching except `r2 < h2` masks
  (compiles to masked moves).
- **Toolchain:** C or Rust → `-O3 -msimd128` (clang) emits wasm SIMD
  intrinsics-free auto-vectorization; Rust nightly `target_feature=+simd128`.
  Load via a single ~50–100 KB module, share the same Float32Array backing
  store (`WebAssembly.Memory` + copy-in/copy-out per step, or pass views).
- **Cost:** new build step (wasm toolchain in CI), FFI boundary design,
  harder debugging, dual-source-of-truth risk while JS fallback exists.
  Realistically 2–4 days including bench integration.
- **When it's worth it:** after (a) lands and the profile shows the pair loops
  still dominate. If (a) leaves the sim comfortably under frame budget at
  target particle counts, skip this entirely.
- **Expected gain (estimate):** 2–4x on the ported phases; end-to-end depends
  on phase share — likely 1.5–3x total. Estimate.

### (c) Sleep/wake for settled particles

**Idea:** particles below a velocity threshold stop participating in pair
solves until disturbed. Standard hysteresis scheme:

- sleep when `|v| < v_min` for N consecutive frames AND neighbor-count stable;
- wake any particle whose neighbor moved more than ε, or on collider contact,
  drain-region entry, or spawn overlap.

**Why tempting:** in fill-and-settle scenes most of the pool is asleep within
seconds — potential large win scales with how static the scene is.

**Risks — why it's ranked third despite the headline number:**
- Pour scenes (our primary showcase) keep the whole column agitated; wake
  propagation through 20+ neighbor hops can thrash the sleep set per frame.
- Sleeping particles must still be *collided* and must still contribute rho to
  awake neighbors, or the free surface sags wrong. Getting the asymmetric
  interaction right reintroduces exactly the kind of asymmetry bugs the
  maxPairs removal just fixed.
- Correctness risk > perf risk. Only attempt behind a flag with visual A/B.

**Effort:** medium-high. **Gain (estimate):** scene-dependent, 0x (pours) to
3–5x (settled fills). Estimate.

### (d) Math micro-opts

Small wins, cheap, safe to batch anytime:

- **Avoid sqrt where q² suffices:** `_relax` uses `q` and `q²` only — but q is
  computed once in `_buildPairs` and cached, so this is mostly about keeping
  it that way. The remaining sqrts are one per pair in the build (needed for
  the unit normal) — could defer normal computation to viscosity-only pairs,
  since relaxation needs direction only up to sign... actually it needs the
  signed direction; skip this idea. The real micro-win: `Math.hypot` calls in
  `_collide` are slow paths in some engines — replace with manual
  `sqrt(x²+y²+z²)`.
- **Precompute per-frame constants** hoisted out of loops: `invDt`, `dt2`,
  `h2`, `1/h` are already hoisted in the hot paths; audit `_collide` (re-reads
  `this.p.*` per particle — hoist per call) and cache `this.cellSize` getter
  results locally in `_buildGrid`/`_buildPairs` (getter call per access,
  trivially hoistable).
- **SoA→AoS consideration:** pos is already interleaved `[x,y,z]` AoS-style
  which is fine for the collide loop; leave it.
- **Effort:** hours. **Gain (estimate):** 3–8% combined. Do alongside (a).

### Suggested order

1. (a) counting sort — best ratio, de-risks nothing else.
2. Re-profile. If pair loops still dominant and more headroom needed:
3. (d) micro-opts ride along; (c) only if fill-scene perf matters more than
   pour scenes; (b) last, as the big-hammer option with real maintenance cost.

---

## 3. References

1. **Clavet, Beaudoin, Poulin — "Particle-based Viscoelastic Fluid
   Simulation."** *Symposium on Computer Animation (SCA) 2005.*
   The double-density relaxation + pairwise viscosity impulse model this
   solver implements directly (including the q²/q³ kernels and the
   negative-pressure behavior motivating the P ≥ 0 clamp).
2. **Macklin, Müller — "Position Based Fluids."** *ACM Transactions on
   Graphics (SIGGRAPH) 2013.* PBF framing: predict → project constraints →
   derive velocities; source of the maxSpeed clamp rationale and the
   velocity-from-position pattern in `step()`.
3. **Green — "Particle Simulation using CUDA."** *GDC 2010.* Uniform-grid
   neighbor search with sorting/counting-sort by cell id — the basis of
   backlog item (a); also documents the 13-neighborhood half-open pair
   enumeration used in `_buildPairs`.
4. **Teschner et al. — "Collision Detection for Deformable Objects."**
   *Computer Graphics Forum 24(1), 2003.* Survey; background for the collider
   projection types (plane/box/heightfield) and broad-phase spatial hashing
   trade-offs discussed in the backlog.
