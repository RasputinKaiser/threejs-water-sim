# Next-gen metaball surfacing — sparse surface nets, GPU field build, temporal patches

Lane M2 · 2026-08-25 · research-only. Read against `src/water-pack/surface.js` (current MC
path + P1 optimizations), `src/water-pack/solver.js` (grid/hashing conventions),
`src/water-pack/screen-fluid.js` and `src/water-pack/index.js` (mode wiring),
`src/water-pack/{async-sim,sim-worker}.mjs` (off-thread infrastructure), the addon source
`node_modules/three/examples/jsm/objects/MarchingCubes.js`, and
`research/perf-headroom.md` + `research/next-gen-perf.md` (measured baselines).

**Baseline this document plans around:**

| component | cost | source |
|---|---|---|
| metaballs MC `update()` | ~28 ms/frame @ 15k particles, ~64 ms @ 26k (integration-lane report) | task brief, `next-gen-perf.md` baseline table |
| screen-fluid `renderWater()` | ~0.5 ms | `next-gen-perf.md`, measured |
| solver `step()` | 21.7 ms @ 14k → 50.9 ms @ 26k (median, main thread; worker lane moves it off) | `perf-headroom.md` measured |

Everything marked **(est.)** is an estimate. No new benchmarks were run for this doc;
arithmetic below traces to the constants in `surface.js` / `solver.js`.

---

## 0. Where the 28 ms actually goes (why the addon can't be tuned further)

The three.js `MarchingCubes` addon is a **dense cubic-grid** implementation:

- `reset()` zeroes the whole scalar field **and** the whole normal cache:
  `size3` floats + `size3*3` floats per slab (`MarchingCubes.js:893-894`). At the
  auto-layout's target slab resolution ≈56 (`surface.js:216-224`) that is
  56³ × 16 B ≈ **2.8 MB of memset per slab**, every frame.
- `update()` calls `polygonize()` for **every cell in the grid**
  (`MarchingCubes.js:929`) — a 256-case edge-table switch per cell — even though
  only cells near the iso surface emit anything. At k=3 slabs that is
  ~527k cell visits/frame; measured scene output is ~5–15k triangles, i.e.
  **>95% of polygonize work emits nothing**.
- The saturated-cell culling (`surface.js:26-30, 287-305`) already minimizes the
  *field fill* side (`addBall` splats); what remains dominant is exactly the two
  costs above, which scale with **grid volume**, not with water.

P1's slab splitting attacks this by shrinking the cubic grid, but it is fundamentally
bounded: the addon offers no way to skip empty cells. Any next-gen path must make
work proportional to **active cells**, not res³.

### Our exact case, in cells (20k particles, h=0.35)

Constants from `surface.js:83-84` and `solver.js` defaults:

- particle spacing = 0.55h = **0.1925 m**; ball visual diameter dWorld = 2·h·0.55 =
  **0.385 m**; auto-res rule ⇒ grid cell = dWorld/1.5 = **0.2567 m**.
- lone-ball iso radius = 0.55h ≈ 0.1925 m; kernel reaches zero at ≈2.8× that
  (see `surface.js:34-39`) ⇒ field reach **R₀ ≈ 0.54 m ≈ 2.1 cells** around each
  particle.
- 20k particles on the spawn lattice occupy ≈ 20000·0.1925³ ≈ **143 m³** of fluid
  (ram-packed pools compress well below this; spread-out creek scenes exceed the
  AABB but not the fluid volume).
- Active cells (within R₀ of any particle): ≈ fluid volume / cell³ plus the
  R₀ shell ⇒ roughly **10k–30k cells** depending on packing and splashiness *(est.)*.
- Surface-crossing cells: for a representative 300 m² plan-area pool
  (~0.5 m deep ≈ 143 m³), wetted surface ≈ 650 m² ÷ cell-face area (0.066 m²)
  ⇒ **~10k sign-changing cells** ⇒ ~10k quads — same order as today's
  `mcTris/2`, as expected (same iso, same look).

So the useful work is ~1–3% of what the addon does. That ratio **is** the headline
opportunity.

---

## 1. Sparse surface nets (the replacement candidate)

### Why surface nets, not marching cubes or dual contouring

- **Marching cubes** on a sparse set reproduces the addon's 256-case machinery and
  its crack/consistency subtleties, for no gain over surface nets here.
- **Dual contouring** needs per-corner gradient signs *and* feature-aware QEF
  placement to justify itself (sharp features); water has none. Its failure mode
  (degenerate/quashed cells) needs guards that eat the win.
- **Naive surface nets** (Gibson's "density-based" variant; also Mikola Lysenko's
  "culling" writeup) places ≤1 vertex per active cell at the **centroid of the
  cell's iso-edge crossings**, then connects vertices across faces where the
  field changes sign. It is crack-free by construction, needs no tables except a
  12-edge mask, produces smooth water-appropriate surfaces, and its cost is
  exactly proportional to active cells. This is the right fit.

### Algorithm shaped for THIS codebase

**(a) Sparse field built only within R of any particle — no global scan.**

Key realization that keeps this simple: `fitVolume()` already quantizes the fitted
AABB to cell multiples (`surface.js:147-154`), so the fitted volume has **stable
integer cell dimensions** frame to frame. A dense staging field over just the
fitted volume is tiny (a 120×24×48-cell fit is ~138k corners ≈ 550 kB as
Float32) — the memory was never the problem, the *iteration count* was. So:

```
// persistent buffers (reused every frame, zero steady-state alloc)
field   : Float32Array((nx+1)*(ny+1)*(nz+1))   // corner samples, fitted volume
occ     : Uint8Array(nx*ny*nz)                 // active-cell marks
touched : Int32Array(capCells)                 // indices written last frame (for O(active) clear)
active  : Int32Array(capCells)                 // this frame's active cell list
```

Per frame:

1. **Clear** only entries listed in `touched[]` (last frame's active set) —
   O(prev-active), not O(volume). Same trick the solver uses conceptually with
   its per-frame `rho.fill(0, 0, n)` bounds (`solver.js:309`).
2. **Stamp**: for every particle (reuse the existing derived `inputStride`
   logic, `surface.js:249-259`), OR-in occupancy over the 3³ cell block around
   it (R₀ ≈ 2.1 cells ⇒ radius-1 block suffices; radius-2 only for lone spray
   droplets — special-case those, they're rare). Append newly-set cells to
   `active[]`. Cost: 27 stamped writes × ~20k particles ≈ 540k mostly-hitting
   writes ≈ **0.3–0.6 ms (est.)**.
3. **Splat field** additively into corners, but ONLY for active cells: walk
   `active[]`; for each active cell gather contributions from the particles in
   its 3³ particle-cell neighborhood — the solver's flat wrap-hash grid
   (`solver.js:85, 274-289`: `cell = (x & 127) + (y & 127)*128 + (z & 127)*16384`,
   `cellHead`/`next` linked lists, `cellSize = h`) can be queried directly for
   this, or the surface pass keeps its own identical-shape hash at
   `cellSize = baseCell()`. Kernel = the addon's `addBall` formula
   (`strength/(0.000001+d2) - subtract`) verbatim, so the **look is bit-comparable**
   to today's isolation-82 surface. Cost: proportional to surface-layer balls,
   which culling already reduces today — call it the current field-fill budget
   minus waste, **~1–2 ms (est.)**.
   - Cheaper alternative that skips the gather entirely: invert loop 2 — while
     stamping, splat each particle straight into the corners of its 4³ corner
     block, clamped to cells that are or become active. Same asymptotics, one
     pass. Either shape works; benchmark both *(est. parity)*.

**(b) One quad per active-cell face crossing.**

4. **Polygonize** walking `active[]` only: read the cell's 8 corners, build the
   sign mask; skip instantly if 0 or 255 (interior/exterior — this is the >95%
   the addon wasted). Place the vertex at the centroid of the crossed edges
   (edge crossing points via linear interpolation on the corner field — same
   interpolation the addon does, `MarchingCubes.js:225+`). Then, for each of the
   6 faces whose neighbor cell is NOT active-but-signed-opposite (i.e. the face
   has a sign change across its 4 corner pairs), emit ONE quad joining the
   vertices of the 4 cells sharing that face. Because a face's quad needs
   vertices in all 4 adjacent cells, the stamp radius in step 2 must cover the
   diagonal partners — radius-1 stamping already does (all 4 cells share a
   corner with the stamped cell).
5. **Normals**: central differences of `field` at each vertex (3 taps × 6).
   Quality equals the addon's `normal_cache` gradients; no normal cache memset
   needed because normals are computed at emit time only for emitted vertices.
6. **Upload**: one `BufferGeometry`, preallocated position Float32Array +
   Uint32 index, `setDrawRange` + `needsUpdate` — the exact pattern the addon
   already uses (`MarchingCubes.js:939-943`). Budget: ~60k verts / 120k indices
   covers 20k-particle scenes with slack; grow-on-demand like `_grow` in
   `solver.js:99`.

**Slab splitting disappears entirely.** Slabs exist only because the addon forces
one cubic global grid (`surface.js:16-23`); a sparse builder has no global grid, so
no seams, no per-slab budgets ("bite" bug, `surface.js:118-119`), no `poolKey`
rebuild churn, no ball-straddle double-splatting. Deleting `ensurePool`/
`disposePool`/slab-placement code (`surface.js:101-131, 227-239`) removes a real
maintenance burden, not just cost.

### Memory & CPU comparison (20k particles, h=0.35)

| | MC addon (today, k=3 slabs @ res 56) | sparse surface nets |
|---|---|---|
| field storage | 3 × (56³·4 B field + 56³·12 B normals) ≈ **8.5 MB** persistent | 1 fitted-volume field ≈ **0.5 MB** + occ/touched/active ≈ 0.7 MB |
| vertex buffers | 3 × budget(≥108k tri)·36 B ≈ **11 MB** prealloc | 1 × ~60k vert·12 B + idx ≈ **1.2 MB** |
| per-frame memset/reset | 3 × 2.8 MB + 3 × 175k-cell polygonize walk | O(prev-active) clears only |
| cell visits / frame | ~527k (all cells, all slabs) | ~10–30k active (est.) |
| emitted geometry | ~10–15k tris | ~10k quads (same surface) |
| meshes/draw calls | k (≤4) | 1 |
| allocations in steady state | 0 (already good) | 0 |

### Speedup, honestly

The addon's ~28 ms @ 15k decomposes (est., from the structure above — not
measured): field fill post-culling maybe 4–6 ms, reset ~2–3 ms, polygonize walk
~15–20 ms, upload/misc remainder. Surface nets replaces the middle three with
O(active) work and keeps field fill ≈ flat:

- **15k particles: ~28 ms → ~3–6 ms (est., 5–8×).**
- **26k particles: ~64 ms → ~5–10 ms (est.).** Scaling becomes ~linear in
  surface cells rather than in fitted-volume cells; the 26k-vs-15k gap narrows a lot.
- Floor is the field splat + JS overhead; do not expect screen-fluid's 0.5 ms.
  If the result lands above ~8 ms at 26k, combine with §Ladder rung 2
  (move it into the sim worker) and it leaves the render thread completely.

Main risks: (i) look parity — mitigate by keeping the addon kernel/isolation
constants verbatim (§step 3); (ii) centroid vertices slightly "shrink" thin jets
vs MC — usually invisible at cell = diameter/1.5, verify against bucket/pool
scenes; (iii) the face-quad bookkeeping has more edge cases than the addon's
per-cell triangle fan — needs a careful unit test on single-ball, two-ball merge,
and drain-to-zero transitions (headless Node test like `effects-smoke.mjs` would
catch cracks).

Effort: **~2–3 days** incl. tests *(est.)*. This is the highest-payoff CPU path.

---

## 2. GPU field build

Two sub-options, given **r185 + WebGLRenderer** (the whole app — screen-fluid RTs,
`effects.js`, physical materials — assumes WebGLRenderer; `index.js:35` loads lanes
through it):

### 2a. Point-sprite density splat → texture, CPU reads back the shell

Reuse is genuinely tempting: `screen-fluid.js` already renders 20k+ camera-facing
point sprites with a projected-size vertex shader (`POINTS_VS`,
`screen-fluid.js:59-96`) at ~0.5 ms total. Rendering the same sprites into a
**2D-atlas-packed slice stack** (each tile = one z-slice of the fitted volume,
additive blending into a HalfFloat RT) writes density on GPU timescales (<1 ms).

But the pipeline dies at readback:

- Full-field readback of even a modest 138k-corner fit is ~1–2 MB/frame through
  `readRenderTargetPixels` — a synchronous GL stall that **flushes the pipeline**;
  expect 2–5 ms added *and* lost overlap with the worker sim (est.). Every frame.
- "Read back only the shell" presumes you already know which cells are surface —
  which requires the field you haven't read back yet. A GPU reduction pass can
  encode shell-cell IDs into a compact pixel buffer, but that's a second custom
  pass + an even more latency-critical readback, and the CPU still redoes
  polygonize. You keep the worst of both sides.
- WebGL2 has no compute; scatter/additive splat precision at HalfFloat, blending
  guarantees, and atlas packing bugs make this a multi-week project to save a
  field-fill that is already only ~1–2 ms after culling (est.).

**Verdict: don't.** The expensive part (dense polygonize) stays on CPU either way;
GPU-assisting the cheap part buys nothing and adds a sync stall.

### 2b. Full-GPU marching cubes / surface nets via WebGPU compute

The technically clean version: field build + sign scan + vertex compaction
(scan/atomic-append) + index emit as WGSL compute passes; mesh stays GPU-resident;
zero readback. Three r185's WebGPURenderer/TSL can express this (scene nodes exist
for exactly this pattern).

Blocking realities for THIS app:

- `WebGPURenderer` is a **whole-renderer migration** — screen-fluid's RT/blit
  compositor, `effects.js` post chain, and material setups are written against
  WebGLRenderer. That's a project an order of magnitude bigger than the surfacing
  problem, touching every lane.
- Shipping bar: mid-range hardware WebGPU coverage in 2026 is good on Chrome/Edge
  and macOS Safari, still uneven elsewhere (est.); a browser game needs the
  WebGL fallback anyway — so the CPU path must exist regardless.
- The CPU solver remains on the main/worker thread regardless of GPU surfacing;
  GPU MC doesn't touch the actual bottleneck pair-build (perf-headroom.md).

**Verdict: correct long-term architecture, wrong project now.** Revisit only when/
if the app migrates to WebGPURenderer for other reasons.

---

## 3. Mesh caching / temporal patches

Idea: between frames, most cells keep their sign; only re-polygonize regions where
occupancy or the sign mask changed, and splice new quads into a persistent index
buffer ("patch mesh updates").

What it takes to be correct:

- Per-cell state bits (sign mask + occupied) persisted across frames; a cell is
  dirty if its 8 corner signs changed **or** any neighbor's did (quads reference
  4 adjacent cells' vertices ⇒ 1-cell dirty halo, or you get hairline cracks).
- Vertex identity: a surface-nets vertex moves whenever ANY of its 8 corners
  flips, so all quads touching that cell must be re-emitted — patch granularity
  is inherently cell+halo, not per-triangle.
- Index-buffer management: either a per-cell slot free-list (fragmentation,
  complexity) or simply rebuild the index buffer each frame from dirty flags —
  rebuilding 20–30k quad indices is ~0.1 ms (est.), so the *full rebuild* variant
  is right; then temporal caching saves only the per-cell polygonize+centroid
  math, not buffer work.

Payoff depends entirely on **churn** — fraction of surface cells changing per frame:

- Calm settled pool: velocities ~0.1–0.5 m/s ⇒ ~0.02–0.1 cell/frame drift ⇒
  churn plausibly <10% ⇒ big win (est.).
- Pour/splash/creek flow (this game's signature moments!): fronts sweep many
  cells/frame; churn plausibly 40–70% (est.) ⇒ bookkeeping overhead with little
  gain.
- Critically: **§1's sparse builder already reduced per-cell cost ~30×**. Caching
  optimizes the remaining small number. On top of surface nets its ceiling is
  maybe 3–6 ms → 1.5–3 ms in calm scenes (est.) — nice, not transformative. On
  top of the *addon* it would be a large relative win but built on a foundation
  (dense grid walk) you'd be deleting anyway.

Complexity: high (dirty tracking, halo rules, stale-normal flicker, crack bugs
that only show in motion). **Recommendation: instrument churn first** (one counter
in whatever builder ships — trivial), and only build patching if calm-scene churn
measures <20% AND calm-scene surface cost still matters. Do not build speculatively.

---

## 4. The pragmatic ladder

Ranked by (payoff ÷ (effort × risk)) for a browser game on mid-range hardware.

| # | Option | Effort | Risk | Render-thread payoff @15k→26k | Verdict |
|---|--------|--------|------|-------------------------------|---------|
| 1 | **Screen-fluid exclusively** (null option) | 0 (done) | low | 28–64 ms → **0.5 ms** | **Ship default.** Tradeoffs below. |
| 2 | **Keep MC, tuned params + worker offload** | 0.5–1 d | low | 28 ms → ~1 ms main thread (total unchanged, est.) | Cheap insurance while #4 is built; reuses sim-worker/SAB plumbing. |
| 3 | Current MC as automatic fallback | 0 (done) | — | unchanged | Keep forever as the non-screen-mode path. |
| 4 | **Sparse surface nets (§1)** | 2–3 d | medium | **~3–6 ms → ~5–10 ms (est.)**; scales linearly | **Build this** when true 3D metaballs are needed (close-ups, refraction, silhouette). |
| 5 | Temporal patches (§3) on top of #4 | 3–5 d | high | calm scenes only, ~2× on the remaining few ms (est.) | Only after churn instrumentation says so. |
| 6 | GPU field splat + CPU readback (§2a) | 2–3 wks | high | negative-to-neutral (sync stalls) | Skip. |
| 7 | WebGPU compute MC (§2b) | multi-wk + renderer migration | high | large, but out of scope until WebGPURenderer | Park; revisit on renderer migration. |

**Null option tradeoffs (screen-fluid exclusively)** — why it isn't automatically
the end of the story even at 0.5 ms:

- Screen-space surfacing has **no true 3D silhouette**: thin streams/jets can
  break up up close, water seen edge-on thins out, and there is no geometric
  surface for refraction *of the water body itself*, shadow casting, or object
  intersection silhouettes (it composites over the blitted backdrop,
  `screen-fluid.js:19-22`). At game camera distances it looks great — the lane
  reports it does — but it constrains framing.
- Depth-based effects (Beer-Lambert absorption etc.) are view-dependent approximations.
- Metaballs remain the only mode with a real mesh — needed for any future
  "grab the water", physics-visible surface, or offscreen reflection of water.

**Recommended sequence:** ship on #1+#3 (both done), land #2 if the metaballs mode
must stay usable at 15k+ meanwhile, then invest in #4 when a concrete feature
actually needs a true mesh. Gate #5 behind churn measurements. Never start #6/#7
without a renderer-level reason.

---

## Appendix: grounded file references

- `surface.js:83-84` — dWorld/baseCell definitions used throughout §0/§1 math.
- `surface.js:101-131, 227-239` — slab pool + placement that surface nets deletes.
- `surface.js:245-258` — ball strength/R₀n/stride derivation to reuse verbatim.
- `surface.js:287-305` — saturated-cell culling (concept carries over to stamping).
- `solver.js:85, 112, 274-289, 574` — wrap-mask flat hash grid, `cellSize = h`;
  the surface-pass cell hash mirrors this shape at `cellSize = baseCell()`.
- `solver.js:99, 304-310` — growable scratch + bounded-clear conventions reused.
- `MarchingCubes.js:893-894, 929, 939-943` — reset/polygonize-all/upload behavior
  underlying §0.
- `screen-fluid.js:59-108, 437-454` — point-sprite pass + half-res HalfFloat RTs
  (the reuse candidate assessed in §2a).
- `sim-worker.mjs:32-38, 86-97` — double-buffered SAB position publishing; the
  vehicle for ladder rung #2 and potentially for running surface nets off-thread.
- `research/perf-headroom.md` — measured solver baselines cited in the header table.
- `research/next-gen-perf.md` — 64 ms @ 26k MC figure; its §"GPU-compute solver"
  reasoning about WebGPU constraints applies equally to §2b here.
