# Next-gen performance architecture — GPU compute solver, LOD water, interest culling, persistence

Lane F1 · 2026-08-25 · research-only. Read against `research/perf-headroom.md` (measured
phase profile), `research/R4-flow-rendering.md` (screen-space pipeline), and the current
source of `src/water-pack/{solver,surface,screen-fluid}.js`, `async-sim.js`, `sim-worker.mjs`,
and `src/scenes/creek.html.js`.

**Baseline this document plans around** (from perf-headroom.md, measured, h=0.35):

| component | cost @ 26k particles | notes |
|---|---|---|
| `surface` MC `update()` | ~64 ms/frame (reported by integration lane) | CPU scalar-field fill + polygonize, three `MarchingCubes`, res ≤ 160 |
| solver `step()` | ~51 ms/step (50.87 median) | `_buildPairs` = 71%, `_viscosity` 17.6%, `_relax` 8.5%, `_collide` 1.2% |
| screen-render pass (`renderWater`) | ~0.5 ms | GPU, half-res RTs |
| foam (`FoamSystem`) | ~1–2 ms | ≤4000 ring-buffered points |

Everything marked **(est.)** is an estimate. No new benchmarks were run for this doc;
all numbers trace to perf-headroom.md or the cited source lines.

---

## 1. GPU-compute solver path

### What would actually be ported

The step is Clavet double-density relaxation over a **shared pair list**
(`solver.js::_buildPairs` builds it once; `_viscosity` and `_relax` consume it). The
GPU-native restructuring differs from the JS shape in one important way:

- **The scattered density write (`rho[i] += q2; rho[j] += q2`) does not survive the
  port.** WebGPU has integer atomics only — no `atomicAdd<f32>`. The standard fix is to
  invert scatter→gather: each thread owns particle *i*, walks *i*'s neighbor list, and
  accumulates its own ρ/ρNear/nCount in registers. This is strictly cheaper than the JS
  version (no half-list bookkeeping, no 13-offset walk duplicated per endpoint) and
  removes the pointer-chasing linked-list grid entirely.
- **Neighbor discovery becomes a counting-sort uniform grid**, not the flat
  128³ `cellHead`/`next` linked list (`GRID_DIM = 128` wraps via mask — that trick also
  doesn't port; sort-based grids need explicit bounds handling). Classic sequence:
  hash cell index → prefix-sum counts → sorted particle index buffer. At 26k particles
  this is 2–3 small dispatches.
- **Neighbor list materialization**: store up to K neighbors/particle (K=32 covers the
  measured worst case of 17.5 pairs/particle at 30k with headroom) in a fixed-stride
  storage buffer. All later passes (density, pressure, relax, viscosity) become
  embarrassingly parallel gathers over these lists — no atomics anywhere.
- **Viscosity symmetry**: the JS code applies ±I to both endpoints of each pair. On GPU,
  each thread recomputes I from its own neighbor list and applies only its own −I. Both
  endpoints compute bit-identical I (same operands, same formula, incl. the `Icap`
  stability clamp at `solver.js:407`) provided the expression is written identically —
  momentum conservation is preserved *exactly*, not approximately. Same trick works for
  the pair displacement in `_relax` (each thread moves itself by −Dmag·n·0.5).
- **Heightfield collider ports trivially and stays GPU-resident.** The creek heightfield
  is 161×97 Float32 ≈ 61 KB (`creek.html.js:19`); upload once as a storage buffer,
  reimplement the bilinear sample + central-difference gradient from
  `solver.js:505–526` in WGSL (~25 lines). `bedFriction`/`slopeAssist`/restitution are
  all closed-form per-particle — direct ports. **This collider never needs to come back
  to the CPU.**

### What stays on the CPU

- **Spawn/drain/recycle.** `spawn()`'s overlap guard walks the hash grid; `drain()` and
  the leak-kill loop swap-remove (`removeParticle`). These mutate `count` and reorder
  arrays — keep them CPU-side, running against last-frame positions (one frame stale is
  invisible for an emitter/drain region), then upload the compacted active range. The
  creek's emit→step→drain loop (`creek.html.js:314–339`) keeps working unchanged.
- **Camera/frustum logic** (§3 below) and all GUI/metrics plumbing.

### three.js r185 reality check

- The repo renders with `WebGLRenderer` end-to-end: `MarchingCubes` addon (surface.js),
  `RawShaderMaterial` GLSL1 pipeline (screen-fluid.js), `PointsMaterial` overlays.
  **You cannot mix `WebGPURenderer` and `WebGLRenderer` in one frame** — adopting TSL
  compute means either porting the entire render stack (weeks, touches every lane's
  files) or running compute-only.
- Compute-only is possible in principle: a bare `WebGPUDevice`-style setup with
  storage buffers + per-frame `mapAsync` readback of positions (26k×12 B = 312 KB/frame —
  bandwidth-trivial) feeding the existing WebGL `BufferAttribute`. But r185 does not
  expose a clean device-without-renderer path; the supported route is `WebGPURenderer`,
  which drags the render-stack question back in. Status as of r185: TSL/WebGPU is
  functional but the addon surface (`three/webgpu`) still shifts between minors; pinning
  matters. **(est.)**
- **WebGL2 transform-feedback fallback:** technically reachable (TF can stream `pos`
  updates into ping-pong VBOs) but it has *no* atomics, *no* random-access buffer
  writes, and *no* compute dispatches — the gather restructure above still has to be
  done, plus a vertex-pass neighbor-list builder, which is the hardest part of the port
  with none of WebGPU's ergonomics. Not recommended: if you pay the algorithmic
  restructuring cost, pay it where it lands on the modern API.

### Expected gains and verdict

- Pair-build+density (36 ms measured) becomes 2–3 tiny dispatches; total GPU step for
  26k particles **(est.) well under 2 ms even on integrated graphics**. But the win at
  26k buys little: the *solver* is only ~51 ms of a ~115 ms frame whose biggest single
  line item is the 64 ms MC surface — which another lane is attacking, and which the
  screen-fluid path already renders in 0.5 ms.
- The real payoff is headroom: **(est.) 100k–200k particles at 60 fps** on a mid-range
  discrete GPU, which nothing else on the roadmap delivers.
- **vs WASM-in-worker** (perf-headroom Priority 2, est. 2–3× → 18–25 ms in-worker):
  for a game shipping at ≤30k particles, WASM+SIMD in the *already-built*
  `sim-worker.mjs` pipeline hits the target at a fraction of the risk. **Verdict: park
  WebGPU compute until (a) the product demands ≥60k particles, and (b) the renderer
  migrates to WebGPU anyway.** Revisit the TSL node-material port of screen-fluid at
  the same time — the two migrations share most of their cost.

---

## 2. LOD water strategy

### Near representation (already exists)

Full PBF sim + surfacing. Two surfacers ship today: MC metaballs (surface.js, CPU-bound)
and the screen-space fluid (screen-fluid.js, ~0.5 ms). LOD policy should pick the
surfacers by screen coverage first (MC for close-ups where its silhouettes beat sprite
depth; screen-fluid otherwise) — that switch is orthogonal to distance LOD below.

### Far representation: spline ribbon

The creek scene already defines the far geometry analytically:
`channelZ(x)`/`channelDz(x)` (`creek.html.js:22–27`), parabolic bed profile, and
`waterSurfaceY(x) = -SLOPE·x - DEPTH·(1-WATER_FILL)` (:95). For a general heightfield,
extract the equivalent:

- **Centerline extraction:** for each grid column xᵢ, thalweg z*(xᵢ) = argmin_z
  `heights[iz·NX+ix]`; smooth with a small box filter (2–3 taps), reject columns whose
  minimum isn't meaningfully below the banks (no channel → gap in the ribbon). Fit a
  `THREE.CatmullRomCurve3` through (xᵢ, bed+z*, z*) samples — the same object the scene
  already conceptually owns analytically. One-time cost: NX=161 argmin scans over NZ=97
  values ≈ negligible, done at load.
- **Ribbon mesh:** sweep a cross-section (width 2·HALF_W, y = local water surface) along
  the curve, ~2 verts every 0.25 m → 320×2 verts for the creek. Material: the flow-wave
  normal perturbation R4 §1b already specifies (world-anchored sines along `uFlowDirW`)
  applied to a translucent `MeshPhysicalMaterial` matching the near water's tint, plus
  scrolling streaks. Total far-water draw cost: **(est.) < 0.2 ms**.

### Seamless handoff

Distance-based crossfade keyed on **distance from camera to the nearest ribbon arc**,
not per-pixel:

1. **Sim-side:** particles beyond `dFar` from the camera are excluded from the sprite
   depth/thickness passes (screen-fluid POINTS_VS gets a per-particle `aActive` or simply
   a tightened draw-range after a CPU partition pass — the position array is already
   compacted by swap-remove, so a stable partition would fight `removeParticle`; prefer
   a shader-side discard driven by a per-particle distance attribute updated on spawn).
   MC is worse at this (its scalar field is global) — another reason to prefer
   screen-fluid as the near surfacer at scale.
2. **Render-side:** the ribbon fades in over a band `dNear..dFar` (~6–10 m, tune live)
   via material opacity; because the ribbon is rendered *into `tScene` before
   `renderWater()` runs*, the near water's refraction automatically shows the ribbon
   underneath it during the overlap — that is what makes the seam soft rather than a
   hard cut. The composite's existing `edge` alpha term handles the final blend at the
   sprite silhouette.
3. **Motion match:** ribbon scroll speed must track the sim's real flow (~0.5 m/s jet,
   R3 §2.2 speeds 0.3–1.5 m/s) or the handoff reads as a speed discontinuity. Cheapest
   honest source: average |vel| of particles in the outermost active band, smoothed,
   fed as the ribbon's scroll uniform.
4. **Physics continuity is NOT preserved across the handoff — by design.** Far water is
   decorative; particles drained at the outflow and re-emitted upstream (the recycle
   loop) mean the far reach doesn't need a correct volume, only a correct *look*. If a
   gameplay interaction (explosion, boulder drop) lands in the far zone, wake it into
   full sim first (§3) and fade the ribbon out locally.

Effort **(est.)**: centerline extractor + ribbon generator 1–1.5 days; crossfade +
speed-match 1 day; per-particle distance attribute 0.5 day. Risk: medium — the failure
mode is a visible "carpet edge" at dNear; mitigate with the tScene-under-refraction
trick and generous overlap band.

---

## 3. Occlusion / interest culling (simulate only what's watched)

### Segmentation along the channel

Partition the sim domain into segments along the centerline (creek: ~4 m arcs → 10
segments; general case: slice the bounds AABB perpendicular to the extracted centerline).
Per segment keep an AABB (bed + bank shoulder + margin) and a state:
`ACTIVE / SLEEPING / FROZEN`.

The solver currently has no notion of subsets — `_buildGrid`/`_buildPairs` iterate
`0..count` unconditionally (`solver.js:278, 309`). The minimal-intrusion design is an
**active-index compaction list**: an Int32Array of active particle ids rebuilt only when
segment states change (not per step), consumed by a range-guard at the top of
`_buildGrid`/`_buildPairs`. Particles in FROZEN segments are skipped entirely — no grid
insert, no pairs, no integrate; their `prev == pos` so the derive pass yields zero
velocity naturally.

### Sleep/wake rules

- **Freeze** a segment when: fully outside frustum expanded by an N-meter margin
  (N ≈ 15 m, i.e., ~2 meander periods) for T consecutive seconds (T ≈ 1 s, so quick
  camera pans don't thrash).
- **Wake** when any of: (a) camera ray/AABB enters margin; (b) a *boundary band* of the
  adjacent ACTIVE segment (particles within h of the shared plane) shows motion — poll
  max |Δx| of that thin band per step, cheap; (c) spawn/drain event intersects the
  segment (emitter at x=−18 and drain at x>16.5 each pin their own segments permanently
  awake in the creek).
- **Never freeze** the segment containing the player-interaction point or an active pour.

### Interaction with the recycling loop

This is the subtle part. The creek loop is emit → step → `drain(drainRegion)` → count
(`creek.html.js:314–339`), and `removeParticle` **swap-deletes from the end**
(`solver.js:174–182`), invalidating index identity every drain tick. Therefore:

- The active-list rebuild must run *after* drain each step, or (cheaper) maintain the
  list lazily: keep a `segmentOf[i]` byte array updated on spawn and on swap-remove
  (when slot i receives particle from `last`, copy its segment byte too — 3 extra lines
  in `removeParticle`). Rebuilds then happen only on segment-state transitions.
- Drain/emitter segments pinned awake means the steady-state creek keeps flowing even
  when unwatched — which is *desired*: mass balance (pour rate vs drain) is what keeps
  the pool level right when the player looks back. Freezing the mid-reach segments
  saves their pair work without breaking the budget.
- **Expected savings (est.):** a 40 m creek viewed from one end has ~60–70% of its
  length outside a 15 m margin → roughly proportional cut in the 71%-dominant pair
  phase, i.e., est. 50 ms → ~20 ms at 26k. Scene-shaped: zero benefit in a single-basin
  pour scene.
- **Failure mode:** frozen-looking water revealed on wake (perf-headroom flags this for
  sleep-skip too). Mitigate with a "low-rate tick" mode: FROZEN segments step at 1 Hz
  with dt×60 clamped by maxSpeed — keeps pools level and currents creeping at ~1% of
  the cost. Strongly recommend shipping tick-mode freeze, not hard freeze.

Effort **(est.)** 2–3 days including the lazy segment tracking and GUI overlay showing
segment states. Risk medium-high (waking correctness, interactions with the async
worker — the segment table lives worker-side, transitions arrive via the existing
command mailbox in `sim-worker.mjs`).

---

## 4. Persistent simulation (IndexedDB)

### Size math (26k particles)

| data | bytes |
|---|---|
| pos (Float32 ×3) | 26,000 × 12 = 312 KB |
| vel (Float32 ×3) | 312 KB |
| nCount (Int32) | 104 KB — *skippable*, recomputed next step |
| header (count, params hash, schema version, timestamp) | ~64 B |

**~624 KB serialized, ~700–800 KB in IndexedDB after structured-clone overhead** — four
orders of magnitude under quota (browsers grant hundreds of MB–GB per origin). Even a
future 100k-particle world is ~2.4 MB. Feasibility is unambiguously yes.

### Design

- **Serialize** `{ v: 1, count, paramsFingerprint, pos: ArrayBuffer, vel: ArrayBuffer }`
  into a single `Blob`, `put()` under one key (`"watersim-current"`). Copying into the
  blob is one memcpy of 624 KB ≈ tens of µs — free.
- **When to write:** debounce ≥5 s AND on `pagehide`/`visibilitychange→hidden`. Never
  per-step: a pouring creek mutates constantly and you'd just burn IO. On restore-miss,
  fall back to today's cold start (spawnBlock or empty).
- **Restore path:** on load, read blob → wrap `pos/vel` in Float32Arrays → `sim.count =
  count; sim.pos.set(...); sim.vel.set(...); sim.prev.set(sim.pos)` (seeding `prev` from
  `pos` gives zero derived velocity on step 1 — the correct cold-start semantics the
  solver already assumes). With the async wrapper, add a `'restore'` message type to
  `sim-worker.mjs` carrying transferable ArrayBuffers — the command mailbox pattern is
  already there.
- **Invalidation:** store a fingerprint of params (h, spacingRatio, restDensity,
  gravity) + a world/collider version. Mismatch → discard. Water saved against an edited
  heightfield will be underground or airborne; re-simming one settle second after load
  is cheaper than detecting per-particle penetration.
- **What persistence does NOT buy:** it is not a checkpoint system for correctness, it
  is *continuity polish* — "the creek I dammed yesterday is still dammed." That has real
  value for a sandbox game and near-zero risk.

Effort **(est.)** ~1 day (serialize/debounce/restore/invalidation + worker message).
Risk low. Independent of every other item — can land whenever.

---

## 5. Decision matrix

Target: 60 fps sustained on mid-range laptops at 26k particles. Current worst-case
frame: ~64 (MC) + ~51 (solver, sync) + 0.5 + 1–2 ms. Order chosen so each step is
shippable alone and de-risks the next.

| # | Item | Effort *(est.)* | Risk | Payoff *(est.)* | Why this position |
|---|------|-----------------|------|-----------------|-------------------|
| 1 | **Make the built async worker the default** (`maybeCreateAsyncSim` + COOP/COEP vite headers + 30 Hz sim cadence) | 0.5–1 day | low | removes ~51 ms from the render thread — *guarantees* render-side 60 fps regardless of simMs | Already implemented (`async-sim.js`, `sim-worker.mjs`), just not wired as default. Perf-headroom's own #1 recommendation. No physics change. |
| 2 | **Default the surfacer to screen-fluid at scale** (keep MC for close-up/auto) | 0.5 day + tuning | low-medium | 64 ms → 0.5 ms on the frame's largest line item | Screen pipeline already ships and looks right (R4). MC's AUTO-resolution rule caps it, but its cost is inherent. Biggest single constant-factor win available. *(Surfacers belong to other lanes — coordinate, don't fork.)* |
| 3 | **Compression trims** (spacingRatio 0.55→0.62 + `restDensityForSpacing`, pack knobs) | 1 day | medium (retune) | −25–40% of remaining pair work; matters more once #1/#2 expose simMs as the wall-clock bottleneck | Cheapest lever on the measured 71%-dominant phase. Behavioral change — gate behind params, default off, tune on the creek bench. |
| 4 | **Interest culling w/ tick-freeze** (§3) | 2–3 days | med-high | est. 50–70% of pair work on long-channel scenes; scales with world size, not camera | Do after #3 so trims and culling compose multiplicatively. Worker-first architecture (#1) is where the segment table lives. |
| 5 | **LOD ribbon** (§2) | 2.5–3.5 days | medium | mostly *visual* scalability: far water stops costing anything and worlds can extend past the sim bounds | Depends on #4's segmentation for its particle-distance plumbing; also unlocks bigger maps, which increases #4's payoff. |
| 6 | **WASM+SIMD solver in the worker** (perf-headroom P2) | 1–2 weeks | medium | est. 2–3× throughput (26k: 51→18–25 ms in-worker) | Only needed if #3+#4 leave simMs above the worker's own 16 ms budget at target particle counts. Slots into #1's boundary unchanged. |
| 7 | **IndexedDB persistence** (§4) | ~1 day | low | continuity polish; no frame-budget effect | Anytime. Listed last only because it's orthogonal; safe first-issue for a new contributor. |
| 8 | **WebGPU/TSL compute solver** (§1) | 3–6 weeks incl. renderer migration | high | est. <2 ms steps; real target is 100k+ particles | Parked. Prerequisite (WebGPU renderer) invalidates work in lanes 2/5's shaders. Trigger: product needs ≥60k particles, or WebGPU migration happens for rendering reasons anyway. |

**Net path to 60 fps at 26k:** items 1+2 alone take the main thread from ~116 ms to
~52 ms of work that fits a 16.6 ms frame after #1 moves the solver off-thread and #2
collapses the surfacer; items 3–5 then attack the worker-side simMs so the *simulation*
itself sustains 60 Hz (or honest 30 Hz) as particle counts grow. Items 6–8 are the
next-generation bets, explicitly sequenced behind the cheap structural wins.

## Reproduce / verify hooks

```
node src/water-pack/bench-report.mjs     # baseline phase profile (perf-headroom.md)
node src/water-pack/solver-bench.mjs     # regression gate any solver change must pass
```

Any implementation of #3–#6 should extend `bench-report.mjs` with the relevant counter
(active-particle fraction for #4, pair-count delta for #3) rather than trusting frame
anecdotes.
