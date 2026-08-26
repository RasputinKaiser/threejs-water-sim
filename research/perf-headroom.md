# Perf headroom study — WaterSim phase profile @ h=0.35, 8k–30k particles

Lane C1 · 2026-08-25 · measured on this machine via `node src/water-pack/bench-report.mjs`
(subclass of the current `WaterSim` with an instrumented verbatim copy of `step()`;
solver.js untouched). All projections below are **estimates** unless shown as
"measured".

## Method

- Scene: slab lattice spawn at spacing 0.55h collapsing onto an infinite plane
  floor, dt = 1/60, 10 warmup + 60 timed steps, median reported.
- Phases timed around the exact method boundaries used by `WaterSim.step()`:
  gravity+predict (inline loop), grid build (`_buildGrid`), pair build +
  density scatter (`_buildPairs`), viscosity (`_viscosity`), relax-displace
  (`_relax`), collide (`_collide`), velocity derive (inline loop).
- NaN check passes at every count.

## Measured numbers

| particles | pairs/step | pairs/particle | total ms/step (median) | ms per k-particles |
|----------:|-----------:|---------------:|-----------------------:|-------------------:|
|     8,000 |     55,191 |            6.9 |                 11.21 |              1.40 |
|    14,000 |    132,935 |            9.5 |                 21.68 |              1.55 |
|    21,402 |    272,080 |           12.7 |                 38.78 |              1.81 |
|    26,505 |    401,050 |           15.1 |                 50.87 |              1.92 |
|    30,723 |    536,136 |           17.5 |                 52.25 |              1.70 |

### Phase breakdown (median ms, % of step)

| phase                |   8k          |  14k          |  22k          |  26k          |  30k          |
|----------------------|---------------|---------------|---------------|---------------|---------------|
| gravity + predict    | 0.03 ( 0.3%)  | 0.05 ( 0.2%)  | 0.08 ( 0.2%)  | 0.10 ( 0.2%)  | 0.12 ( 0.2%)  |
| grid build           | 0.24 ( 2.2%)  | 0.29 ( 1.4%)  | 0.31 ( 0.8%)  | 0.34 ( 0.7%)  | 0.31 ( 0.6%)  |
| **pair build + ρ**   | **7.14 (63.7%)** | **15.14 (69.8%)** | **27.10 (69.9%)** | **36.05 (70.9%)** | **35.86 (68.6%)** |
| viscosity            | 2.14 (19.1%)  | 4.44 (20.5%)  | 6.92 (17.9%)  | 8.93 (17.6%)  | 10.10 (19.3%) |
| relax displace       | 1.01 ( 9.0%)  | 2.13 ( 9.8%)  | 3.65 ( 9.4%)  | 4.31 ( 8.5%)  | 5.17 ( 9.9%)  |
| collide              | 0.27 ( 2.4%)  | 0.38 ( 1.8%)  | 0.55 ( 1.4%)  | 0.60 ( 1.2%)  | 0.67 ( 1.3%)  |
| velocity derive      | 0.04 ( 0.3%)  | 0.06 ( 0.3%)  | 0.10 ( 0.3%)  | 0.12 ( 0.2%)  | 0.14 ( 0.3%)  |

## Findings

1. **`_buildPairs` is the dominant phase at every count — ~64–71% of the step.**
   It does the only pointer-chasing walk in the solver (linked-list grid
   traversal) plus a scattered write pattern (`rho[j] += …` random-access into
   two Float32Arrays per pair). Everything downstream is contiguous pair-list
   iteration and cheap by comparison.
2. **Cost scales super-linearly with N through compression, then saturates.**
   Pairs/particle grows 6.9 → 17.5 as the deepening pool ram-packs (deep water
   has more in-radius neighbors than a fresh 0.55h lattice). ms/k-particles
   rises 1.40 → 1.92 from 8k→26k; the 30k row is within run-to-run noise of
   26k (~±10% between runs observed).
3. **Headroom verdict for a 26k creek:** single-threaded main-thread sim costs
   ~45–55 ms/step — i.e. **one sim step already eats ~3 frames** at 60 Hz.
   No micro-opt closes that gap alone; it takes either off-main-thread
   execution, a big constant-factor win (WASM/SIMD), reduced neighbor work,
   or running the sim at a lower rate (substeps every other frame).

---

## Optimization options (prioritized; gains are ESTIMATES unless "measured")

### Priority 1 — (e) Web Worker + SharedArrayBuffer double-buffering

**What it buys:** does NOT make the solver faster — it removes ~100% of the
sim's ~50 ms from the render thread's critical path. Render keeps its 16.6 ms
budget regardless of simMs; the sim can also drop to 30 Hz (step every other
rAF with 2·dt) to fit inside the worker's own budget. This is the only option
that guarantees 60fps-ish *rendering* at 26k without touching physics fidelity.

**Effort:** medium (1–2 days). **Risk:** medium — API becomes async; no
physics risk since the math is unchanged.

Feasibility is good precisely because three.js reads `sim.pos` directly:
a `Float32Array` view over a `SharedArrayBuffer` is drop-in for a
`THREE.BufferAttribute` — zero-copy, no per-frame transfer.

Integration steps (concrete, for `water-pack/index.js` and debug harness):

1. **SAB layout.** One control SAB + one state SAB:
   - `ctl = new SharedArrayBuffer(64)` — Int32 header: `[0]` version/frame
     counter (written with `Atomics.store`, read with `Atomics.load`),
     `[1]` particleCount, `[2]` command mailbox, `[3]` ack.
   - `state = new SharedArrayBuffer(maxParticles*3*4 * 2)` — TWO back-to-back
     `pos` buffers (double buffer) so the worker writes frame N+1 into the
     back copy while the renderer reads frame N. `vel` can stay worker-local
     (renderer doesn't need it); `nCount` (foam signal) needs its own SAB if
     the scene reads it.
2. **Solver-side.** Add a construction option (e.g. `new WaterSim({ sabState })`)
   OR wrap: keep WaterSim owning plain arrays and `state.posFront.set(sim.pos)`
   after each step — measure first; a 26k×3 float copy is ~0.1–0.3 ms and may
   make true SAB-backed solver arrays unnecessary for v1. True zero-copy means
   the solver allocates `this.pos = new Float32Array(sab, 0, n*3)` etc.; both
   work with three.js.
3. **Worker (`sim-worker.mjs`).** Imports WaterSim, owns the instance,
   loops: wait on command (Atomics.wait on ctl, or self-scheduled rAF-ish
   setTimeout loop), run `step(dt, colliders)`, flip front/back index,
   `Atomics.store(ctl,0,++frame)`. Colliders arrive as serialized plain
   objects via postMessage when they change (cheap, happens rarely).
4. **Main thread (`index.js`).**
   - `geometry.setAttribute('position', new THREE.BufferAttribute(frontPosView, 3))`.
   - Per rAF: `if (Atomics.load(ctl,0) !== lastFrame) { swap attribute array
     reference to the new front view (or set `needsUpdate` if copying);
     lastFrame = frame }`. Swapping the BufferAttribute's `.array` to the
     other half avoids re-upload semantics issues; alternatively keep one
     attribute over the full 2× SAB and move an offset — simplest correct v1
     is: renderer reads front half; worker writes back half; flip = update
     `attribute.array` offset + `needsUpdate = true`.
5. **Async API migration.** `spawn()/spawnBlock()/drain()/reset()` become
   postMessage commands with an optional ack (Promise). Callers that read
   `sim.count` immediately after spawn must instead read `Atomics.load(ctl,1)`
   (worker-updated) or await the ack. The debug-harness loop must poll frame
   version instead of assuming sync stepping after `sim.step()`; any test that
   asserts on positions right after a step needs `await ack`.
6. **Dev-server requirement.** SharedArrayBuffer needs cross-origin isolation:
   add COOP/COEP headers in vite config:
   `server.headers: { 'Cross-Origin-Opener-Policy': 'same-origin',
   'Cross-Origin-Embedder-Policy': 'require-corp' }`. Any third-party script/
   image loaded without CORP will break under require-corp — audit scene
   assets (textures from CDNs need `crossorigin`/CORS).
7. **Fallbacks.** Feature-detect `typeof SharedArrayBuffer !== 'undefined' &&
   crossOriginIsolated`; fall back to current synchronous path (already works).

**Risks recap:** spawn/drain become async (call-site churn); one frame of
positional latency (invisible in practice); SAB availability depends on
isolation headers (fallback kept).

### Priority 2 — (d) SIMD/WASM port estimate

**Expected gain (estimate):** the pair-list consumers (`_viscosity`,
`_relax` pass 2) are contiguous, branch-light, float32 loops — good SIMD
candidates, est. 3–5× on those phases. `_buildPairs` is harder: linked-list
traversal and scattered `rho[j] +=` don't vectorize well; est. 1.5–2.5× via
restructure (sort pairs by j, or density-scatter into per-cell accumulators).
Combined step-level estimate: **~2–3× overall (26k: ~51 ms → ~18–25 ms)**.
WASM also gets you multithreading via workers + atomics later.

**Effort:** high (1–2 weeks: toolchain, JS/WASM boundary, collider plumbing).
**Risk:** medium — numeric parity (float32 ordering changes), debugging cost,
build complexity in a plain-JS/vite repo. Do AFTER (e), since (e)'s worker
boundary is exactly where a WASM module slots in.

### Priority 3 — (b) reduce neighbor work (the real lever on the dominant phase)

Correction to the original hypothesis, now **measured**: search radius is `h`,
not speed-dependent, and since spacing = 0.55h scales WITH h, cutting
`maxSpeed` or `h` does NOT cut neighbors-per-particle. What inflates neighbor
work is **over-compression**: pairs/particle grew 6.9 → 17.5 purely from pool
depth/packing (measured above). Options that actually shrink the pair list:

- Raise spacing ratio 0.55h → 0.62h: pairs ∝ ratio³ ⇒ est. **~30–40% fewer
  pairs** (≈ −25% step time). Requires recomputing restDensity ρ0 for the new
  lattice (kernel-sum formula is in the params comment: 12q²+6q³) and
  retuning stiffness/viscosity. Effort low-medium, risk medium (behavioral
  change everywhere; scenes get slightly coarser water).
- Cap packing pressure (stronger near-pressure / earlier velocity clamp):
  targets exactly the deep-pool inflation seen at 22k+. Est. **10–20%** on
  deep scenes, ~0 on sprays. Low effort, low-medium risk.

### Priority 4 — (a) skip density recomputation for sleeping/steady particles

In THIS architecture density is scattered pairwise during `_buildPairs`
(no separate per-particle density pass), so the win comes from skipping
pair *creation* for mutually-static neighborhoods: keep per-particle
`sleepSpeed` threshold; a pair is skipped only if BOTH endpoints slept;
any collision/impulse/spawn wakes neighbors (wake radius = h). Expected gain
is scene-shaped: settled pools/basins est. 30–60% of phases 2–5 skipped;
an active creek with continuous flow est. <15%. Effort medium-high (waking
correctness is subtle — frozen-looking water is the failure mode), risk
medium-high. Note the KE trace in the bench shows this test scene never
settles within 60 steps, so validate sleep logic on long-run settle benches.

### Priority 5 — (c) typed-array micro-opts — largely ALREADY DONE

Verified in source: every hot loop hoists the getters once per call
(`const h = this.h`, `const inv = 1 / this.cellSize` at the top of
`_buildPairs`/`_viscosity`/`_buildGrid`/`_relax`) — **no `get h()` calls in
the per-particle/per-pair hot path**; expected gain ≈ 0%. Remaining micro
candidates, all small: merge the 7 pair SoA arrays into one interleaved
Float32Array for locality (est. 3–8% on pair-heavy phases); replace
`Math.floor(x*inv)` with a fast floor (est. 1–2%). Only worth doing as
bycatch while implementing something bigger. Effort low, risk low, reward low.

---

## Recommended sequence

1. **Do (e) first.** It is the only change that guarantees the 60fps render
   target at 26k regardless of simMs, requires zero physics retuning, and its
   worker boundary is reusable for (d) later. Pair it with 30 Hz sim
   (step every other frame, dt=2/60) to fit the worker budget.
2. Then (b)-compression trims (spacing ratio / packing cap) if sim throughput
   itself must rise — cheapest wins against the measured 70% dominant phase.
3. (d) WASM+SIMD as the throughput endgame once the worker pipeline exists.
4. (a) sleep-skip only if scenes prove pool-dominated; (c) as bycatch.

## Reproduce

```
node src/water-pack/solver-bench.mjs   # old-vs-new regression gate (existing)
node src/water-pack/bench-report.mjs   # this study: phase profile @ h=0.35
```
