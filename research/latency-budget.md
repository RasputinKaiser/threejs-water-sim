# End-to-end latency budget — creek scene @ 26k particles

Lane E · 2026-08-26 · research-only, no src touched.
Grounded in: `src/water-pack/sim-worker.mjs` (pacing, SAB publish, phase profiler),
`src/water-pack/async-sim.js` (`step()` backlog guard, `.alpha`, `.posLerp`),
`src/water-pack/screen-fluid.js` (`uploadPositions` ~734, `renderWater` ~751),
`src/water-pack/bench-report.mjs` (phase bench), `metrics.js` (telemetry hook),
and the research baselines `perf-headroom.md`, `motion-quality.md`,
`mc-next-gen.md`, `next-gen-perf.md`.

**Two measured profiles coexist and must not be conflated:**

| profile | total/step | pairs | source |
|---|---|---|---|
| main-thread bench (collapsing lattice, max ram-packing) | 50.9 ms | 36.1 ms (70.9%) | perf-headroom.md, measured 2026-08-25 |
| **worker phase profile, this session (live creek sim)** | **~21 ms** (sum below) | **13.3 ms (~63%)** | worker profiler via `phaseStats`, measured this session |

Session per-step split @26k: **pairs 13.3 · collide 2.6 · viscosity 2.4 · relax 1.8 ·
grid 0.8** ms; gravity+derive ≈ 0.3 ms (est. remainder). The gap vs the bench is
expected, not contradictory: the bench deliberately drives a slab lattice into
maximum compression (17.5 pairs/particle); the live creek flows at lower packing,
so the pair list — and everything proportional to it — is ~2.4× cheaper. The
*shape* of the profile (pairs dominant at ~⅔) matches everywhere. Everything
marked **(est.)** is an estimate.

---

## 1. The full chain

The pipeline is: main thread posts `step` → worker solves → publishes to SABs →
renderer samples + interpolates every rAF regardless of worker cadence. There is
no round-trip wait: the `'frame'` postMessage back is informational only
(`async-sim.js:156-180` updates stats; nothing blocks on it).

| # | stage | where | cost | basis |
|---|-------|-------|------|-------|
| 1 | input event sampled by rAF loop | main thread | 0–16.7 ms (avg ~8) | event lands mid-frame; waits for next tick *(structural)* |
| 2 | `sim.step(dt)` → `postMessage` | `async-sim.js:271-287` | **~5–50 µs** | structured clone of a tiny msg; colliders deduped (`collidersChanged`) so no heightfield copy per frame |
| 3 | message pickup + queue wait | worker | 0 – one batch interval | if the worker is still solving batch N−1 the new step queues behind it; backlog guard drops rather than queues (`async-sim.js:276`) |
| 4 | **solver batch** | `sim-worker.mjs:245-285` | **~16–28 ms observed; ~21 ms median per step** | session phase profile above. Adaptive `maxBatchMs=12` clamp yields **1 step/batch** at 26k on this machine (per-step EMA > 12 ⇒ budget = floor(12/21) = 1) |
| 5 | `syncStateToSAB(flip=true)` | `sim-worker.mjs:196-207` | **~0.2 ms** | pos+vel+nCount `.set()` copies @26k (comment-estimated in both files); prev half may tear during it — accepted, invisible at render cadence |
| 6 | SAB visibility (`Atomics.store` ctl[0], ctl[4]) | — | **≈ 0 µs** | sequentially-consistent atomics; renderer polls `ctl[0]` every rAF, never waits |
| 7 | render-side interpolation | `async-sim.js:235-243` `posLerp` | **~0.02–0.05 ms** *(est.)* | 78k lerps + alpha bookkeeping per rAF |
| 8 | GPU attribute upload | `screen-fluid.js:734-748` | **~0.5–1 ms** *(est.)* | 312 KB `bufferSubData` under the hood via `needsUpdate` |
| 9 | water raster + composite | `renderWater` | **~0.5 ms** | measured (next-gen-perf baseline table); half-res RTs, few fullscreen passes |
| 10 | foam/detail-wave layers | `effects.js`, R4 waves | ~1–2 ms | FoamSystem ≤4000 pts; animates on the *render* clock — carries motion between sim frames for free |
| 11 | display scanout | — | ~1 frame (~16.7 ms) | usually excluded from engine budgets; noted for completeness |

### Interpolation lag

`.posLerp` displays `lerp(prev, curr, α)` with α sweeping over the EMA-estimated
batch interval (`async-sim.js:218-227`). Consequences:

- The **newest displayed state is always between two solver snapshots**, i.e.
  the visible surface lags wall clock by **0 → 1 batch interval (0–28 ms, typ.
  ~10–15 ms)** beyond the solve itself. This is the price of smoothness; the
  hybrid interp+capped-extrapolation scheme in motion-quality §1 recovers most
  of it without re-introducing stepping.
- When a batch is late (>interval), α saturates at 1 and the surface freezes —
  the "α pinned" fraction is measurable and should be a guardrail metric (§4).

### Effective motion-to-photon (est.)

For a discrete interaction (drop an obstacle, toggle a collider) that must
affect the water:

```
sample wait        avg  8 ms   (0–16.7)
post               ~0.05 ms
queue wait         avg ~10 ms  (0–21; uniform behind an in-flight batch)
solve batch        ~21 ms      (16–28 observed)
SAB publish        ~0.2 ms
rAF pickup         avg  8 ms   (0–16.7)
upload + GPU       ~1.5 ms     (0.5–1 upload + 0.5 composite)
───────────────────────────────
typical            ≈ 49 ms     plausible band ≈ 30–75 ms, worst ≈ 90 ms
```

Plus scanout ≈ +16 ms if you count photons-on-glass. For *continuous* flow the
notion collapses to: **the visible water is always ~1 batch old (~21–28 ms)
plus render-side sampling jitter** — the sim advances in solver-sized quanta no
matter how fast the renderer draws. The only levers on that floor are solve
time itself (§3) and extrapolation past `curr`.

---

## 2. Measured vs theoretical minimum

Frame budget at 60 fps = **16.7 ms**. Stages 1–3, 5–10 sum to **~11–12 ms**
fixed overhead around the solve (avg-case; worst-case ~35 ms of pure scheduling
jitter). So:

**60 fps *rendering* at 26k: already met.** The worker owns the solve;
main-thread cost is ~12 ms of render-side work (screen-fluid path). This is the
shipped architecture's whole point — rendering never waits on physics.

**60 Hz *simulation* at 26k requires solve ≤ ~14 ms/step** (leaving ≥2.5 ms
margin inside the 16.7 ms worker budget so batches never queue against
themselves). Measured: **~21 ms — we are at ~45 Hz sim ceiling today.** That is
exactly why `maxBatchMs` clamps to 1 step/batch here: faster cadence beats
bigger batches for latency.

| scenario | est. solve/step | verdict vs 14 ms |
|---|---|---|
| 26k JS, today (measured) | ~21 ms | miss — 45 Hz ceiling |
| 40k JS *(est.)* | **~70–95 ms** | ms/k rises super-linearly through packing (1.40→1.92 measured 8k→26k); even the saturated 30k rate (1.70) gives ≥68 ms |
| 26k WASM+SIMD in worker *(est., perf-headroom P2)* | **~12–15 ms** | **pass** — pairs 13.3→~5–7 (vectorizable pair-list consumers 3–5×; pair-build restructure 1.5–2.5×), other phases scale similarly. Clears true 60 Hz sim at 26k |
| 40k WASM *(est.)* | ~45–55 ms | miss — WASM alone does not buy 40k@60Hz; needs neighbor-work cuts (compression trims / culling) too |
| surfacing, metaballs mode today | MC `update()` ~64 ms @26k **on the main thread** | dominates that mode's entire chain; also blocks rAF directly (worst kind of latency — frame *drops*, not lag) |
| surfacing, sparse surface nets *(est., mc-next-gen §1)* | **~5–10 ms @26k**, linear scaling, off-threadable to the existing worker | makes metaballs mode viable; render-thread cost ~1 ms |

**Theoretical minimum chain:** post (µs) + solve (irreducible) + publish
(0.2 ms) + lerp/upload/composite (~2 ms) + one rAF of pickup (avg 8 ms).
I.e. **minimum motion-to-photon ≈ solve time + ~10–12 ms**. Every millisecond
of the latency argument is a millisecond of `_buildPairs`-dominated solve time —
which is why the decision tree below ranks by attack on that number first and
everything else second.

---

## 3. Decision tree for future work

Start from telemetry (`sim.phaseStats`, `sim.simMs`, α-saturation — all already
exposed):

```
Q0. Is RENDER janky (dropped rAFs, long frames)?
 ├─ YES + metaballs mode active → surface nets (D) or default to screen-fluid.
 │     MC's 64 ms sits ON the render thread; nothing else matters until gone.
 └─ NO ↓

Q1. Does the water LOOK stepped/laggy despite the worker?
 ├─ YES, mostly freeze-at-α=1 (late batches) → tune pacing: maxBatchMs,
 │     maxStepsPerBatch, up-sim boost on disturbance (cheap, hours).
 ├─ YES, micro-stutter between frames → hybrid interp+capped extrapolation
 │     (motion-quality §1, ~0.5 d). Highest perceived-latency win per effort
 │     in the whole tree. Do this before any throughput work.
 └─ NO ↓

Q2. Is sustained meanStepMs > 14 ms at target count? (today @26k: yes, ~21)
 ├─ YES → throughput ladder, in order:
 │   (B) compression/packing trims (spacingRatio 0.55→0.62, pack cap):
 │         −25–40% pair work, ~1 d, medium risk (retune ρ0/stiffness).
 │         Re-measure. If now ≤14 ms → STOP.
 │   (A) WASM+SIMD port of pairs+pair-consumers in the existing worker:
 │         est. 21→12–15 ms @26k, 1–2 wk, medium risk (numeric parity).
 │         Slots into the unchanged worker boundary. Do AFTER (B): trims are
 │         multiplicative with WASM and cost 10× less.
 │   (C) interest culling w/ tick-freeze (next-gen-perf §3): scene-shaped
 │         (long channels win big, basins win zero); 2–3 d, med-high risk.
 └─ NO → sim keeps 60 Hz; invest in visual scalability instead:
       LOD ribbon (next-gen-perf §2) — buys far-water cost ~0 and bigger maps,
       NOT latency. Only after Q2 is green.
```

Ranked by **latency-impact-per-effort** (payoff grounded above):

| # | option | effort | risk | latency payoff | recommendation |
|---|--------|--------|------|----------------|----------------|
| 1 | Hybrid interp/extrapolation (already-shipped worker + `posLerp` kept as base) | 0.5 d | low | kills perceived stepping; recovers up to 1 batch (~21–28 ms) of display lag | **Do first.** No worker change; vel already in SAB |
| 2 | Pacing tuning (`maxBatchMs`, disturbance up-sim) | hours | low | caps worst-case queue+solve burst at a chosen budget | Bycatch with #1 |
| 3 | Compression/packing trims | 1 d | medium | 21→~13–16 ms (est.): may alone clear the 60 Hz bar at 26k | Cheapest lever on the measured dominant phase; gate behind params |
| 4 | WASM+SIMD pairs/consumers in worker | 1–2 wk | medium | 21→12–15 ms @26k (est.): guarantees 60 Hz sim headroom; enables 40k@30 Hz honest | The throughput endgame; slots into existing boundary unchanged |
| 5 | Sparse surface nets (metaballs mode) | 2–3 d | medium | removes 64 ms main-thread blocker → 5–10 ms off-threadable | Only when metaballs close-ups are actually needed; screen-fluid default otherwise |
| 6 | Interest culling / tick-freeze | 2–3 d | med-high | up to ~50% pair work on long-channel scenes (scene-shaped) | After #3/#4 compose multiplicatively |
| 7 | LOD ribbon | 2.5–3.5 d | medium | none direct (visual scalability, map size) | After Q2 green |
| 8 | WebGPU compute solver | multi-wk + renderer migration | high | <2 ms steps (est.), real target 100k particles | **Parked** (next-gen-perf verdict stands): needs WebGPURenderer migration; revisit only then |

---

## 4. Budget guardrails

Concrete gates that catch regressions before they ship. All inputs already
exist or are one-line additions to existing telemetry surfaces.

**G1 — solver bench gate (CI, exists, extend):** `node src/water-pack/bench-report.mjs`
already NaN-checks and exits non-zero on NaN. Add thresholds:
- median total @26k ≤ **55 ms** (recorded baseline 50.9; +8% tolerance) and @8k ≤ **13 ms**;
- pairs share of step ∈ [55%, 80%] — a collapse signals an accidental
  algorithmic change, not a win;
- pairs/particle within ±15% of recorded (packing drift changes everything downstream);
- keep `solver-bench.mjs` (old-vs-new regression) mandatory on any `solver.js` diff.

**G2 — worker phase gate (runtime, data exists):** `phaseStats` posts every 30
batches (`sim-worker.mjs:167-187`). Gate on:
- `meanStepMs` p95 over rolling window ≤ **14 ms @26k** (the 60 Hz line from §2);
- `dominant === 'pairs'` — any flip is a red flag worth a human look;
- `pairsPerStep` drift > +20% week-over-week flags compression inflation
  (scene change or solver regression).

**G3 — batch-cadence gate:** from ctl cells + async-sim state:
- `simMs` (ctl[2], µs) p95 ≤ **1.5 × maxBatchMs** (adaptive clamp is holding);
- EMA frame interval (`_emaIntervalMs`) ≤ **34 ms** — sim never slower than
  ~30 Hz in steady state;
- dropped-step count (`step()` returning −1, `async-sim.js:276`) === **0** over
  a 60 s soak — nonzero means the worker can't keep up and latency is growing unboundedly.

**G4 — interpolation-health gate (needs one counter, ~5 lines):** sample α each
rAF; report fraction pinned at 1 ("late-batch freeze time"). Gate: **<5%** of
frames over a soak. This is the single best end-to-end latency proxy: it goes
up exactly when the chain in §1 breaks down.

**G5 — fuzz suite pass (in-flight lane):** any solver/worker change must pass
NaN/finiteness checks plus fixedSteps replay determinism (`fixedSteps:true`
exists precisely for this, `sim-worker.mjs:249-251`) and spawn/drain/reset
storms interleaved with stepping (exercises the in-place `syncStateToSAB(false)`
teleport path that must never be interpolated).

**G6 — metrics snapshot (plumb-through):** `setupMetrics()` already POSTs JSON
snapshots — include `simMs`, `meanStepMs`, `perPhaseMs`, `pairsPerStep`,
α-saturation, drop count. Gates G2–G4 become greppable assertions on the
`.metrics/<label>.json` artifacts instead of vibes.

## Reproduce

```
node src/water-pack/bench-report.mjs    # G1 baseline + phase profile
node src/water-pack/solver-bench.mjs    # regression gate
# live G2–G4: run creek scene, read sim.phaseStats / .simMs / alpha telemetry
```
