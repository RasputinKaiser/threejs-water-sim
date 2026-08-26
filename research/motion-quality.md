# R5 — Making 35 Hz particle water LOOK 60 Hz-smooth

Research lane R5 · motion quality for the worker-simmed creek (26k particles,
solver at ~35 Hz / ~28 ms per batch in `sim-worker.mjs`, render at 60 Hz).
Read against `research/perf-headroom.md` (worker/SAB plan),
`research/R4-flow-rendering.md` (composite upgrades), `src/water-pack/screen-fluid.js`
(shaders/pipeline), `src/water-pack/effects.js` (`FoamSystem`),
`src/water-pack/sim-worker.mjs` (what crosses the boundary).
Everything marked *(est.)* is an estimate. No code touched — research only.

**Code facts this is grounded in:**

- `uploadPositions()` (screen-fluid.js ~548) copies `sim.pos` → GPU once per
  `renderWater()` call. With the worker, `sim.pos` only changes when a new
  `frame` message lands — so between worker frames the SAME positions are
  rasterized ~1.7× each. The water surface is literally frozen for ~28 ms,
  twice per sim step. That freeze, not shading, is the dominant "stepping" cue.
- `syncStateToSAB()` (sim-worker.mjs ~62) copies pos/vel/nCount into **single**
  SAB buffers with a plain `.set()` — non-atomic vs. the render thread's read.
  There is no previous-position history anywhere on the render side today.
- The SAB already ships `vel` (`velView`) and `nCount` to the main thread —
  per-particle velocity is FREE on the render side. Every technique below that
  "needs velocity" costs nothing extra to transport.
- `ctl` cells: `[0]` frameId, `[1]` count, `[2]` simMs(µs), `[3]` ready.
  Main thread can timestamp frameId changes to know the live sim period.
- Depth pass writes only `.r` of a MAX-blended HalfFloat RGBA RT
  (DEPTH_FS line ~93) — **G/B/A are free per-sprite payload**, exactly as R4
  already proposed for flow encoding.
- `FoamSystem.update()` advances its own ring-buffer particles on the RENDER
  clock (effects.js) — foam already animates at 60 Hz even when the sim doesn't.

---

## 1. Render-side position interpolation vs extrapolation

### Baseline (today)

Renderer draws whatever is in `sim.pos` right now, whenever it draws. Motion
rate == sim rate (≤35 Hz), plus a phase jitter because worker completion drifts
against rAF. Perceived as micro-stutter / "stepping", worst on fast riffles.

### Pure interpolation — zero lag, capped at sim rate

Keep two snapshots on the render side (`posPrev`, `posCurr`, copied out when
`Atomics.load(ctl,0)` changes — 26k×3 float copy ≈ 0.05–0.15 ms *(est.)*, do it
in the rAF tick, NOT in the worker callback, and take `performance.now()` stamps
`tPrev`, `tCurr` at copy time). Then every rendered frame evaluates:

```
α = clamp((now − tCurr) / max(tCurr − tPrev, 1e−3), 0, 1)
x̂ᵢ = lerp(posPrevᵢ, posCurrᵢ, α)
```

Upload `x̂` instead of `pos` in `uploadPositions()` (reuse the existing
attribute array; the extra lerp is 26k × 3 flops ≈ 0.02 ms *(est.)*).

Properties:

- ✅ Sample times are evenly spaced → motion is perfectly smooth at 60 Hz,
  zero overshoot, zero bank penetration. This alone converts "stepping"
  into "slightly slow".
- ❌ Motion is still *sim-rate* information resampled to 60 Hz: effective
  temporal resolution stays 35 Hz. Fast splash fronts show a subtle
  "under-cranked camera" feel — positions are always BETWEEN real solver
  states, never ahead.
- ❌ Adds one full sim period of latency (28 ms input lag on interactions).
  For a decorative creek this is invisible; for player-stirred water it is
  borderline noticeable.
- Failure mode: if a worker frame is LATE (>1.5×Δt), α saturates at 1 and the
  surface freezes again — worse than baseline because the freeze happens at a
  predictable spot in the motion. Mitigation below (hybrid).

### Pure extrapolation — full-rate motion, overshoot risk

Predict forward from the newest state using the SAB velocity:

```
τ = clamp(now − tCurr, 0, τmax)
x̂ᵢ = posCurrᵢ + velCurrᵢ · τ        (world units; vel is m/s, τ seconds)
```

Properties:

- ✅ Motion continues at full rate between sim frames — genuinely 60 Hz+
  movement, no added latency beyond the one already-unavoidable worker frame.
- ❌ Overshoot artifacts wherever velocity changes faster than the sim
  reports: splash impacts, bank collisions, pour entry. Particles fly
  through walls for up to τmax, then snap back when the next frame lands.
  At τ ≈ 10–20 ms and creek speeds ~1 m/s the error is ~1–2 cm ≈ 0.3–0.6·h —
  visible as surface "fuzzing" right at contact lines *(est.)*.
- ❌ Velocity is itself stale by up to Δt; error compounds quadratically
  (position error ≈ ½·a·τ²). High-acceleration regions are exactly the
  splashes people look at.
- τmax must exist. Good default: `τmax ≈ 0.6·Δt_sim` (~17 ms at 35 Hz) —
  caps worst-case penetration at ~½ a solver step of travel while covering
  normal frame jitter.

### Hybrid (RECOMMENDED): interpolate + capped, per-particle-gated extrapolation

```
τ    = clamp(now − tCurr, 0, 0.6·Δt)
base = lerp(posPrevᵢ, posCurrᵢ, α)          // α as above
pred = posCurrᵢ + velCurrᵢ · τ
// per-particle trust weight: don't extrapolate particles whose velocity
// changed a lot last step (accelerating ⇒ prediction unreliable)
acc  = |velCurrᵢ − velPrevᵢ| / Δt           // needs velPrev: second vel copy
wᵢ   = clamp(1 − acc/aRef, 0, 1)            // aRef ≈ 20–50 m/s² (tune)
x̂ᵢ   = base + (pred − posCurrᵢ) · wᵢ · kExt // kExt ∈ [0..1] global knob
```

- Steady laminar flow (low acc): w≈1 → full extrapolation → crisp full-rate
  motion where the eye tracks individual streaks.
- Splashes/impacts (high acc): w→0 → falls back to pure interpolation →
  no overshoot exactly where extrapolation would fail.
- `velPrev` costs one more 312 KB snapshot copy per worker frame (~0.05 ms
  *(est.)*). `aRef` should go in the GUI next to the other water knobs.
- Global `kExt` lets you dial from "pure interp" (0) to "full extrapolation"
  (1) live while watching the creek — tune once, ship the number.
- Remaining failure modes: (a) late worker frames still stall α — pair this
  with a "stretch" fallback: if `(now − tCurr) > 1.3·Δt`, hold α=1 and let
  extrapolation carry the gap (that is precisely the case extrapolation is
  good at); (b) spawn/drain events make posPrev/posCurr index-shifted —
  reset snapshots (set prev:=curr) whenever `ctl[1]` (count) jumps or a
  spawn ack fires, accepting one frozen frame on those rare events.

Effort: ~half a day *(est.)* — snapshot manager + lerp/extrapolate loop in
`uploadPositions()` path + 3 GUI knobs. No worker changes required (the
ping-pong SAB from perf-headroom Priority 1 composes fine; snapshots can live
main-side regardless of SAB layout). **This is the highest-leverage single
change in this document.**

---

## 2. Screen-space motion blur from per-sprite velocity

R4 §1-option-(a) already designed the carrier: encode view-space velocity in
the free G/B channels of the MAX-blended depth RT. That data serves TWO
consumers: flow-advection (R4) and motion smear (this section). Build it once.

What the composite can do with it, cheapest first:

**(a) Smear INSIDE the water shading, not the backdrop (recommended).**
Offset the refraction lookup along the per-pixel flow direction:

```glsl
vec2 vflow = texture2D(tWaterDepth, vUv).gb * 2.0 - 1.0;   // px/sec in UV space
vec2 ruvM  = ruv + vflow * uMBAmt;                          // uMBAmt ~ 0.004 (est.)
// optionally average ruvM with ruv (2-tap) instead of replacing:
sceneCol = 0.5 * (texture2D(tScene, ruv).rgb + texture2D(tScene, ruvM).rgb);
```

Same for the mirror lookup `muv`. Cost: +1–2 texture fetches in an already
fullscreen pass ≈ negligible *(est. <0.3 ms at 1080p half-res inputs)*.
Because the smear is applied to the *refracted image*, not composited over
the scene, there is no classic motion-blur ghosting of background objects —
the bed streaks downstream, which reads as flow, exactly like long-exposure
water photography.

**(b) Sprite-level stretch in POINTS_VS:** elongate `gl_PointSize` by the
screen-space speed so each sprite covers the distance it will travel this
frame:

```glsl
float spdPx = length((modelViewMatrix * vec4(aVel,0.)).xy) * uProj11 / dist
            * (uViewportH * 0.5) * uFrameDt;
gl_PointSize = clamp(ps + spdPx * uStretch, 1.0, 512.0);
```

Isotropic caveat: `gl_PointSize` can't stretch directionally (see §3 for the
real fix); enlarging circles along fast runs thickens/thins coverage and the
thickness RT absorbs some of it. Use mildly (≤1.5×) if at all.

**(c) Full-screen directional blur of the final water color** along vflow
(3–5 taps weighted by speed): genuine motion blur, but it IS ghosting-prone
where water overlays detailed geometry (banks, rocks through refraction), and
it costs a fullscreen multi-tap pass. Only worth it if (a) feels too subtle.

Velocity→UV conversion needs: view-space vel (transform world vel in VS with
`viewMatrix` — no CPU work), multiply by `proj11/dist` and viewport half-height,
divide by dt to get px/frame or keep px/sec with a `uTimeDelta` uniform.
MAX-blend channel bias (G/B hold fastest-sprite flow, not nearest) is benign
here for the same reason R4 gives — top/fastest particles are the visible
surface in a creek *(est.)*.

Ghosting verdict: (a) barely ghosts and mostly helps; (c) hurts. Ship (a),
park (c).

Effort: shared velocity plumbing ~4 h + (a) ~1 h *(est.)*. Rank: good value
AFTER §1, since interpolation removes most of the perceived need.

---

## 3. Particle sprite softening — target the OUTLIERS

At 35 Hz the jumps that read as "pops" are dominated by outlier particles:
spray ejected off splash sheets, droplets crossing air gaps, neighbor-starved
particles the pressure solve barely constrains. Interior pool particles move
<1 px/frame even at 35 Hz and never pop. So don't spend budget on all 26k —
spend it on the few hundred outliers.

**Detection is already free:** `FoamSystem`'s spawn predicate
(`speed² > maxSpeed² || nCount < minNeighbors`, effects.js ~187) selects
exactly the outlier population. Reuse it.

Techniques, ranked:

1. **Jump-fade (cheapest, do first).** During the upload pass (CPU, once per
   worker frame), compute per-particle frame displacement
   `dᵢ = |posCurrᵢ − x̂ᵢ_prev|`; write `aJump[i] = clamp(1 − d/h, 0, 1)` into
   a 1-float dynamic attribute (26k ops ≈ 0.03 ms *(est.)*). Multiply sprite
   alpha/coverage by `mix(aFloor, 1.0, aJump)` in DEPTH_FS/THICKNESS_FS
   (`aFloor ≈ 0.4`). A particle that teleports 1.5h this frame dims instead
   of popping; the surrounding blur + thickness hides the hole. With hybrid
   interpolation from §1 the relevant displacement is the *residual*
   `|x̂ᵢ − lerp(posPrev,posCurr,α)|`… practically: just gate on raw
   inter-frame sim displacement, it correlates perfectly with visible pops.
2. **Velocity-stretched sprites (instanced quads for the spray tier ONLY).**
   True directional stretching needs quads, not `gl_PointSize`. Render
   outliers (few hundred) as ONE instanced draw: per-instance pos + vel +
   seed; VS builds a camera-facing quad stretched along the screen-projected
   velocity (`len ∝ 1 + s·|v_screen|·dt`, width constant), FS uses the usual
   gaussian falloff writing into the same depth/thickness RTs. Streaky spray
   reads as FAST — the single strongest "this water moves fast" perceptual
   cue there is. Cost: one extra instanced draw of ≤1–2 k quads into two
   half-res targets ≈ sub-ms *(est.)*. Effort ~1 day *(est.)* including
   feeding it the same predicate FoamSystem uses.
3. **Size-vs-speed tradeoff:** shrink outlier radius with speed
   (`radius_i = base·clamp(1 − 0.3·spd/spdRef, 0.7, 1)` as a per-particle
   attribute — same upload pass as jump-fade). Smaller sprites pop less
   (fewer px of discontinuity) but contribute less thickness; compensate by
   NOT shrinking inside the thickness pass, only the depth pass. Free-ish
   once attributes exist; tune `spdRef` against visible popping.
4. **Do NOT** add random per-frame jitter or soften ALL sprites: softening
   everything is how you get the metaball "slime" the user already dislikes —
   sharp interiors + hidden outliers beats uniformly mushy.

---

## 4. Surface-field smoothing beyond bilateral

Our "surface" is the smoothed eye-linear depth RT (`rtSmooth`). Bilateral
(9-tap, σr) is one H+V pass. Options to improve it, honestly assessed:

**Curvature-flow-style iterations.** Green 2010 (*Particle Simulation Using
CUDA*, §smoothing) iterates `p ← p + λ·κ·n` on PARTICLE POSITIONS — volume
preserving, kills blob noise. We can't move solver positions (worker lane owns
them; also research lane says touch no src), so the honest adaptation is N
extra Jacobi relaxation passes on the depth FIELD:

```
d ← (1−λ)·d + λ·laplacian9(d)      // λ ≈ 0.2–0.33 per iteration
```

- Cost: each iteration is a fullscreen half-res 9-tap pass ≈ 0.1–0.2 ms
  *(est.)*; 2–4 iterations ≈ 0.2–0.8 ms. Fine.
- BUT: plain Laplacian on a heightfield FLATTENS CRESTS — it's a Gaussian in
  disguise, i.e. strictly weaker than what bilateral+range-weight already
  achieves, minus the edge preservation. Green's volume-preservation term
  doesn't translate to a screen-space height field. Verdict: **low expected
  gain, park it.** If normals ever look "blobby" again, prefer raising
  bilateral sigma slightly over adding Laplacian passes.

**Temporal coherence (RECOMMENDED instead).** Normal "boiling" (per-frame
normal re-derivation noise on the stepped surface) is a big part of the
non-liquid feel. Add a temporal EMA on the smoothed depth:

```
dAccum = mix(dNew, dPrevReprojected, uT)     // uT ≈ 0.4–0.6
guard: reject history where |dNew − dPrev| > thresh  (disocclusion / splash)
```

Fixed-camera creek scenes make reprojection nearly identity (store prev
`viewProj` anyway for correctness). One extra sample+write fullscreen pass
(~0.1 ms *(est.)*). Kills boiling, further stabilizes normals AND the
foam/edge terms that derive from them. Ghosting guard is mandatory — without
the |Δd| clamp, splashes smear into trails.

**Order-of-operations trick (free):** run the bilateral BEFORE temporal
accumulation, accumulate in the eye-linear domain (current representation),
and derive normals LAST from the temporally-accumulated field. Normals then
inherit 60 Hz temporal stability instead of 35 Hz steps.

---

## 5. What shipped titles actually do (patterns worth stealing)

Named examples are from public postmortems/talks; treat specifics as reported,
not verified here.

1. **Full-rate AMBIENT MOTION LAYER over a low-rate sim.** The universal
   trick (Valve flow-mapped water — "Water Flow in Portal 2", SIGGRAPH 2010;
   virtually every river in every AAA since): the *shader-animated* surface
   (scrolling normals, flow maps, foam textures) animates at render rate and
   carries the perception of motion, while the expensive sim underneath
   updates slowly and only shapes the bulk. **We already built exactly this**
   — R4 §1b advected detail waves run off `uTime` at 60 Hz. Keep them strong;
   they mask sim stepping better than any interpolation scheme. Corollary:
   when tuning §1's knobs, evaluate with detail waves ON.
2. **Whitewater as a SEPARATE FULL-RATE system.** Film and game FX pipelines
   (Houdini whitewater convention, adopted widely in games) decouple spray/
   foam from the fluid rate. Our `FoamSystem` already integrates its own
   particles on the render clock — it is structurally the industry pattern.
   Strengthening foam coverage (R4 §3) directly buys motion-quality: dense
   white chaotic motion hides stepping behind it completely.
3. **Up-sim on interaction, decay back to base rate.** Games with interactive
   water run the sim hot only while disturbed (splash triggers N high-rate
   substeps, then blends back to the idle rate). Cheap for us: the worker
   accepts a `steps` count (sim-worker.mjs already loops `msg.steps`) — a
   "disturbance boost" that temporarily requests 2 steps/frame when the user
   drops objects in, reverting after ~1 s. Pairs badly with pure
   interpolation (more latency when it matters) — pairs PERFECTLY with §1
   hybrid extrapolation.
4. **Spatial LOD of the sim.** Shipped titles concentrate particles near the
   camera/interaction (adaptive emission, drain-and-respawn far fields —
   cf. From Dust, various Ubisoft talks). Out of scope for this lane but the
   motion corollary matters: distant water should rely MORE on the ambient
   shader layer and LESS on particle motion, so stepping is least visible
   where particles are sparsest.
5. **Temporal upsampling/reprojection of effect elements** (engine-level TAA
   applied to volumetrics/effects): the same math as §4's temporal EMA —
   evidence that the cheap version (fixed-camera EMA with clamps) captures
   most of the benefit without full reprojection machinery.

---

## Ranking — impact per effort FOR OUR PIPELINE

| # | Technique | Effort *(est.)* | Impact | Why here |
|---|-----------|------------------|--------|----------|
| 1 | §1 Hybrid interp+capped extrapolation (per-particle acc gate) | 0.5 d | ★★★ | Removes the actual 28 ms freeze; velocity already in SAB; no worker change |
| 2 | §3.1 Jump-fade attribute for outliers | 2–3 h | ★★★ | Cheapest kill of the worst remaining artifact (spray pops); rides the same upload pass |
| 3 | Keep/strengthen 60 Hz layers: R4 detail waves + FoamSystem into composite | (R4 scope) | ★★★ | Industry-standard masking; already built — just don't regress them |
| 4 | §4 Temporal EMA on smoothed depth (+normals derived after) | 3–4 h | ★★☆ | Kills normal boiling; cheap; mandatory clamp guard |
| 5 | §2(a) Refraction/mirror smear along velocity (shared G/B plumbing with R4 opt-a) | 5–6 h total | ★★☆ | Nice-to-have motion richness AFTER #1; no real ghosting risk |
| 6 | §3.2 Instanced velocity-stretched spray quads | ~1 d | ★★☆ | Strong stylistic win for rapids; only worth it after #1/#2 prove the concept |
| 7 | Worker "up-sim on disturbance" boost (steps mailbox) | 2–3 h | ★☆☆ | situational; shines only once #1 exists |
| 8 | §3.3 Speed-scaled radius | 1–2 h | ★☆☆ | Bycatch while building #2's attribute |
| 9 | §4 Curvature-flow (Laplacian) iterations on depth | — | ✗ park | Strictly dominated by bilateral on a heightfield; no volume term survives |
| 10 | §2(c) Fullscreen water-color directional blur | — | ✗ park | Ghosts banks/rocks; superseded by #5 |

**Sequencing suggestion:** #1 + #2 in the first pass (same file region,
immediately A/B-able via `kExt` and `aFloor` GUI knobs), evaluate with R4
waves on (#3), then decide between #4 and #5 based on whether residual
artifacts read as "boiling" (do #4) or "too clean/static" (do #5).

## Reproduce / verify notes

- A/B harness: pause the worker (`ctl[0]` stop advancing) and drive α by hand
  to visualize interpolation paths; `window.__waterDebugTint` exists for
  isolating rasterization already.
- Metric idea: record `performance.now()` deltas between frameId changes over
  10 s → live Δt estimate feeds `τmax`/α instead of a hardcoded 28 ms.
