# R3 — Creek/River Terrain Generation + Continuous-Flow Techniques

Research lane R3 · for the PBF/Clavet particle water solver (`src/water-pack/solver.js`).
Read against: `src/scenes/terrain.html.js`, `solver.js`, `water-pack/effects.js`.
Everything marked *(est.)* is an estimate, not a measurement.

---

## 1. Terrain generation for a convincing creek channel (40 × 24 m play area)

Target: x ∈ [−20, 20], z ∈ [−12, 12], water flows toward +x.

### 1.1 Meandering centerline

**Simple (recommended first pass)** — sine-perturbed centerline, sampled per terrain column:

```js
// amplitudes/wavelengths tuned so max|zc| ≈ 4 m (< 12 − bank width) and
// min radius of curvature ≈ 4 m (≈ 4× half-width — no self-intersection).
const zc = (x) => 3.2 * Math.sin(0.16 * x + 0.9) + 1.4 * Math.sin(0.37 * x + 2.4);
```

Verified numerically for these constants: max |zc| = 3.98 m, min radius of curvature
R = (1 + zc′²)^1.5 / |zc″| ≈ **4.1 m**. Rule of thumb: keep R ≥ 3–4× channel half-width
or the meander pinches and the heightfield can't resolve the bend.

**Better (v2)** — replace one sine with value-noise/fBm guiding:
`zc(x) = A1·sin(k1·x) + fbm1D(x·f)·A2` with f low (~0.05–0.1 cycles/m) so bends stay
wide; clamp `|zc″|` after generation or just eyeball R via the formula above.
Midpoint displacement on the centerline polyline works too but gives more
high-frequency wiggle than a creek needs at this scale.

Parameterize by **arc length**, not x, when you add slope in §1.4 — otherwise the
grade varies with meander phase and pools form on the outside of bends.

### 1.2 Channel cross-section

Parabolic (preferred over V — V pinches particles into the thalweg and starves the
near-pressure term of support):

```
d   = distance from centerline (per column, since meander is gentle: d ≈ |z − zc(x)|)
w   = half-width      → 1.0 m  (2 m geom width)
Dmax= bankfull depth  → 0.45 m
terrainInsideChannel(d) = zb + Dmax * (d/w)²        // zb = channel bottom elevation
```

Creek proportions: natural creeks run width:depth ≈ 6–12:1; w/Dmax = 2.2 reads as a
chunky mountain creek at toy scale, which suits h=0.35 particles (see §4 — depth must
hold ≥ 2 particle layers). Don't go deeper/narrower without raising the particle budget.

Bank lip + blend into floodplain (smoothstep over ~0.6 m so the heightfield resolves it):

```js
if (d < w + bw) {
  const t = smoothstep(w - bw*0.5, w + bw, d);       // bw = 0.6 blend width
  h = lerp(zb + Dmax*(d/w)**2, baseTerrain + 0.15 /*raised crest*/, t);
}
```

Heightfield resolution matters: at SIZE_X=40 use **NX ≥ 201 (DX ≤ 0.2 m)** so the
2 m channel spans ≥ 10 cells. The solver bilinearly interpolates heights and takes
central-difference normals (`_collide`, heightfield branch) — features narrower than
~2 cells are under-represented as colliders even if the mesh shows them.

### 1.3 Downstream gradient

**1–3% grade.** For 40 m at 2%: drop = 0.8 m.

```js
const GRADE = 0.02;
const zb = (x) => ZB_HEAD - GRADE * (x + 20);   // zb(head)=ZB_HEAD, falls toward +x
```

Keep the grade monotonic — no local dips inside the channel (the existing terrain scene
already hit this: "local dips must not trap the channel flow" comment). The rolling-hills
base noise must be either excluded from the channel interior or small (< grade·λ).

### 1.4 Rocks & sandbars

Local Gaussian bumps added to heights AFTER the channel carve:

```js
// rocks: isotropic, near thalweg or on the outside of bends
h += rockAmp * Math.exp(-((d - dRock)**2 + sAlong**2) / (2 * sigma²));
rockAmp 0.15–0.35 m, sigma 0.25–0.5 m   // ≥ 2–3 cells wide or the collider misses them
// point bars: anisotropic (elongated along flow), INSIDE of bends, alternating sides
// scale d by ~0.4 across-channel and sAlong by ~2.0 along-channel, amp 0.08–0.15 m
```

Place rocks by walking the arc-length parameter; deterministic PRNG seed so runs are
reproducible. Keep rock tops below Dmax + 0.1 unless you want exposed boulders with
persistent white water (actually desirable — see §3).

---

## 2. Particle-flow quality on slopes (this solver specifically)

### 2.1 The big one: contact friction stalls the creek ⚠️

`solver.js _collide()` applies friction to tangential velocity **every step while
touching**: `nt = tSp·(1−cf) − cf·|g|·dt`. Computed against the solver's own model:

| cf | velocity below which Coulomb term kills slide | steady creep speed on 2% grade (per-step balance) |
|----|------|------|
| 0.25 (current global) | 0.055 m/s | **~0.013 m/s — effectively stalled** |
| 0.05 | 0.009 m/s | 0.065 m/s |
| 0.02 | 0.003 m/s | 0.164 m/s |

At 2% grade the gravity drive is g·sinθ ≈ 0.196 m/s², and a 0.3–0.4 m deep creek means
**most particles touch the bed every step** (only ~2 layers at h=0.35), so bed friction
acts on nearly the whole column. Current `contactFriction: 0.25` will not produce a river.

Fixes, in order of preference:

1. **Per-collider friction override** (small solver change): let each collider carry an
   optional `friction` field; `_collide` uses `c.friction ?? this.p.contactFriction`.
   Riverbed heightfield: 0.01–0.03. Bank walls / basin: 0.2–0.3.
2. If you don't want to touch the solver: split the world into two heightfields won't
   work (one collider array, same param) — instead accept a single low value
   (0.02–0.04) for the whole creek scene; banks don't need to grip because the parabolic
   section plus raised crest contains water geometrically.
3. Slope-classified friction inside the heightfield branch (normal tilt < ~8° → low
   friction) — hacky, couples rendering intent to physics; avoid.

Also relevant: `viscositySigma: 40` is very high (it's what makes pours calm); linear
viscosity damps exactly the shear a river core needs. For this scene estimate
**sigma 5–12, beta 2–6** *(est. — tune live)*; keep the impulse-clamp guard as-is.

### 2.2 Other pitfalls

- **Creep-stop:** covered above — it's friction, not the pressure solve. Symptom check:
  KE decays to ~0 while particles sit distributed along the whole channel (vs. all
  collected at a pool).
- **Jitter at low depth:** shallow sheets over a bumpy bed re-collide every step and
  buzz. Mitigate: keep the wetted bed smooth (no high-frequency noise inside the
  channel; rocks only where depth ≥ 2 layers), keep depth ≥ 2 particle spacings.
- **Gravity component handling:** nothing to change — full g is applied in step() and the
  tangential component drives the flow; the normal component is absorbed by the position
  projection. Do NOT add a downhill bias force; it fights the density solve and causes
  banding.
- **maxSpeed 12** is fine; creek speeds will be ~0.3–1.5 m/s *(est.)*.

### 2.3 Inflow emitter design

- **Submerged inlet > waterfall inlet.** Spawn a block just below the upstream water
  surface at the channel head with initial velocity matching the target steady speed
  (~0.5–1 m/s along +x, i.e. mostly +x, slight −y). A waterfall pour entrains air
  (nCount collapses → foam rule spams, see §3), splashes over banks, and takes much
  longer to reach steady state.
- Rate-limit like terrain.html.js does (accumulator + `while (emitAcc >= 1 …)`).
  Note `spawn()`'s overlap guard rejects placements < 0.35h from any neighbor — at
  steady state the inlet region is already packed, so effective inflow rate self-limits;
  size the emitter cross-section (nx × ny lattice at 0.55h spacing) for the target
  volumetric rate rather than cranking `rate`.
- Steady-state check: `sim.count` plateaus and `kineticEnergy` plateaus (not oscillating).

### 2.4 Outflow drain + closed-loop recycling

Budget recycling is what makes "continuous" affordable (§4):

- **Option A — delete+respawn:** `sim.drain(region)` exists (axis-aligned box at the
  downstream end); respawn the same count at the inlet. Simple, but `spawn()`'s overlap
  guard makes the return rate jittery once the inlet is packed.
- **Option B — teleport (recommended):** capture particles entering the drain box and
  move them to the next free inlet lattice slot, preserving count exactly.
  Code-level gotchas, both mandatory:
  ```js
  // teleporting WITHOUT syncing prev detonates the derived-velocity step:
  sim.pos[i*3]=nx; sim.pos[i*3+1]=ny; sim.pos[i*3+2]=nz;
  sim.prev[i*3]=nx; sim.prev[i*3+1]=ny; sim.prev[i*3+2]=nz;  // else v=(p-prev)/dt spikes
  sim.vel[i*3]=vinX; sim.vel[i*3+1]=vinY; sim.vel[i*3+2]=vinZ;
  ```
  Snap the landing spot to a 0.55h lattice slot (precomputed ring of inlet slots,
  round-robin) so the overlap guard / near-pressure never sees a close pair.
- Drain box placement: sink it slightly INTO the bed at the downstream pool so only
  arriving water crosses it, sized ≥ 2×2 cells wide to avoid stragglers orbiting the rim.

---

## 3. Making it READ as a river (mapping cues to existing data)

All three signals already exist: `sim.vel[i]` (speed), `sim.nCount[i]` (neighbors),
and `FoamSystem` (`effects.js`) whose spawn predicate accepts ANY `(i) => bool` via
`spawnFromRule`.

| Visual cue | Data source | How |
|---|---|---|
| **Foam lines on fast cores** | `sim.vel` speed | Custom foam rule: `speed > kLocalMean` (compare vs. mean of grid neighbors, computable in one extra pass like `computeCohesionField`) or simply absolute threshold ~0.7–1.0 m/s *(est.)* — lower than the pour scene's default 1.6. Foam inherits velocity (`velInherit`), so streaks already advect downstream. |
| **White water around rocks** | speed gradients + `nCount` | Flow separating over a rock drops nCount (aeration/spray) and spikes speed — the stock flags `{maxSpeed, minNeighbors}` catch both once `minNeighbors` ≈ 8–10 and `maxSpeed` ≈ 0.8 *(est.)*. Exposed rock tops (§1.4 amp high enough to break surface) guarantee a persistent gradient site. |
| **Directional surface features** | mean flow direction | `makeRippleDecalQuad` scrolls its normal map by fixed `scrollX/Y`; rotate the quad to align with local flow and modulate scroll speed by sampled `sim.vel` magnitude near the quad (average particles within a few meters upstream) so ripples visibly accelerate in rapids, slow in pools. |
| **Bank wetting** | particle positions vs. heightfield | Per-frame tint pass: for each particle within ε (~0.05 m) of ground height, darken the corresponding terrain vertex color (heights array indexing is identical to the collider's). O(n), cheap. Alternatively let the metaball compositor's edge handle it and skip. |
| **Glassy slow pools vs. rippled riffles** | speed | Gate ripple-decal opacity/normal strength on local mean speed: strong ripples only above ~0.4 m/s *(est.)*. |

Priority order if doing one at a time: rock white water (free via foam tuning) →
core foam lines (custom rule) → directional ripple scroll → bank wetting.

---

## 4. Performance & feasibility

### Particle count for a filled channel

Lattice spacing s = 0.55h (solver convention), volume/particle = s³:

| h | s | vol/part | full 30×2×0.4 m (=24 m³) | realistic wetted 30×1.6×0.3 m |
|-----|-------|--------|------|------|
| 0.30 | 0.165 m | 4.49 L | 5,343 | 3,206 |
| 0.35 | 0.1925 m | 7.13 L | **3,364** | **2,019** |
| 0.40 | 0.220 m | 10.65 L | 2,254 | 1,352 |

### Feasibility verdict: comfortably feasible ✅

Current cost ~28 ms @ 22k particles scales roughly linearly in n (pair-list
architecture, flat grid, zero steady-state alloc). Linear extrapolation *(est.)*:
3.4k particles ≈ **4–5 ms/step**, 5.3k ≈ 7 ms. Even a 2–3× safety factor leaves the
naive full fill inside a 16 ms frame alongside rendering. Pair-list capacity: 3.4k
particles × ~12–18 neighbors ≈ 40–60k pairs, well under the initial 262k cap.

Grid check: GRID_DIM=128 wrap-masked hash with cellSize=h gives 40/0.35 ≈ 115 cells
along x — fits under 128 without aliasing; negative coordinates already proven fine in
terrain.html.js. Bounds box must span [−21..21]×[y]×[−13..13]-ish; remember leaks get
KILLED (`killLeaks`), so a mis-sized bounds box silently eats recirculated particles.

So the real constraints are **stability (§2.1 friction) and visual quality**, not count.
Strategies if you still want headroom (e.g. h=0.3 for finer spray):

- **Closed-loop recycling (§2.4):** runtime count stays fixed no matter how long the
  river runs; budget = steady-state inventory, not cumulative throughput.
- **Shorter visible reach:** simulate 18–22 m of channel and fade/occlude the rest;
  cuts counts ~35–40%.
- **Larger h downstream only** — not possible today (single global h in solver params);
  would require solver work, not worth it given the numbers above.

---

## TL;DR implementation notes

1. Terrain: sine-perturbed centerline (constants in §1.1 verified), parabolic channel
   w=1.0 / Dmax=0.45, NX≥201 for DX≤0.2, monotone 2% grade, Gaussian rocks σ≥0.25 m.
2. MUST-DO solver change: per-collider `friction` override (riverbed ~0.02) — global
   0.25 mathematically stalls a 2% grade. Drop viscositySigma to ~8 for this scene.
3. Emitter: submerged block spawn at ~0.5–1 m/s + accumulator; recycle by TELEPORT with
   prev[] synced (else velocity spike) to precomputed inlet lattice slots.
4. Visuals: retune FoamSystem thresholds down (speed ~0.8, minNeighbors ~8) for instant
   rock white water; custom speed-gradient rule for core foam lines; rotate/scroll
   ripple quad along flow.
5. Budget: 3.4k particles fills the full 30×2×0.4 channel at h=0.35 → ~4–5 ms/step
   est. Feasible; recycling keeps it fixed forever.
