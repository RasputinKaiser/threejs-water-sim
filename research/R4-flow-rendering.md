# R4 — Flow-aware rendering upgrades for the screen-space water shader

Research lane R4 · for `src/water-pack/screen-fluid.js` (GPU screen-space fluid renderer).
Read against: `research/R3-creek.md` (§3 visual cues), `research/R1-rendering.md` (§3 foam),
`src/water-pack/effects.js` (`FoamSystem`, `computeCohesionField`).
Everything marked *(est.)* is an estimate, not a measurement. No benchmarks were run.

**Problem:** the creek reads as a static dark-teal ribbon with a white edge outline.
The composite pass has zero motion cues and its edge treatment fakes "water" with a
constant rim. Everything below injects into the existing 4-pass pipeline
(points depth → bilateral blur → normals → composite) without changing its shape.

Code facts this is grounded in (read from screen-fluid.js):

- Composite inputs are exactly five textures: `tScene`, `tSceneDepth`, `tWaterDepth`
  (= smoothed eye-linear depth, half res), `tNormal` (view-space, raw signed), `tThick`.
- Depth pass writes `vec4(max(0.05, vDist - zOff), 0, 0, 1)` with **MAX blending**
  (`MaxEquation`) — channels G/B/A of `rtDepth` are free real estate per sprite.
- Thickness accumulates in `.r` of `rtThick`, scaled by `uThickScale` (0.12), clamped to 4.0
  in the composite; Beer–Lambert via `T = exp(-uAbsorb * thick)`.
- Normals come from central differences on smoothed depth → they are already noisy-free but
  perfectly motionless; there is **no time uniform anywhere** in the file.
- The composite has no world-position reconstruction — everything happens in view space /
  screen space. `update(camera)` already per-frame transforms the light dir into view space,
  so adding camera basis uniforms there matches the existing pattern.
- FoamSystem owns a ring-buffer `geometry` (`position`, `aLife`, `aSize` attributes,
  `setDrawRange(0, foam.count)`), currently rendered straight to canvas by a ShaderMaterial
  whose frag includes `<colorspace_fragment>` — i.e. it is NOT wired into this pipeline at all.

---

## Priority order (impact per effort, one day total)

| # | Item | Effort *(est.)* | Impact |
|---|------|-----------------|--------|
| 1 | World-space procedural waves along `uFlowDir` (option b) | 2–3 h | ★★★ — first actual motion cue |
| 2 | Depth-based color (bed tint from existing thickness) | ~1 h | ★★ — kills "flat teal ribbon" |
| 3 | Replace hard white edge with curvature foam + Fresnel damping | 1–2 h | ★★★ — removes editor-ish look |
| 4 | Streaked highlights along flow (rides on #1's infra) | 1–2 h | ★★ — directional read |
| 5 | FoamSystem mask rendered into the composite | 3–4 h | ★★★ — white water where it belongs |
| 6 | Velocity-advection of detail UVs (option a) | 6–8 h *(est.)* | ★★ — do only if #1 feels too uniform |

Items 1–5 together ≈ one day *(est.)* and cover every recommendation below except option (a)
and curl noise, which are explicitly parked.

---

## 1. Flow-aligned detail normals

### Option (b) — RECOMMENDED FIRST: world-space procedural waves along a fixed flow dir

The composite is fullscreen; to make detail *world-anchored* (so it doesn't swim when the
camera moves) we need the water surface's world XZ at each pixel. We already have eye-linear
depth `wd`; reconstruct the ray in the vertex-less fullscreen pass:

**JS — new uniforms on `compMat` + fill them in `update(camera)` (existing pattern):**

```js
// compMat.uniforms additions:
uTime:     { value: 0 },
uFlowDirW: { value: new THREE.Vector2(1, 0) },   // downhill direction, world XZ, normalized
uCamPos:   { value: new THREE.Vector3() },
uCamRight: { value: new THREE.Vector3() },
uCamUp:    { value: new THREE.Vector3() },
uCamFwd:   { value: new THREE.Vector3() },
uTanHalf:  { value: new THREE.Vector2() },       // tan(fovY/2), tan(fovX/2)

// update(camera), after the existing light-dir block:
cu.uTime.value = performance.now() / 1000;
const e = camera.matrixWorld.elements;
cu.uCamPos.value.set(e[12], e[13], e[14]);
cu.uCamRight.value.set(e[0], e[1], e[2]);
cu.uCamUp.value.set(e[4], e[5], e[6]);
cu.uCamFwd.value.set(-e[8], -e[9], -e[10]);
const fovY = THREE.MathUtils.degToRad(camera.fov) * 0.5;
cu.uTanHalf.value.set(Math.tan(fovY) * camera.aspect, Math.tan(fovY));
```

**GLSL — inject into COMPOSITE_FS right after `vec3 n = normalize(...)` (line ~201) and
BEFORE the refraction lookup**, so the perturbed normal feeds Fresnel, refraction offset,
mirror offset AND the Blinn-Phong term in one shot:

```glsl
// --- flow-aligned procedural detail (world-anchored sum of sines) ---
vec3 camRay = normalize(uCamRight * (vUv.x * 2.0 - 1.0) * uTanHalf.x
                      + uCamUp    * (vUv.y * 2.0 - 1.0) * uTanHalf.y
                      + uCamFwd);
vec3 wpos = uCamPos + camRay * wd;              // wd = eye-linear depth, already sampled

// fade detail out at silhouettes where the depth reconstruction is unreliable
float detailFade = smoothstep(0.0, uEdgeSoft * 2.0, sd - wd); // sd computed just below; reorder
// two travelling waves along uFlowDirW + one cross ripple for breakup:
float ph = dot(wpos.xz, uFlowDirW);
float w1 = sin(ph * 9.0  - uTime * 4.0);
float w2 = sin(ph * 17.0 - uTime * 7.3 + wpos.z * 2.0);
float wc = sin(dot(wpos.xz, vec2(-uFlowDirW.y, uFlowDirW.x)) * 11.0 - uTime * 2.0);
vec2 slope = uFlowDirW * (9.0  * cos(ph * 9.0  - uTime * 4.0)
                        + 17.0 * cos(ph * 17.0 - uTime * 7.3))
           + vec2(-uFlowDirW.y, uFlowDirW.x) * (11.0 * cos(dot(wpos.xz, vec2(-uFlowDirW.y, uFlowDirW.x)) * 11.0 - uTime * 2.0));
n = normalize(n + vec3(slope, 0.0).xzy * -uDetailAmp * detailFade); // see basis note
```

Basis note: `tNormal` is view-space with Z = toward camera (see `NORMAL_FS`: `vec3(dR-dL, dT-dB, 2.0)`).
For small amplitudes you can add the perturbation directly in view space by transforming the
world-space slope with the upper-left 3×3 of the view matrix (pass as `uViewRot`, mat3, same
`update()` pattern). If you skip that transform, waves will rotate with the camera — visible
at amplitude > ~0.15; worth doing properly since it's 3 lines.

Starting constants *(est., tune live)*: `uDetailAmp ≈ 0.10–0.20`, wave numbers 9/17/11 m⁻¹
(wavelengths ~0.7/0.37/0.57 m — sub-particle-spacing ripples, which is exactly what the
smoothed surface can't produce on its own), speeds 4/7/2 m/s scaled against expected creek
speeds of 0.3–1.5 m/s from R3 §2.2 — scroll faster than the real flow; perceptually the
*contrast* of moving detail matters more than physical accuracy.

Also modulate amplitude by calmness so pools stay glassy (R3 §3 cue): multiply `uDetailAmp`
by a per-pixel factor once option (a)'s speed channel exists; until then keep it global.

Effort: ~2–3 h including the camera-basis plumbing *(est.)*.

### Option (a) — advect detail UVs by per-particle velocity (park until (b) proves insufficient)

Encode per-particle flow in the free channels of the MAX-blended depth RT:

```glsl
// POINTS_VS: add attribute + varyings
attribute vec3 aVel;                 // uploaded alongside sim.pos in uploadPositions()
varying vec3 vVel;
...
vVel = aVel;

// DEPTH_FS: replace gl_FragColor line
gl_FragColor = vec4(max(0.05, vDist - zOff),
                    vVel.x * 0.5 + 0.5,      // view-space flow dir xy packed to [0,1]
                    vVel.y * 0.5 + 0.5,
                    length(vVel));           // speed, for gating
```

`uploadPositions()` gains a second `BufferAttribute` fed from `sim.vel` transformed into view
space on the CPU (or pass world vel and transform in the VS with `viewMatrix`).

Composite decodes:

```glsl
vec4 dw = texture2D(tWaterDepth, vUv);          // sample ONCE, reuse .r everywhere
float wd    = dw.r;
vec2  vflow = dw.gb * 2.0 - 1.0;                // view-space flow dir
float spd   = dw.a;

// scrolling detail texture instead of analytic sines:
vec2 duv = vUv * uDetailScale - vflow * (uTime * uScrollK * clamp(spd / uSpdRef, 0.25, 2.0));
vec3 dn = texture2D(tDetailN, fract-duv-wrap).xyz * 2.0 - 1.0;   // RepeatWrapping!
n = normalize(n + dn * uDetailAmp * smoothstep(0.1, 0.8, spd / uSpdRef));
```

**Caveat (why it's ranked second):** MAX blending applies per channel independently, so G/B
hold the flow of the *fastest* sprite covering the pixel, while R holds the depth of the
*nearest* sprite. In a creek the visible surface is usually the top layer and top particles
are the fastest, so the bias is mostly benign — but at splash sheets and near banks the
decoded direction can disagree with the visible surface *(est.)*. Also note the bilateral blur
only processes `.r`; the flow channels stay unfiltered (fine — bilinear sampling smooths them).
Extra cost: one more dynamic attribute upload + CPU-side or VS-side view transform.
~6–8 h *(est.)* including tuning.

### Option (c) — curl-noise perturbation: park

Divergence-free swirl looks great on lakes/eddies but needs either a 2D noise texture sampled
twice (gradient trick) or analytic simplex derivatives, plus a time dimension for evolution.
No motion-direction information the sines of option (b) don't already provide for a
channelized creek. Revisit only if the creek later gets recirculation eddies behind rocks.

---

## 2. Streaking / stretching highlights along the flow

Cheapest directional cue after #1. Two variants; both assume `wpos` + `uFlowDirW` from §1b:

**(i) Stretched noise streaks added as micro-normals** — compress the noise coordinate ALONG
the flow so features elongate downstream:

```glsl
// inside the §1b block, replace/augment the sine phases:
float along = dot(wpos.xz, uFlowDirW);
float across = dot(wpos.xz, vec2(-uFlowDirW.y, uFlowDirW.x));
// aspect 1:5 streaks — long downstream, short across:
float streakPhase = along * 2.5 - uTime * 3.0 + sin(across * 14.0 + uTime * 1.5) * 0.35;
float streak = sin(streakPhase);
slope += uFlowDirW * (cos(streakPhase) * 2.5 * uStreakAmp);
```

**(ii) Anisotropic specular** — compute Blinn-Phong twice, on normals perturbed only
perpendicular to flow, and combine with different exponents (classic Kajiya-Kay-style hack):

```glsl
// after existing spec line (line ~234):
vec3 nPerp = normalize(n + vec3(vec2(-uFlowDirW.y, uFlowDirW.x), 0.0).xzy * uViewRot * 0.06);
float specStreak = pow(max(dot(nPerp, hv), 0.0), uShininess * 0.35); // wider along-flow lobe
col += uSpecular * 0.5 * specStreak;
```

Start with (i) only — it reuses #1's machinery verbatim and reads strongly at grazing angles.
`uStreakAmp ≈ 0.3–0.5 × uDetailAmp`. *(est. 1–2 h)*

---

## 3. Foam from data we have — feed FoamSystem into the composite

FoamSystem already spawns on `speed² > maxSpeed² || nCount < minNeighbors`
(`effects.js` lines ~172–187), inherits velocity, and owns GPU buffers
(`this.geometry` with `position`/`aLife`/`aSize`, drawRange set to live count).
Today those points render straight onto the canvas, floating OVER the water shading with an
sRGB-encoded material — disconnected from Fresnel/refraction and unaffected when the composite
redraws underneath them. Move them INTO the pipeline:

**Step 1 — new RT + raw material in `createScreenFluid`:**

```js
const rtFoam = new THREE.WebGLRenderTarget(HW, HH, rtOpts);   // same HalfFloat RGBA opts

// Raw material mirroring FOAM_VERT/FOAM_FRAG but WITHOUT <colorspace_fragment>
// (RT content must stay linear; tone/colorspace stays a final-output concern):
const foamMat = new THREE.RawShaderMaterial({
  vertexShader: /* same body as FOAM_VERT: aLife/aSize -> vLife, point size */,
  fragmentShader: /* glsl */
    `precision highp float; varying float vLife;
     void main(){
       vec2 uv = gl_PointCoord - 0.5; float r2 = dot(uv, uv);
       if (r2 > 0.25) discard;
       float soft = smoothstep(0.25, 0.02, r2);
       gl_FragColor = vec4(soft * vLife, 0.0, 0.0, 1.0);   // mask only, no color
     }`,
  blending: THREE.AdditiveBlending, depthTest: false, depthWrite: false, transparent: true,
});
const foamPoints = new THREE.Points(opts.foam.geometry, foamMat);  // shares FoamSystem's geometry
foamPoints.frustumCulled = false;
```

Integration contract change: accept `opts.foam` (the FoamSystem instance) in
`createScreenFluid(...)`. No changes needed inside FoamSystem — its geometry/drawRange updates
keep driving both the legacy overlay (disable it once this lands) and this RT draw.

**Step 2 — render between the thickness pass and the blur (in `renderWater` step 1b):**

```js
if (foamPoints) {
  foamPoints.visible = true;
  clearTo(0, 0, 0, 1);
  r.setRenderTarget(rtFoam);
  r.clear(true, false, false);
  r.render(foamPoints.sceneParentOrOwnScene, camera);  // reuse pointsScene pattern
  foamPoints.visible = false;
}
compMat.uniforms.tFoam.value = rtFoam.texture;
```

**Step 3 — shade foam in COMPOSITE_FS** (after `col = mix(refr, refl, fres)`):

```glsl
uniform sampler2D tFoam;
uniform float uFoamScale;   // accumulation normalizer, start ~2.0 (est.)
...
float foam = clamp(texture2D(tFoam, vUv).r * uFoamScale, 0.0, 1.0);
// foam = opaque white, high roughness: kill Fresnel reflection AND sun glint under it,
// keep a little refracted bed so it doesn't look like paint:
col = mix(col, mix(refr, vec3(0.93, 0.96, 1.0), 0.85), foam);
col *= (1.0 - foam * uSpecularKill);   // uSpecularKill ~ 0.9 — rough foam has no glint
// and soften the alpha edge where foam reaches the shoreline:
edge = max(edge, foam * 0.8);
```

High-roughness reading comes for free from killing the specular + Fresnel terms — foam is
diffuse white, water is glossy; the contrast IS the roughness cue. Optionally break up the
white with a high-frequency noise multiply (`foam *= 0.75 + 0.25 * noise(wpos.xz * 30.)`)
using `wpos` from §1b so clumps aren't blob-uniform.

*(est. 3–4 h: RT + material + contract + tuning thresholds down for creek speeds — R3 §3
already recommends `maxSpeed ≈ 0.8 m/s, minNeighbors ≈ 8–10` vs. the pour scene defaults.)*

---

## 4. Depth-based color — shallow bed tint using existing thickness

Already half-built: `thick` (clamped to 4) drives Beer–Lambert absorption, which darkens deep
water correctly. What's missing is the shallow end reading as *transparent over a brownish
bed*. One mix, injected right after the existing `refr` computation (line ~216):

```glsl
uniform vec3 uBedColor;    // wet creek-bed tint, e.g. (0.42, 0.33, 0.22)
uniform float uBedDepth;   // thickness (world units) at which bed influence ends

// thick is in world units after uThickScale; creek Dmax = 0.45 m (R3 §1.2),
// so full bed tint gone by ~half bankfull depth:
float bedMix = exp(-max(thick, 0.0) * (3.0 / uBedDepth));   // 1 at surface-edge, ~0 at uBedDepth
refr = mix(refr, sceneCol * uBedColor, bedMix * uBedStrength);  // uBedStrength ~ 0.85
```

Constants *(est., tune live)*: `uBedDepth = 0.22` (≈ Dmax/2), `uBedColor` warm brown-grey —
sample the actual terrain material if available; `sceneCol` here should be the UNREFRACTED
lookup (`texture2D(tScene, vUv)`) because bending the bed under a few cm of water looks wrong.
Exponential falloff (not smoothstep) keeps a thin bright rim at the waterline, which doubles
as a natural shore highlight and reduces reliance on the fake outline (§5).

Note `thick` saturates fast at the current gaussian-falloff sprite kernel; if `bedMix` never
reaches ~0 mid-channel, lower `uThickScale` (GUI already exposes it) rather than retuning the
shader. *(est. ~1 h.)*

---

## 5. Edge whitening — kill the constant outline, whiten by speed/curvature instead

Diagnosis of the current look: the "hard white outline" isn't drawn anywhere — it emerges
because reconstructed silhouette normals tilt steeply → `cosT → 0` → Schlick `fres → 1` →
the bright sky/horizon reflection dominates exactly along the water boundary, and
`edge = smoothstep(0.0, uEdgeSoft, sd - wd)` cuts alpha sharply right there. Constant rim,
independent of flow — hence "editor-ish".

Fix in three parts, all in COMPOSITE_FS:

```glsl
// (a) damp Fresnel near the shoreline so the rim stops being a mirror:
fres *= edge;                                   // after computing `edge`, before mix()
col = mix(refr, refl, fres);

// (b) curvature/thin-water foam: gradient magnitude of the SMOOTHED depth.
//     Splash sheets, spray and bank shear all produce large |∇wd|; slow glassy pools don't.
float dl = abs(D(vUv - vec2(uTexel.x,0)) - D(vUv + vec2(uTexel.x,0)))
         + abs(D(vUv - vec2(0,uTexel.y)) - D(vUv + vec2(0,uTexel.y)));  // D = tWaterDepth fetch
float edgeFoam = smoothstep(uGradLo, uGradHi, dl);   // start (est.): 0.02 / 0.08 eye-linear units

// (c) combine with the data-driven foam mask from §3; speed-gating comes from FoamSystem's
//     own spawn rule, so banks whiten ONLY where flow is actually fast:
float foamTotal = clamp(edgeFoam * uEdgeFoamAmt + foam, 0.0, 1.0);
col = mix(col, vec3(0.93, 0.96, 1.0), foamTotal * 0.85);
```

This needs `uTexel` added to `compMat` (it already exists on `blurMat`/`normalMat` at the same
resolution — reuse the Vector2). Once (b)+(c) carry the shoreline, delete any remaining
reliance on the bright rim: keep `edge` purely as the alpha softness it was meant to be.

Why speed-driven rather than pure distance-to-bank: R3 §3 established that `sim.vel` +
`nCount` cleanly separate riffles from pools, and FoamSystem's predicate is already
speed-gated — routing bank whitening through it means a slow pool edge stays dark/glassy
while a fast run against the same bank turns white. Pure geometric edges can't tell the two
apart. *(est. 1–2 h.)*

---

## Parked / not recommended now

- **Option (a) advection (§1)** — correct end-state for non-uniform flow, but option (b) gets
  80% of the perceptual win for 30% of the work *(est.)*. Do it after playtesting #1.
- **Curl-noise eddies** — no eddy sites exist until rocks create recirculation (R3 §1.4);
  revisit then.
- **MRT to avoid the extra foam draw** — WebGL2 MRT would batch depth+thickness+foam into one
  geometry pass, but three's RawShaderMaterial path here uses plain render targets and the
  refactor touches every pass; not worth it for one extra half-res additive draw of ≤4000
  points *(est.: sub-millisecond)*.
- **True particle-level velocity smoothing (flow RT + bilateral on G/B)** — only needed if
  option (a)'s MAX-blend bias becomes visibly wrong.

## Sequencing note

Do §1b + §4 + §5(a) as one commit (all touch the same composite region, immediately visible);
§3 next (contract change, biggest single wow); §2 and §5(b)/(c) as polish. Add GUI knobs for
`uDetailAmp`, `uFlowDirW` angle, `uFoamScale`, `uBedDepth` alongside the existing
`addGui` entries so tuning stays live.
