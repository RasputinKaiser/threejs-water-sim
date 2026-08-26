# R1 — Modern real-time fluid RENDERING for WebGL2 / three.js r185

**Goal:** replace the CPU `MarchingCubes` metaball surface (`src/water-pack/surface.js`) with a
screen-space fluid pipeline that renders the PBF particles from `src/water-pack/solver.js`
directly — no grid, no mesh extraction, camera-independent cost.

**Codebase facts this is grounded in** (read from solver.js / surface.js):

- Particle data lives in flat typed arrays on the CPU: `sim.pos` (Float32Array, cap*3),
  `sim.vel`, and — critically for foam — **`sim.nCount` (neighbor count per particle) is already
  computed every relax pass**, plus `sim.nCount`-based foam thresholds already exist in surface.js
  state (`foamMaxSpeed`, `foamMaxNeighbors`). The screen-space renderer can reuse all of it.
- Upload path already exists (surface.js lines 62–69): a preallocated Float32Array copied per frame
  into a BufferAttribute with `needsUpdate = true`. Same pattern carries over unchanged.
- Repo currently has **zero** EffectComposer/postprocessing usage; three r185 ships the needed addons
  under `three/addons/postprocessing/` (verified against the r185 tree: EffectComposer, ShaderPass,
  Pass, RenderPass, OutputPass, ClearPass, MaskPass…). There is **no** bilateral/depth-blur addon —
  that part must be custom GLSL.
- Note: brief says ~22k particles; `DEFAULT_PARAMS.maxParticles = 9000` (solver.js line 26).
  Everything below scales to either; sprite rasterization of ≤25k quads is sub-millisecond.

---

## 1. Recommended pipeline (screen-space fluid rendering)

The canonical technique: render particles as screen-space spheres writing eye-space **depth** into
a float RT → smooth that depth field (curvature flow / bilateral blur) → reconstruct normals from
the smoothed field → shade with refraction + reflection + absorption. Cost is O(particles) +
O(pixels), independent of fluid volume size — unlike MC, where quality ∝ grid resolution³.

### Pass graph

```
[scene opaque] ──► sceneColorRT + sceneDepthTexture          (RenderPass w/ depthTexture)
[sim.pos]      ──► PASS A "sphere depth"  → fluidDepthRT   (R32F, nearest)
                    ├─ (MRT or second pass) → thicknessRT  (R16F, additive)
                    └─                     → foamMaskRT    (R8/R16F, additive)
PASS B "bilateral/curvature smooth"        → fluidDepthSmoothRT  (half-res OK)
PASS C "normals from depth"                → fluidNormalRT  (RGBA16F)
PASS D "composite"                         → refraction, Fresnel, env reflect,
                                             Beer–Lambert absorption from thickness,
                                             foam blend, specular sun/env
final composite over sceneColor            → screen (or feed into your existing post chain)
```

### Which existing three.js classes apply

| Piece | Use | Notes |
|---|---|---|
| Scene render | `EffectComposer` + `RenderPass` | give the composer's render target a `depthTexture` (`THREE.DepthTexture`, `UnsignedIntType`) so the composite pass can read scene depth for water-vs-solid occlusion |
| Custom passes | `ShaderPass` subclassing `Pass` | each of passes B/C/D is one fullscreen quad + one fragment shader. Copy `ShaderPass`'s structure: `FullScreenQuad` from `three/addons/postprocessing/Pass.js` does the quad work |
| Render targets | `WebGLRenderTarget` | fluidDepth: `{ format: RedFormat, type: FloatType, minFilter/magFilter: NearestFilter }`; thickness/foam: HalfFloatType with `AdditiveBlending`; normal RT: RGBA half-float, LinearFilter |
| Final output | `OutputPass` | handles tone mapping + sRGB at the end of the chain |

What must be custom GLSL (no addon exists):
1. sphere-sprite depth shader (§2),
2. depth-aware smoothing — Green's curvature-flow or bilateral (sketch §GLSL-B),
3. normal reconstruction (§GLSL-C),
4. composite/shading (§GLSL-D).

You can drive these either as `EffectComposer` passes appended after a `RenderPass`, **or** without
the composer at all: just N `WebGLRenderTarget`s cycled through `FullScreenQuad.render(renderer)`
calls. For a game that likely has its own post stack later, plain manual passes are simpler to
insert selectively (only when water is visible).

---

## 2. Rendering ~20k particles as screen-space spheres

Two viable sprite mechanisms:

**(a) `THREE.Points` + custom ShaderMaterial (recommended first).**
One vertex per particle, zero index buffer, one draw call. Vertex shader computes point size;
fragment shader carves a circle out of `gl_PointCoord`, solves the sphere silhouette, offsets the
view-space position along the view ray by z, and writes `gl_FragDepth`.

```glsl
// ---- vertex ----
uniform mat4 projectionMatrix, modelViewMatrix; // provided by ShaderMaterial
uniform float uSphereRadius;                    // ≈ 0.55 * sim.h * 0.5 … tune vs isolation look
in vec3 position;                               // particle center (from sim.pos attribute)
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  // perspective size attenuation: project radius at depth -mv.z
  gl_PointSize = uSphereRadius * projectionMatrix[1][1] * /* viewportH/2 */ uViewport.y / -mv.z;
}
// ---- fragment ----
precision highp float;
uniform float uSphereRadius; uniform vec2 uViewport; uniform mat4 projectionMatrix;
in float vEyeZ;                       // pass -mv.z from vertex
layout(location=0) out vec4 outDepth; // R32F target: linear eye-space depth
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(d, d);
  if (r2 > 1.0) discard;
  float nz = sqrt(1.0 - r2);                 // hemisphere height
  // move along view ray to the sphere front surface:
  float eyeZ = vEyeZ + nz * uSphereRadius;   // view looks down -Z
  vec4 clip = projectionMatrix * vec4(0.0, 0.0, -eyeZ, 1.0);
  gl_FragDepth = 0.5 * (clip.z / clip.w) + 0.5;
  outDepth = vec4(-eyeZ, 0.0, 0.0, 1.0);     // store LINEAR depth for smoothing/normals
}
```

Because every sprite is convex and you want the frontmost surface anyway, you don't even need
blending here — regular depth-test-less overwrite works; enabling the hardware depth test against
the RT's own depth buffer saves fill on back spheres.

> ⚠️ **Pitfall — point-size limits.** `gl_PointSize` is clamped to `ALIASED_POINT_SIZE_RANGE`,
> which some drivers cap low (64–256). Desktop M-series/Metal→ANGLE allows large points, but if
> particles get close to a moving camera their sprites exceed any cap and pop to small squares.
> Guard: query `renderer.capabilities` / `gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)` once; if
> gameplay allows near-camera water, switch to option (b).

**(b) Instanced billboard quads (fallback, fully robust).**
`InstancedBufferGeometry` (one 4-vert quad) × `InstancedBufferAttribute` holding `sim.pos`.
Vertex shader expands the quad in view space by projected radius (same attenuation math as above,
no clamp issue) and passes the local offset as a varying replacing `gl_PointCoord`. Identical
fragment logic otherwise. Costs 4× vertices but removes every driver limit. Given a moving-camera
game, **this is the safe end-state**; start with points, swap when close-ups matter — the fragment
shader is shared verbatim.

**Data upload:** exactly what surface.js already does — keep one persistent `Float32Array`
attribute bound to `sim.pos` (same buffer object identity lets you skip copies entirely:
`new THREE.BufferAttribute(sim.pos, 3)` with `usage: DynamicDrawUsage`, then
`attr.needsUpdate = true; attr.updateRanges = [{start:0, count: sim.count*3}]` each frame —
r185 supports partial `updateRanges`). Also pack per-particle speed and `nCount` into a second
vec2 attribute for foam/thickness modulation (both already computed CPU-side).

---

## 3. Depth thickening (absorption) + foam

**Thickness.** Second additive-blended RT (R16F, `AdditiveBlending`, no depth test). Per sprite,
integrate the chord length through the sphere at `gl_PointCoord`:

```glsl
float chord = 2.0 * sqrt(1.0 - r2) * uSphereRadius;  // r2 from gl_PointCoord as above
outThick = vec4(chord);
```

Summed thickness T feeds **Beer–Lambert absorption**: `color *= exp(-sigma * T)` with a tinted
`sigma` (e.g. σ_rgb ≈ (0.45, 0.12, 0.06)·k for blue-green water). This is what gives deep pools
dark teal and shallow edges bright cyan — the visual cue MC-with-uniform-opacity can't produce.
(Original idea: Kipfer & Westermann 2005; standardized in Green GDC 2010.)

**Foam — two sources, use both:**

1. **Data-driven (primary, free):** the solver already computes `sim.nCount` per particle every
   step and surface.js already thresholds it (`foamMaxNeighbors < nCount` ⇒ sparse ⇒ foam, plus
   speed > `foamMaxSpeed`). Render qualifying particles as soft white sprites into an additive
   foamMaskRT (soft radial falloff, not hard circles), then composite the mask onto the water
   surface only where fluid exists (mask × fluid-depth-validity). This replaces today's separate
   `foamPoints` THREE.Points overlay with something anchored to the actual surface.
2. **Depth-discontinuity extraction (secondary):** in the composite pass, compare raw vs smoothed
   depth or take |∇depth| of the *smoothed* field: large gradients at splash sheets/spray where
   thin water should read white. One texture fetch neighborhood, nearly free:

```glsl
float dl = abs(D(vUv - e.x) - D(vUv + e.x)) + abs(D(vUv - e.y) - D(vUv + e.y));
float edgeFoam = smoothstep(uEdgeLo, uEdgeHi, dl);
```

---

## 4. Alternatives vs MarchingCubes for a moving-camera game

| Option | Verdict for THIS repo |
|---|---|
| **Screen-space sprites (above)** | ✅ Recommended. O(20k) sprites + 3–4 half/fullscreen passes ≈ few ms GPU; zero CPU per-frame surface cost (kills the entire `mc.addBall` loop + `mc.update()` mesh rebuild, which today scales O(n · res³-ish)); domain not limited to the fixed `bounds` box; quality independent of camera. Weaknesses: no silhouettes behind other water (fine), thin features need care, transparent sorting handled implicitly. |
| **GPU marching cubes via WebGPU/TSL** | Feasible-but-not-required. three r185 WebGPURenderer + TSL compute could build the density field and run MC in compute shaders (the classic Museth approach), removing the CPU `addBall` sweep while keeping a real mesh (nice for shadows/refraction). But: splits rendering into WebGPU-only path (WebGL2 fallback still needs the sprite path), TSL MC plumbing is DIY, and it keeps the resolution³ grid cost we're trying to shed. Park unless a hard requirement for true refracted mesh appears. |
| **SDF raymarch from 3D-texture splat** | Render particles into a 3D density texture, raymarch an SDF per pixel. Beautiful but fill-rate-heavy for full-screen water; worse than screen-space for this budget. |
| **Keep CPU MC as a toggle** | Keep `surface.js` working as a debug/fallback mode (it already has `state.mode`). Cheap insurance while the new pipeline matures. |

---

## GLSL sketches (the four custom pieces)

### A. Sphere-sprite depth — see §2 code above.

### B. Depth-aware bilateral smoothing (separable, 2 passes H+V)

Green's curvature flow is just this filter iterated ~20× at half-res; a single H+V bilateral at
half-res with a wide-enough kernel (9–13 taps each way) already reads as "liquid" for games.

```glsl
uniform sampler2D tDepth; uniform vec2 uTexel; uniform vec2 uDir; // (1,0)/(0,1)*spread
uniform float uSigmaR; // depth-range sigma
in vec2 vUv; layout(location=0) out vec4 outD;
float D(vec2 uv){ return texture(tDepth, uv).r; }
void main(){
  float c = D(vUv);
  float sum = c, wsum = 1.0;
  for (int i = 1; i <= 6; i++) {
    float wSpatial = exp(-float(i*i) * 0.15);              // gaussian in pixels
    for (int s = 0; s < 2; s++) {
      vec2 off = uDir * uTexel * float(i) * (s == 0 ? 1.0 : -1.0);
      float d = D(vUv + off);
      float wr   = exp(-(d - c)*(d - c) / (uSigmaR*uSigmaR)); // range weight: don't blur across splashes
      sum += d * wSpatial * wr; wsum += wSpatial * wr;
    }
  }
  outD = vec4(sum / wsum, 0.0, 0.0, 1.0);
}
```

Bilateral **upsample** variant: identical weights, but sample the low-res source with bilinear
coords and multiply spatial weight by the bilinear footprint — used if you run smoothing at half
res and upsample straight into the composite (avoids one fullscreen copy).

### C. Normal reconstruction from linear eye-space depth

```glsl
uniform sampler2D tSmoothDepth; uniform vec2 uTexel;       // of the FULL-res depth RT
uniform mat4 projectionMatrix;                              // to convert Z-delta to view XY scale
in vec2 vUv; layout(location=0) out vec4 outN;
float D(vec2 uv){ return texture(tSmoothDepth, uv).r; }
void main(){
  float dC = D(vUv);
  float dx = D(vUv + vec2(uTexel.x, 0)) - D(vUv - vec2(uTexel.x, 0));
  float dy = D(vUv + vec2(0, uTexel.y)) - D(vUv - vec2(0, uTexel.y));
  // convert depth deltas to view-space meters: dz/dx = deltaZ * dC/dNdC_x etc.
  // For linear eye-Z stored directly, slope scaling:
  vec3 n = normalize(vec3(-dx * uDepthScaleX, -dy * uDepthScaleY, 2.0 * uTexel.x)); // uDepthScale = proj scale terms
  outN = vec4(n * 0.5 + 0.5, dC);   // encode normal + keep depth for composite fetches
}
```

(`uDepthScaleX/Y` derive from `projectionMatrix[0][0]/[1][1]` × depth value — precompute on CPU per
frame; exact form in Green 2010 slides.)

### D. Composite / shading

```glsl
uniform sampler2D tSceneColor, tSceneDepth;   // opaque pass results
uniform sampler2D tFluidNormal;               // .xyz normal, .w fluid linear depth
uniform sampler2D tThickness, tFoamMask;
uniform samplerCube uEnvMap;
uniform vec3 uSigmaAbsorb; uniform vec3 uSunDir; uniform vec2 uViewport;
in vec2 vUv; layout(location=0) out vec4 outColor;

void main(){
  vec4 fn = texture(tFluidNormal, vUv);
  if (fn.w <= 0.0 || fn.w > uFarPlane) { discard; }        // no fluid here
  vec3 n = normalize(fn.xyz * 2.0 - 1.0);

  // --- occlusion vs solids: skip water behind walls using hardware scene depth ---
  float sceneZ = linearizeDepth(texture(tSceneDepth, vUv).x);   // standard perspective linearization
  if (sceneZ < fn.w - 0.05) { discard; }                        // solid in front of fluid

  // --- refraction: bend the scene lookup by the surface normal ---
  vec2 refractOff = n.xy * uRefractStrength / max(fn.w, 0.5);   // shallower = less bend
  vec3 refr = texture(tSceneColor, vUv + refractOff).rgb;

  // --- absorption from summed chord thickness (Beer-Lambert) ---
  float T = texture(tThickness, vUv).r;
  vec3 absorb = exp(-uSigmaAbsorb * T);

  // --- reflection: Fresnel-weighted env probe ---
  vec3 V = normalize(uCamPos - viewPosFromDepth(vUv, fn.w));
  vec3 R = reflect(-V, n);
  float fres = pow(1.0 - max(dot(n, V), 0.0), 5.0);
  fres = mix(0.02, 1.0, fres);                                  // Schlick, F0 = water
  vec3 refl = textureCube(uEnvMap, R).rgb;

  // --- specular highlight ---
  vec3 H = normalize(uSunDir + V);
  float spec = pow(max(dot(n, H), 0.0), 220.0);

  // --- foam ---
  float foam = clamp(texture(tFoamMask, vUv).r + edgeFoamTerm(), 0.0, 1.0);

  vec3 col = mix(refr * absorb, refl, fres);
  col += spec * vec3(1.0);
  col = mix(col, vec3(0.92, 0.96, 1.0), foam * 0.85);
  outColor = vec4(col, 1.0);                                    // writes over opaque scene
}
```

---

## Integration plan specific to this repo

1. **New module** `src/water-pack/screenSurface.js` exporting the same contract as
   `createWaterSurface(sim, scene, bounds, opts) → { update, addGui, state }` so call sites don't
   change. Internally owns its RTs + passes; `update()` uploads `sim.pos` (+ vel-speed/nCount
   packed attr), runs passes A–D after the main render (hook: `renderer.onAfterRender` or wrap the
   app's render call — repo has no composer today, so simplest is a tiny `renderWithWater(renderer, scene, camera)` helper).
2. **solver.js untouched.** Optionally expose `sim.nCount` upload alongside pos (it's already there).
3. **Half-res smoothing** (fluidDepthSmooth at 0.5×) + bilateral upsample — biggest perf lever;
   artifacts invisible under refraction distortion.
4. **GUI parity:** map old controls → new ones: `resolution/isolation` die (no MC grid);
   `opacity` dies (absorption replaces it); `foamMaxSpeed/foamMaxNeighbors` carry over verbatim.
5. **Tuning starting points:** sphere radius = `0.55 * sim.h * 0.75`; smoothing σR ≈ 0.3·h in
   eye meters; `uRefractStrength ≈ 0.03..0.08`; absorption σ scaled so T ≈ 4·h gives ~50% absorption.

## Sources

- Crane, Llamas, Tariq — *Real-Time Simulation and Rendering of 3D Fluids*, GPU Gems 3 ch. 30
  (screen-space sphere sprites → depth smoothing → normals): https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids ✅ verified live
- Simon Green — *Screen Space Fluid Rendering with Curvature Flow*, GDC 2010 (the definitive
  slide deck for passes A–C; curvature flow iteration counts, normal math):
  https://www.nvidia.com/docs/IO/123746/GDC10_ScreenSpaceFluids_SGreen.pdf ✅ verified live
- van der Laan, Green, Sillion — *Improved Rendering of Particle-Based Fluid Surfaces*, IEEE CG&A
  2009 (weighted-mode-filter depth smoothing, better than plain bilateral at silhouettes):
  https://doi.org/10.1109/MCG.2009.51 (DOI resolves; publisher page bot-walled to scripts)
- Kipfer & Westermann — *GPU Construction of Transparent Surfaces* (depth-thickening origin), 2005.
- three.js r185 addon inventory (EffectComposer/ShaderPass/Pass/OutputPass exist; no bilateral pass):
  https://github.com/mrdoob/three.js/tree/r185/examples/jsm/postprocessing ✅ verified via GitHub API
- three.js ShaderPass docs: https://threejs.org/docs/#/examples/jsm/postprocessing/ShaderPass ✅ verified live
