// water-pack/screen-fluid.js — GPU screen-space fluid renderer.
//
// Classic screen-space fluid surfacing (Kipfer & Westermann 2007; van der
// Laan et al. 2009): render PBF particles as camera-facing point sprites,
// build an eye-linear "front cap" depth buffer, bilateral-smooth it,
// reconstruct view-space normals, and composite a Fresnel / absorption
// water shade over the already-rendered scene.
//
// Integration contract (see water-pack/index.js createScreenBridge):
//
//   import { createScreenFluid } from './water-pack/screen-fluid.js';
//   const fluid = createScreenFluid(sim, renderer, scene, bounds,
//                                   { colorTarget, depthTexture });
//   fluid.update?.(camera);
//   fluid.renderWater(camera);   // draws water over the blitted backdrop
//   fluid.addGui?.(gui);
//   fluid.dispose();
//
// Frame flow (caller-owned): scene is rendered into `opts.colorTarget`
// (with `opts.depthTexture`) each frame, its color is blitted as a
// fullscreen backdrop onto the canvas, THEN renderWater() runs and
// composites water on top of the live canvas contents.
//
// Pipeline (all internal RTs are HALF resolution, HalfFloat RGBA):
//   1. points depth pass    -> RECIPROCAL front-cap view depth 1/z with MAX
//      blending (largest 1/z = nearest surface; 0 = no water)
//   1b. points thickness    -> additive gaussian-falloff thickness
//   2. separable bilateral blur (H then V) on the depth, footprint scaled to
//      the projected particle size, empty texels excluded; the H-pass turns
//      1/z back into z and the V-pass blends with last frame's result
//      (temporal smoothing, ping-pong RTs)
//   3. fullscreen normal reconstruction from view-space positions (one-sided
//      differences, never across a silhouette)
//   4. fullscreen composite -> Fresnel reflection (sky + mirrored scene),
//      refracted scene with Beer-Lambert absorption (depth-tinted: murky
//      shallow -> deep colour), flow-advected detail waves perturbing the
//      normal, grazing-angle fine ripples ((1-N.z)^2 gated) so the horizon
//      isn't a flat mirror, dual-lobe flow-stretched specular (tight glint +
//      broad sheen), thickness-gradient surface foam with flow-advected
//      breakup (diffuse whitening + Fresnel kill), fake shallow-water bed
//      caustics, noise-broken edge foam + flow-aligned shallow streaks,
//      thickness-scaled refraction shimmer, soft edges blended into scene.
//
// Zero per-frame allocations: buffers, RTs and scratch objects are created
// once in the constructor; everything hot-path reuses pooled instances.
// When sim.count === 0 nothing is drawn.

import * as THREE from 'three';

/* ------------------------------------------------------------------ */
/* Shaders                                                             */
/* ------------------------------------------------------------------ */

// Shared fullscreen-triangle-covering quad vertex shader (PlaneGeometry 2x2).
const QUAD_VS = /* glsl */ `
precision highp float;
attribute vec3 position;
varying vec2 vUv;
void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}`;

// Particle sprite vertex shader: projects a world-radius sphere such that
// the point sprite covers exactly the sphere's projected silhouette.
const POINTS_VS = /* glsl */ `
precision highp float;
attribute vec3 position;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform float uRadius;     // world-space particle radius (depth relief)
uniform float uSpriteScale; // sprite footprint radius = uRadius * uSpriteScale
uniform float uProj11;     // projectionMatrix[1][1]
uniform float uViewportH;  // height in px of the RT the sprites render into
varying float vDist;       // view depth of the sphere centre
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float dist = max(0.1, -mv.z);
  vDist = dist;
  // projected diameter in px: (2R * proj11 / dist) * (H/2), R = footprint radius
  float ps = uRadius * uSpriteScale * uProj11 * uViewportH / dist;
  gl_PointSize = clamp(ps, 1.0, 512.0);
  gl_Position = projectionMatrix * mv;
}`;

// Front-cap depth: view depth z of the nearest point of the sphere along
// the ray through this fragment. Point sprites share one vertex depth, so
// there is no z-buffer; instead the pass stores 1/z with MAX blending, which
// keeps the NEAREST surface per pixel (MAX of z itself would keep the
// farthest, i.e. shade the back of the fluid). The RT clears to 0 = empty.
const DEPTH_FS = /* glsl */ `
precision highp float;
uniform float uRadius;
varying float vDist;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(p, p);
  if (r2 > 1.0) discard;
  float z = max(0.05, vDist - uRadius * sqrt(1.0 - r2));
  gl_FragColor = vec4(1.0 / z, 0.0, 0.0, 1.0);
}`;

// Thickness: additive gaussian falloff per sprite; accumulated in R.
const THICKNESS_FS = /* glsl */ `
precision highp float;
varying float vDist;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(p, p);
  if (r2 > 1.0) discard;
  float t = exp(-r2 * 3.0) * sqrt(max(0.0, 1.0 - r2));
  gl_FragColor = vec4(t, 0.0, 0.0, 1.0);
}`;

// Separable bilateral blur on eye-linear depth. Range weight stops the blur
// across depth discontinuities (splashes, water-vs-wall boundaries).
//
// The vertical pass additionally blends with LAST frame's smoothed depth
// (tPrev) — cheap temporal anti-jitter that masks per-frame particle
// stepping. The caller ping-pongs two RTs so tPrev never aliases the target.
// Invalid history (sentinel 0 / clear-distance garbage) falls back to the
// fresh sample so no phantom water is ever created.
const BILATERAL_FS = /* glsl */ `
precision highp float;
uniform sampler2D uTex;
uniform sampler2D tPrev;   // last frame's smoothed depth (ping-pong)
uniform float uTemporalAlpha; // 0 = off, ~0.35 = subtle ghost-tolerant blend
uniform vec2 uTexel;   // 1/rtSize
uniform vec2 uDir;     // (1,0) horizontal pass, (0,1) vertical pass
uniform float uSigmaR; // range sigma (view-depth units, m)
uniform float uRecip;  // 1: input holds 1/z (raw depth pass), 0: input holds z
uniform float uRadiusPx; // projected particle radius in texels at z = 1 m
varying vec2 vUv;

float readZ(vec2 uv) {
  float v = texture2D(uTex, uv).r;
  return uRecip > 0.5 ? (v > 0.0 ? 1.0 / v : 0.0) : v;
}

void main() {
  float d0 = readZ(vUv);
  if (d0 <= 0.0) { gl_FragColor = vec4(0.0); return; } // no water here
  // 9-tap gaussian spatial weights (sigma ~ 2 taps). The tap spacing follows
  // the projected particle size so the kernel spans ~1 particle radius at
  // any distance (a fixed 1-texel step left every sphere bump in place).
  // Integer spacing keeps taps on texel centers: linear filtering would mix
  // depths with empty (0) texels at silhouettes.
  float stp = floor(clamp(uRadiusPx / d0 * 0.3, 1.0, 4.0));
  float w[5];
  w[0] = 0.227027; w[1] = 0.194594; w[2] = 0.121621; w[3] = 0.054054; w[4] = 0.016216;
  float inv2s2 = 1.0 / (2.0 * uSigmaR * uSigmaR);
  float sum = d0 * w[0];
  float wsum = w[0];
  for (int i = 1; i <= 4; i++) {
    vec2 off = uDir * uTexel * float(i) * stp;
    float dA = readZ(vUv + off);
    float dB = readZ(vUv - off);
    float wA = dA > 0.0 ? w[i] * exp(-(dA - d0) * (dA - d0) * inv2s2) : 0.0;
    float wB = dB > 0.0 ? w[i] * exp(-(dB - d0) * (dB - d0) * inv2s2) : 0.0;
    sum += dA * wA + dB * wB;
    wsum += wA + wB;
  }
  float d = sum / wsum;
  // temporal blend: keep some of last frame's smoothed depth where that
  // history holds water (0 = none) at a similar depth
  float pv = texture2D(tPrev, vUv).r;
  pv = (pv > 0.0 && abs(pv - d) < 4.0 * uSigmaR) ? pv : d;
  gl_FragColor = vec4(mix(d, pv, uTemporalAlpha), 0.0, 0.0, 1.0);
}`;

// View-space normal from the smoothed view depth: rebuild view-space
// positions (so the slope has the right scale at every depth — raw depth
// differences per texel flattened slopes by ~1/texel-size) and difference
// toward the neighbor with the smaller depth jump, skipping empty texels, so
// silhouettes never produce the steep bogus normals that shaded as dark rims.
// RGB = normal (signed, HalfFloat RT), A = 1 where water exists.
const NORMAL_FS = /* glsl */ `
precision highp float;
uniform sampler2D uTex;
uniform vec2 uTexel;
uniform vec2 uInvProj; // (1/P[0][0], 1/P[1][1])
varying vec2 vUv;

vec3 viewPos(vec2 uv, float z) { return vec3((uv * 2.0 - 1.0) * uInvProj * z, -z); }

void main() {
  float z0 = texture2D(uTex, vUv).r;
  if (z0 <= 0.0) { gl_FragColor = vec4(0.0, 0.0, 1.0, 0.0); return; }
  vec2 ox = vec2(uTexel.x, 0.0), oy = vec2(0.0, uTexel.y);
  float zl = texture2D(uTex, vUv - ox).r, zr = texture2D(uTex, vUv + ox).r;
  float zb = texture2D(uTex, vUv - oy).r, zt = texture2D(uTex, vUv + oy).r;
  vec3 p0 = viewPos(vUv, z0);
  vec3 dx = vec3(2.0 * uTexel.x * uInvProj.x * z0, 0.0, 0.0); // flat fallback
  if (zr > 0.0 && (zl <= 0.0 || abs(zr - z0) < abs(z0 - zl))) dx = viewPos(vUv + ox, zr) - p0;
  else if (zl > 0.0) dx = p0 - viewPos(vUv - ox, zl);
  vec3 dy = vec3(0.0, 2.0 * uTexel.y * uInvProj.y * z0, 0.0);
  if (zt > 0.0 && (zb <= 0.0 || abs(zt - z0) < abs(z0 - zb))) dy = viewPos(vUv + oy, zt) - p0;
  else if (zb > 0.0) dy = p0 - viewPos(vUv - oy, zb);
  gl_FragColor = vec4(normalize(cross(dx, dy)), 1.0);
}`;

// Composite: Fresnel-weighted reflection + refraction with Beer-Lambert
// absorption over the scene colour buffer, flow-advected detail waves,
// anisotropic (flow-stretched) sun glint, depth-tinted shallows and soft
// speed-aware edges. Outputs straight alpha over the already-blitted backdrop.
const COMPOSITE_FS = /* glsl */ `
precision highp float;
uniform sampler2D tScene;      // scene colour (full res)
uniform sampler2D tSceneDepth; // scene DepthTexture (nonlinear 0..1)
uniform sampler2D tWaterDepth; // smoothed view depth z of the water front (half res, 0 = none)
uniform sampler2D tNormal;     // view-space normal (half res)
uniform vec2 uInvProj;         // (1/P[0][0], 1/P[1][1]) — view rays
uniform sampler2D tThick;      // accumulated thickness (half res)
uniform float uNear;
uniform float uFar;
uniform float uClearDist;      // (legacy, unused — kept for uniform-slot stability)
uniform vec3 uLightDirVS;      // light dir in VIEW space (normalized)
uniform vec3 uDeepColor;
uniform vec3 uShallowColor;    // murky green-brown for thin water
uniform vec3 uAbsorb;          // per-channel absorption coefficient
uniform vec3 uSkyColor;
uniform vec3 uHorizonColor;
uniform float uRefract;        // refraction offset strength
uniform float uReflect;
uniform float uSpecular;
uniform float uShininess;
uniform float uFresnel0;
uniform float uEdgeSoft;       // eye-linear units for edge fade
uniform float uThickScale;     // thickness accumulation -> world units
uniform float uThinCut;        // raw thickness at which a sprite edge is opaque
uniform float uDebugTint;      // >0: output red where water depth exists (debug)
uniform vec2  uFlowDirVS;      // world XZ flow dir projected into VIEW space (xy)
uniform float uFlowSpeed;      // advection speed multiplier
uniform float uFlowDetail;     // advected wave amplitude
uniform float uTime;           // seconds (accumulated in update())
uniform float uAspect;         // drawing-buffer aspect (isotropic wave space)
uniform float uSpecStretch;    // 1 = isotropic spec, <1 stretches glints along flow
uniform float uTintDepth;      // thickness at which the deep colour is reached
uniform float uShallowClarity; // absorption multiplier in the shallowest water
uniform float uFoamEdge;       // edge foam strength (0 = no foam)
uniform float uFoamDamp;       // grazing-angle bank-whitening damping
uniform float uFoamBreak;      // rim breakup amount (0 = uniform rim)
uniform float uFoamStreaks;    // faint flow-aligned streak strength
uniform float uShimmer;        // refraction shimmer strength
// --- M3 quality-parity polish -------------------------------------------
uniform float uFoamSurface;    // surface foam strength (thickness-gradient driven)
uniform float uRippleDetail;   // grazing-angle fine ripple amplitude
uniform float uSheen;          // broad specular sheen lobe strength (0 = off)
uniform float uCaustics;       // fake bed-caustic brightening (shallow water)
uniform vec2  uTexel;          // half-res RT texel (thickness-gradient taps)
// --- Lane D shader polish ------------------------------------------------
uniform float uFoamSoft;       // large-scale foam density modulation (soften)
uniform float uFrameDt;        // clamped seconds since last frame (hitch detect)
uniform float uSpecSteady;     // detail-wave hitch damping amount (0 = off)
uniform float uLobeWiden;      // tight spec exponent multiplier (anti-shimmer)
varying vec2 vUv;

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

// Smooth 2D value noise (texture-less) — used for advected foam breakup and
// fake caustics. Bilinear-smoothed hash lattice reads as drifting organic
// patches rather than per-pixel static.
float vnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = hash12(i);
  float b = hash12(i + vec2(1.0, 0.0));
  float c = hash12(i + vec2(0.0, 1.0));
  float d = hash12(i + vec2(1.0, 1.0));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float sceneDist(vec2 uv) {
  float z = texture2D(tSceneDepth, uv).x;
  if (z >= 1.0) return 1e6; // sky / no geometry
  float ndcZ = z * 2.0 - 1.0;
  return (2.0 * uNear * uFar) / (uFar + uNear - ndcZ * (uFar - uNear));
}

// One tap of the validity-weighted bilinear fetch below.
void waterTap(vec2 uv, float w, inout float z, inout vec3 nrm, inout float cov) {
  float d = texture2D(tWaterDepth, uv).r;
  float wk = d > 0.0 ? w : 0.0;
  z += d * wk;
  nrm += texture2D(tNormal, uv).xyz * wk;
  cov += wk;
}

void main() {
  // Validity-weighted bilinear upsample of the half-res depth + normal:
  // hardware filtering would average real depths with empty (0) texels at
  // silhouettes (bogus mid depths, stair-stepped dark outlines). cov is the
  // fraction of bilinear weight on water texels — a smooth coverage value
  // that anti-aliases the silhouette.
  vec2 st = vUv / uTexel - 0.5;
  vec2 f = fract(st);
  vec2 uv0 = (floor(st) + 0.5) * uTexel;
  float wd = 0.0, cov = 0.0;
  vec3 nSum = vec3(0.0);
  waterTap(uv0,                            (1.0 - f.x) * (1.0 - f.y), wd, nSum, cov);
  waterTap(uv0 + vec2(uTexel.x, 0.0),      f.x * (1.0 - f.y),         wd, nSum, cov);
  waterTap(uv0 + vec2(0.0, uTexel.y),      (1.0 - f.x) * f.y,         wd, nSum, cov);
  waterTap(uv0 + uTexel,                   f.x * f.y,                 wd, nSum, cov);
  if (cov < 0.02) { gl_FragColor = vec4(0.0); return; }
  wd /= cov;
  if (uDebugTint > 0.0) { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); return; }
  vec3 n = normalize(nSum);
  // unit vector from the surface toward the camera (view space)
  vec3 V = -normalize(vec3((vUv * 2.0 - 1.0) * uInvProj, -1.0));

  float thickRaw = texture2D(tThick, vUv).r;
  float thick = min(thickRaw * uThickScale, 4.0);
  // thickness floor for SHADING: additive accumulation can round to ~0 at
  // silhouettes and grazing sheets, making absorption/color collapse to
  // "no water"
  thick = max(thick, 0.25);

  // Droplet shading: isolated balls have steep dome-edge normals that shade
  // as dark rings (fresnel + spec at near-grazing dome slopes). Keep the dome
  // SHAPE (it's what makes a bead read as a bead) but kill the dark-ring
  // shading: clamp the normal's slope so rim fresnel can't spike, and add a
  // fake top-light gradient so droplets look rounded instead of flat discs.
  float droplet = 1.0 - smoothstep(0.0, 0.3, thick);
  n = normalize(mix(n, vec3(0.0, 0.0, 1.0), droplet * 0.45)); // keep some dome shape (rim light applied later, after col exists)

  // --- flow frame (view space) -----------------------------------------
  vec2 fd = length(uFlowDirVS) > 1e-4 ? normalize(uFlowDirVS) : vec2(1.0, 0.0);
  vec2 fp = vec2(-fd.y, fd.x);
  vec2 pw = vUv * vec2(uAspect, 1.0) * 14.0;   // isotropic wave space
  float ft = uTime * uFlowSpeed;

  // --- advected detail waves --------------------------------------------
  // Three sine trains advected along the flow (analytic slopes, no noise
  // textures). Amplitude scales with thickness so splashy bodies ripple
  // while shallow banks stay calm. Spec anti-shimmer: when a frame hitches
  // (large frame-to-frame dt) the advected wave phase jumps and every
  // highlight pops at once — damp the DETAIL amplitude proportionally to
  // how stale this frame is (smoothstep(0, 0.05, dt), weighted uSpecSteady).
  float calm = mix(1.0, smoothstep(0.0, 0.05, uFrameDt), uSpecSteady);
  float amp = uFlowDetail * smoothstep(0.03, 0.9, thick) * calm;
  if (amp > 1e-4) {
    vec2 d1 = fd;
    vec2 d2 = normalize(fd * 0.85 + fp * 0.5);
    vec2 d3 = normalize(fd * 0.75 - fp * 0.65);
    float p1 = dot(pw, d1) * 1.0 - ft * 2.6;
    float p2 = dot(pw, d2) * 1.7 - ft * 3.7;
    float p3 = dot(pw, d3) * 2.9 - ft * 5.1;
    vec2 grad = d1 * cos(p1)
              + d2 * (cos(p2) * 1.7 * 0.6)
              + d3 * (cos(p3) * 2.9 * 0.3);
    n = normalize(n + vec3(grad * (amp * 0.10), 0.0));
  }

  // --- grazing-angle ripple detail (M3) -----------------------------------
  // At low view angles the screen-space height field undersamples the real
  // surface and the horizon reads as a flat mirror. Fine cross-flow ripple
  // trains perturb the normal BEFORE fresnel; amplitude is gated by
  // (1 - N.z)^2 so steep views stay calm and only grazing angles light up.
  float ndv = clamp(dot(n, V), 0.0, 1.0);
  float gAmp = uRippleDetail * pow(1.0 - ndv, 2.0);
  if (gAmp > 1e-4) {
    float qa = dot(pw, fp) * 6.5 - ft * 7.0 + dot(pw, fd) * 1.4;
    float qb = dot(pw, fp) * 10.5 - ft * 9.5 - dot(pw, fd) * 2.3 + 2.1;
    vec2 rGrad = fp * (cos(qa) * 0.62 + cos(qb) * 0.38);
    n = normalize(n + vec3(rGrad * (gAmp * 0.09), 0.0));
  }

  // --- surface foam mask (M3) ---------------------------------------------
  // Foam is a SURFACE feature, not just rim dots: driven by the thickness-
  // gradient magnitude (high at edges / shallows / splash boundaries) broken
  // up by two octaves of flow-advected value noise so it drifts downstream
  // as ragged patches. Where the mask is high we whiten the albedo below and
  // kill Fresnel — foam is diffuse, not mirror-like.
  float fSurf = 0.0;
  if (uFoamSurface > 0.001) {
    float tx = uTexel.x * 1.5;
    float ty = uTexel.y * 1.5;
    float thL = texture2D(tThick, vUv - vec2(tx, 0.0)).r * uThickScale;
    float thR = texture2D(tThick, vUv + vec2(tx, 0.0)).r * uThickScale;
    float thB = texture2D(tThick, vUv - vec2(0.0, ty)).r * uThickScale;
    float thT = texture2D(tThick, vUv + vec2(0.0, ty)).r * uThickScale;
    float gradT = length(vec2(thR - thL, thT - thB));
    float fnz = vnoise(pw * 2.1 - fd * (ft * 1.7)) * 0.65
              + vnoise(pw * 4.3 + fp * (ft * 1.1)) * 0.35;
    // Lane D foam softening: one LARGE-scale octave (~1/8 of the breakup
    // frequency) modulates the whole mask smoothly, so foam regions fade
    // in/out over broad areas instead of popping at the hard threshold.
    // Applied BEFORE the final smoothstep so it shapes patch emergence.
    float flarge = vnoise(pw * 0.26 - fd * (ft * 0.22));
    fSurf = smoothstep(0.05, 0.32, gradT)      // edges / shallows / splash rims
          * mix(0.45, 1.15, fnz)               // advected breakup modulation
          * mix(1.0, mix(0.2, 1.3, flarge), uFoamSoft);
    fSurf = smoothstep(0.42, 0.78, fSurf);     // patchy threshold, not uniform
  }

  // soft edge: fade where water surface nearly coincides with the solid
  // surface behind it (container walls) or is occluded by nearer geometry
  float sd = sceneDist(vUv);
  // occlusion: scene geometry in front of the water surface hides it (the
  // tolerance absorbs the half-res vs full-res depth mismatch along walls)
  float occl = clamp((sd - wd) / (0.03 + 0.004 * wd) + 1.0, 0.0, 1.0);
  // thin-sheet visibility floor: where sd ≈ wd (shallow sheets over terrain,
  // grazing view angles) the fade alone would make thin bodies vanish, so
  // shallow water stays ≥85% visible.
  float edge = max(smoothstep(0.0, uEdgeSoft, sd - wd), 0.85);
  float edgeBand = 1.0 - edge;   // 1 right at the bank

  // --- depth-tinted refraction -------------------------------------------
  // Subtle time-varying shimmer added to the refraction offset, scaled by
  // thickness: still water picks up a faint living wobble instead of acting
  // like glass. Smooth sine interference (not per-pixel white noise) so it
  // reads as slow caustic drift rather than static.
  float shX = sin(pw.x * 2.9 + uTime * 2.1) * sin(pw.y * 3.7 - uTime * 1.6);
  float shY = sin((pw.x + pw.y) * 3.3 - uTime * 2.4)
            * sin((pw.x - pw.y) * 2.3 + uTime * 1.9);
  vec2 shimmer = vec2(shX, shY) * uShimmer * uRefract
               * smoothstep(0.05, 0.8, thick);
  vec2 ruv = clamp(vUv + n.xy * uRefract * (0.35 + thick * 0.65) + shimmer,
                   vec2(0.002), vec2(0.998));
  vec3 sceneCol = texture2D(tScene, ruv).rgb;

  // --- fake caustic brightening on the bed (M3) ---------------------------
  // Moving value-noise pattern boosts the refracted scene's luminance,
  // gated by clamped 1/thickness so only shallow water shows the light-
  // focus shimmer (deep bodies physically would not reveal it). Applied to
  // sceneCol BEFORE absorption so the brightening stays water-tinted.
  if (uCaustics > 0.001) {
    float shal = clamp(0.35 / max(thick, 0.03), 0.0, 1.0);
    float cp1 = vnoise(pw * 1.7 - fd * (ft * 2.3));
    float cp2 = vnoise(pw * 3.4 + fp * (ft * 1.4) + 17.3);
    float caus = smoothstep(0.52, 0.92, cp1 * 0.6 + cp2 * 0.4);
    sceneCol *= 1.0 + uCaustics * shal * caus * 0.8;
  }
  float tmix = smoothstep(0.0, uTintDepth, thick);           // 0=shallow 1=deep
  vec3 body = mix(uShallowColor, uDeepColor, tmix);
  vec3 absEff = uAbsorb * mix(uShallowClarity, 1.0, tmix);   // shallows clearer
  vec3 T = exp(-absEff * max(thick, 0.0));                   // Beer-Lambert
  float tAvg = (T.r + T.g + T.b) / 3.0;
  vec3 refr = sceneCol * T + body * (1.0 - tAvg);

  // --- reflection -----------------------------------------------------
  vec3 rDir = reflect(-V, n);
  float t01 = clamp(rDir.y * 0.5 + 0.5, 0.0, 1.0);
  vec3 sky = mix(uHorizonColor, uSkyColor, t01);
  vec2 muv = clamp(vUv - n.xy * uReflect * 2.0, vec2(0.002), vec2(0.998));
  vec3 reflS = texture2D(tScene, muv).rgb;
  vec3 refl = mix(sky, reflS, 0.25);

  float cosT = clamp(dot(n, V), 0.0, 1.0);
  float fres = uFresnel0 + (1.0 - uFresnel0) * pow(1.0 - cosT, 5.0);
  // Remove the hard white outline: grazing-angle fresnel is what paints
  // container banks white. Damp it towards the edge; noise-gated foam
  // below re-adds whitening only where uFoamEdge asks for it.
  fres *= 1.0 - uFoamDamp * edgeBand;
  // foam is diffuse: kill the mirror term wherever the surface-foam mask is
  // high so patches don't glint like glass
  fres *= 1.0 - fSurf * uFoamSurface;

  vec3 col = mix(refr, refl, fres);

  // --- anisotropic (flow-stretched) specular -----------------------------
  // Flatten the normal's along-flow component before the half-vector dot:
  // the highlight smears along the flow direction and reads as current.
  vec3 hv = normalize(uLightDirVS + V);
  vec3 ns = normalize(vec3(fd * (dot(n.xy, fd) * uSpecStretch)
                         + fp * dot(n.xy, fp),
                           max(n.z, 0.15)));
  float nh = max(dot(ns, hv), 0.0);
  // Dual lobes (M3): lobe 1 is the tight sun glint, lobe 2 a broad low-
  // intensity sheen (wider exponent falloff on the same flow-stretched
  // normal) so highlights have body instead of a single hard spark.
  // Lane D anti-shimmer: the tight lobe's exponent is widened by
  // uLobeWiden (default 0.85) — slightly broader glints stop sub-pixel
  // per-frame normal noise from reading as one-pixel sparkle flicker.
  // Specular: damp on thin/edge water (no glassy glint at the contact line)
  // and on camera/hitch motion (uFrameDt large = camera just moved) so the
  // highlight doesn't flare into a bright "shine" while orbiting.
  float specDamp = (1.0 - 0.6 * edgeBand) * mix(1.0, 0.35, clamp(uFrameDt * 40.0, 0.0, 1.0));
  col += uSpecular * pow(nh, max(uShininess * uLobeWiden, 2.0))
       + uSpecular * uSheen * pow(nh, max(uShininess * 0.16, 2.0));
  col *= mix(1.0, specDamp, step(0.001, nh));

  // --- surface foam whitening (M3) ----------------------------------------
  // Blend white into the albedo where the surface-foam mask is high — after
  // specular so patches read as matte foam, before the rim-foam block.
  if (uFoamSurface > 0.001 && fSurf > 0.003) {
    col = mix(col, vec3(0.93, 0.96, 0.98), fSurf * uFoamSurface);
  }

  // --- edge foam: noise-broken rim + flow-aligned shallow streaks --------
  if (uFoamEdge > 0.001) {
    float band = smoothstep(0.55, 0.95, edgeBand);
    float speedProxy = clamp(uFlowSpeed, 0.0, 2.0) * 0.5;

    // Two animated noise scales break the rim into drifting patches instead
    // of a uniform whitewash; the coarse octave also modulates the foam
    // threshold so the rim's WIDTH varies along the bank.
    float n1 = hash12(floor(pw * 1.3) + floor(vec2(ft * 1.1, -ft * 0.7)));
    float n2 = hash12(floor(pw * 3.1) + floor(vec2(ft * 2.0, -ft * 1.3)) + 31.7);
    float nz = mix(n1, n2, 0.45);
    float thr = (1.0 - uFoamEdge) - (n2 - 0.5) * uFoamBreak * 0.5;
    float rim = band * smoothstep(thr - 0.15, thr + 0.15, nz)
              * uFoamEdge * speedProxy;

    // Faint streaks stretched ALONG the flow (coarse along-flow frequency,
    // fine across-flow), gated to shallow fast water so they read as current
    // slipping over the gravel bar, not spray.
    float sAlong = dot(pw, fd);
    float sPerp = dot(pw, fp);
    float st = hash12(floor(vec2(sPerp * 4.0, sAlong * 0.55))
                    + floor(vec2(0.0, ft * 1.4)));
    float shallowFast = smoothstep(0.45, 0.05, thick) * speedProxy;
    float streak = smoothstep(0.72, 0.95, st) * shallowFast * uFoamStreaks;

    col = mix(col, vec3(0.92, 0.96, 0.98), clamp(rim + streak, 0.0, 0.85));
  }

  // droplet rim light (declared in droplet block): steep dome slopes catch
  // light instead of shading dark — makes isolated balls read as beads.
  float rimDark = smoothstep(0.55, 0.95, length(n.xy));
  col *= mix(1.0, 1.6, rimDark * droplet);

  // --- soft edge: blend the fade into the scene colour -------------------
  // Near banks the shaded colour dissolves into the raw scene colour, so
  // the rim reads as clear thin water instead of a hard alpha outline.
  col = mix(texture2D(tScene, vUv).rgb, col, smoothstep(0.0, 0.45, edge));

  // Sprite footprints are wider than the particles, so a lone particle's
  // halo bleeds past the water (over rims, into the air). Bulk water piles
  // up thickness; halo edges have almost none — fade those out.
  float solid = smoothstep(0.0, uThinCut, thickRaw);
  gl_FragColor = vec4(col, edge * occl * solid * smoothstep(0.02, 0.5, cov));
}`;

/* ------------------------------------------------------------------ */
/* Factory                                                             */
/* ------------------------------------------------------------------ */

export function createScreenFluid(sim, renderer, scene, bounds, opts = {}) {
  const {
    colorTarget = null,
    depthTexture = null,
    radiusScale = 0.62,        // particle world radius = sim.h * radiusScale (up from 0.55 so sparse upstream particles merge instead of reading as separate balls)
    resolutionScale = 0.5,     // internal RTs relative to drawing buffer
    lightDir = [0.45, 0.8, 0.35],
    deepColor = [0.02, 0.23, 0.33],     // saturated teal — richer deep body
    shallowColor = [0.47, 0.52, 0.37],  // sandy green-brown for thin water
    absorb = [1.1, 0.55, 0.38],
    skyColor = [0.45, 0.62, 0.85],
    horizonColor = [0.78, 0.86, 0.95],
  } = opts;

  // temporalAlpha kept LOW by default: same-uv history blending misaligns
  // under any camera motion (water slides in screen space), which reads as
  // tracers/ghosting. 0.18 is enough to mask per-frame particle jitter
  // without visible smear; users can zero it via GUI 'temporal smooth'.
  // blurSigma: bilateral depth-range sigma as a multiple of the particle
  // radius — depth steps well above it (separate bodies) stay sharp, the
  // per-particle bumps below it are smoothed away.
  const state = { enabled: true, flowAngleDeg: 0, temporalAlpha: 0.18,
                  adaptiveTemporal: true, blurSigma: 1.0 };

  const bufSize = renderer.getDrawingBufferSize(new THREE.Vector2());
  const _bufNow = new THREE.Vector2();
  let HW = Math.max(4, Math.floor(bufSize.width * resolutionScale));
  let HH = Math.max(4, Math.floor(bufSize.height * resolutionScale));

  const CLEAR_DIST = 10000.0; // legacy uniform value (unused by the shaders)

  /* ---- internal render targets (half res, HalfFloat, no depth) ------ */
  const rtOpts = {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    wrapS: THREE.ClampToEdgeWrapping,
    wrapT: THREE.ClampToEdgeWrapping,
    depthBuffer: false,
    stencilBuffer: false,
    generateMipmaps: false,
  };
  const rtDepth = new THREE.WebGLRenderTarget(HW, HH, rtOpts);   // raw depth
  const rtTemp = new THREE.WebGLRenderTarget(HW, HH, rtOpts);    // blur H
  // Smoothed depth is PING-PONGED for temporal smoothing: the bilateral
  // V-pass writes rtSmoothOut while reading last frame's result from
  // rtSmoothHist, then the two references swap.
  const rtSmoothA = new THREE.WebGLRenderTarget(HW, HH, rtOpts);
  const rtSmoothB = new THREE.WebGLRenderTarget(HW, HH, rtOpts);
  let rtSmoothOut = rtSmoothA;   // latest smoothed depth (read by normals/composite)
  let rtSmoothHist = rtSmoothB;  // previous frame's (temporal history)
  const rtNormal = new THREE.WebGLRenderTarget(HW, HH, rtOpts);
  const rtThick = new THREE.WebGLRenderTarget(HW, HH, rtOpts);

  /* ---- shared fullscreen quad --------------------------------------- */
  const quadGeo = new THREE.PlaneGeometry(2, 2);
  const quadScene = new THREE.Scene();
  const quadMesh = new THREE.Mesh(quadGeo, null);
  quadMesh.frustumCulled = false;
  quadScene.add(quadMesh);
  const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  /* ---- points (one geometry, two materials) -------------------------- */
  const capacity = Math.max(1, (sim.pos ? sim.pos.length : 0) / 3 | 0);
  const posAttr = new THREE.BufferAttribute(new Float32Array(capacity * 3), 3);
  posAttr.setUsage(THREE.DynamicDrawUsage);
  const pointsGeo = new THREE.BufferGeometry();
  pointsGeo.setAttribute('position', posAttr);
  pointsGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Infinity);

  // Sprites cover 2× the depth-relief radius (historically a side effect of
  // sizing half-res sprites by the full-res height, and what the look is
  // tuned for): wide, flattened caps that merge into a smooth sheet.
  const commonPointUniforms = {
    uRadius: { value: (sim.h ?? 0.3) * radiusScale },
    uSpriteScale: { value: 2 },
    uProj11: { value: 1 },
    uViewportH: { value: HH },
  };

  // NB: do NOT declare modelViewMatrix/projectionMatrix in `uniforms`.
  // three.js uploads those built-ins by name for every shader that declares
  // them in GLSL; listing them here made uploadUniforms overwrite the camera
  // matrices with our placeholder AFTER setProgram refreshed them → vertices
  // transformed by an identity view matrix → everything clipped → no water.
  const depthMat = new THREE.RawShaderMaterial({
    vertexShader: POINTS_VS,
    fragmentShader: DEPTH_FS,
    uniforms: {
      ...commonPointUniforms,
    },
    blending: THREE.CustomBlending,
    blendEquation: THREE.MaxEquation,
    blendSrc: THREE.OneFactor,
    blendDst: THREE.OneFactor,
    depthTest: false,
    depthWrite: false,
    transparent: true,
  });

  const thickMat = new THREE.RawShaderMaterial({
    vertexShader: POINTS_VS,
    fragmentShader: THICKNESS_FS,
    uniforms: {
      ...commonPointUniforms,
    },
    blending: THREE.CustomBlending,
    blendEquation: THREE.AddEquation,
    blendSrc: THREE.OneFactor,
    blendDst: THREE.OneFactor,
    depthTest: false,
    depthWrite: false,
    transparent: true,
  });

  const pointsScene = new THREE.Scene();
  const depthPoints = new THREE.Points(pointsGeo, depthMat);
  const thickPoints = new THREE.Points(pointsGeo, thickMat);
  depthPoints.frustumCulled = false;
  thickPoints.frustumCulled = false;
  depthPoints.visible = true;
  thickPoints.visible = false;
  pointsScene.add(depthPoints, thickPoints);

  /* ---- fullscreen-pass materials ------------------------------------- */
  const blurMat = new THREE.RawShaderMaterial({
    vertexShader: QUAD_VS,
    fragmentShader: BILATERAL_FS,
    uniforms: {
      uTex: { value: null },
      tPrev: { value: null },           // temporal history (V-pass only)
      uTemporalAlpha: { value: 0 },     // set per-frame from state.temporalAlpha
      uTexel: { value: new THREE.Vector2(1 / HW, 1 / HH) },
      uDir: { value: new THREE.Vector2(1, 0) },
      uSigmaR: { value: 0.11 },
      uRecip: { value: 1 },
      uRadiusPx: { value: 1 },          // set per-frame in update()
    },
    depthTest: false,
    depthWrite: false,
  });

  const normalMat = new THREE.RawShaderMaterial({
    vertexShader: QUAD_VS,
    fragmentShader: NORMAL_FS,
    uniforms: {
      uTex: { value: null },
      uTexel: { value: new THREE.Vector2(1 / HW, 1 / HH) },
      uInvProj: { value: new THREE.Vector2(1, 1) },
    },
    depthTest: false,
    depthWrite: false,
  });

  const compMat = new THREE.RawShaderMaterial({
    vertexShader: QUAD_VS,
    fragmentShader: COMPOSITE_FS,
    uniforms: {
      tScene: { value: colorTarget ? colorTarget.texture : null },
      tSceneDepth: { value: depthTexture },
      tWaterDepth: { value: rtSmoothOut.texture },
      tNormal: { value: rtNormal.texture },
      tThick: { value: rtThick.texture },
      uNear: { value: 0.1 },
      uFar: { value: 100 },
      uInvProj: { value: new THREE.Vector2(1, 1) },
      uClearDist: { value: CLEAR_DIST },
      uLightDirVS: { value: new THREE.Vector3(0, 1, 0) },
      uDeepColor: { value: new THREE.Color(...deepColor) },
      uShallowColor: { value: new THREE.Color(...shallowColor) },
      uAbsorb: { value: new THREE.Vector3(...absorb) },
      uSkyColor: { value: new THREE.Color(...skyColor) },
      uHorizonColor: { value: new THREE.Color(...horizonColor) },
      uRefract: { value: 0.03 },
      uReflect: { value: 0.02 },
      uSpecular: { value: 0.6 },
      uShininess: { value: 96 },   // softened from 140 — broader, silkier glint
      uFresnel0: { value: 0.02 },
      uEdgeSoft: { value: 0.06 },
      uThickScale: { value: 0.12 },
      uThinCut: { value: 0.35 },
      uDebugTint: { value: 0 },
      // --- flow-aware shading (B-R1) ---
      uFlowDirVS: { value: new THREE.Vector2(1, 0) },
      uFlowSpeed: { value: 1 },
      uFlowDetail: { value: 0.5 },
      uTime: { value: 0 },
      uAspect: { value: Math.max(1e-3, bufSize.width / Math.max(1, bufSize.height)) },
      uSpecStretch: { value: 0.5 },
      uTintDepth: { value: 1.2 },
      uShallowClarity: { value: 0.6 },
      uFoamEdge: { value: 0.18 },
      uFoamDamp: { value: 0.65 },
      // --- polish pass (P3) ---
      uFoamBreak: { value: 0.6 },    // rim breakup amount
      uFoamStreaks: { value: 0.35 }, // faint flow-aligned shallow streaks
      uShimmer: { value: 0.3 },      // refraction shimmer strength
      // --- M3 quality-parity polish ---
      uFoamSurface: { value: 0.35 },   // surface foam strength (0 = off)
      uRippleDetail: { value: 0.5 },   // grazing-angle ripple amplitude
      uSheen: { value: 0.18 },         // broad specular sheen lobe (0 = off)
      uCaustics: { value: 0.25 },      // fake bed-caustic brightening (0 = off)
      uTexel: { value: new THREE.Vector2(1 / HW, 1 / HH) },
      // --- Lane D shader polish ---
      uFoamSoft: { value: 0.6 },     // large-scale foam softening (0 = off)
      uFrameDt: { value: 1 / 60 },   // clamped frame dt (set per-frame in update)
      uSpecSteady: { value: 1 },     // detail-wave hitch damping amount
      uLobeWiden: { value: 0.85 },   // tight spec exponent multiplier
    },
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });

  /* ---- scratch (zero per-frame allocation) --------------------------- */
  const _prevClear = new THREE.Color();
  const _clearColor = new THREE.Color();
  const _prevRT = { value: null };
  let _prevAutoClear = true;
  let _prevClearAlpha = 1;
  let _lastNow = -1;                            // uTime accumulator
  const _flowWorld = new THREE.Vector3();       // flow dir scratch (world)
  const _camPos = new THREE.Vector3();          // last frame's camera position
  const _camQuat = new THREE.Quaternion();      // last frame's camera orientation
  const _pivot = new THREE.Vector3();           // estimated orbit pivot scratch
  let _haveCamPose = false;                     // until first update (pos+quat)
  // Temporal-history reset thresholds. POSITION alone is not enough: slow pans
  // and orbits move the camera < CAM_JUMP_DIST while the whole scene slides
  // across screen space, so the same-uv history blend smears (tracers/ghosting).
  // We combine translation with the rotational sweep of the scene around the
  // orbit pivot: motion ≈ |Δpos| + angle(Δquat) * dist(camera→pivot).
  const CAM_JUMP_DIST = 0.75;                   // combined world-units; beyond → hard reset
  const _dbg = { pointsCalls: 0, pointsPoints: 0 }; // probe stats (see renderWater)
  // Velocity-adaptive temporal alpha: fast water (splashes) must not blend
  // much history or it ghosts. Water motion is estimated from the sim's own
  // kinetic energy (rms particle speed) — this used to be 8 synchronous
  // readRenderTargetPixels calls per frame, each a full GPU pipeline stall.
  let _waterMotionEMA = 0;                      // EMA of rms speed (m/s)
  let _waterMotionMult = 1;                     // adaptive alpha multiplier
  let _historyStale = false;                    // set on resize

  function blit(material, target) {
    quadMesh.material = material;
    renderer.setRenderTarget(target);
    renderer.render(quadScene, quadCam);
  }

  function clearTo(r, g, b, a) {
    renderer.setClearColor(_clearColor.setRGB(r, g, b), a);
  }

  /* ---- per-frame prep ------------------------------------------------ */
  function sampleWaterMotion() {
    if (!state.adaptiveTemporal) {
      _waterMotionEMA = 0;
      _waterMotionMult = 1;
      return;
    }
    const n = sim.count | 0;
    const ke = sim.kineticEnergy;
    if (n > 0 && Number.isFinite(ke)) {
      _waterMotionEMA += (Math.sqrt(2 * ke / n) - _waterMotionEMA) * 0.1;
    }
    // map rms speed -> multiplier: <=0.5 m/s → 1.0 (full alpha), >=3 m/s → 0.3×
    const k = Math.min(1, Math.max(0, (_waterMotionEMA - 0.5) / 2.5));
    _waterMotionMult = 1 - 0.7 * k * k * (3 - 2 * k);
  }

  /** Follow drawing-buffer resizes: internal RTs stay at resolutionScale. */
  function syncSize() {
    renderer.getDrawingBufferSize(_bufNow);
    if (_bufNow.equals(bufSize)) return;
    bufSize.copy(_bufNow);
    HW = Math.max(4, Math.floor(bufSize.width * resolutionScale));
    HH = Math.max(4, Math.floor(bufSize.height * resolutionScale));
    for (const rt of [rtDepth, rtTemp, rtSmoothA, rtSmoothB, rtNormal, rtThick]) rt.setSize(HW, HH);
    for (const m of [blurMat, normalMat, compMat]) m.uniforms.uTexel.value.set(1 / HW, 1 / HH);
    compMat.uniforms.uAspect.value = Math.max(1e-3, bufSize.width / Math.max(1, bufSize.height));
    _historyStale = true; // history RT was reallocated: skip one temporal blend
  }

  function update(camera) {
    if (!state.enabled || !camera) return;
    const cu = compMat.uniforms;
    cu.uNear.value = camera.near;
    cu.uFar.value = camera.far;

    // Guard: projectionMatrix/matrixWorldInverse are null until the renderer
    // first updates this camera (scene.updateMatrixWorld). Early composite
    // calls (or a fresh camera swap) would otherwise throw
    // "Cannot read properties of null (reading 'elements')" every frame.
    syncSize();
    const pm = camera.projectionMatrix;
    if (pm?.elements) {
      const e = pm.elements;
      commonPointUniforms.uProj11.value = e[5];
      normalMat.uniforms.uInvProj.value.set(1 / e[0], 1 / e[5]);
      cu.uInvProj.value.set(1 / e[0], 1 / e[5]);
      blurMat.uniforms.uSigmaR.value = state.blurSigma * commonPointUniforms.uRadius.value;
      // sprite footprint radius in RT texels at view depth 1 m
      blurMat.uniforms.uRadiusPx.value = commonPointUniforms.uRadius.value *
        commonPointUniforms.uSpriteScale.value * e[5] * HH * 0.5;
    }
    commonPointUniforms.uViewportH.value = HH;

    // light direction -> view space (fall back to world dir until inverse exists)
    const m = camera.matrixWorldInverse ?? camera.matrixWorld;
    if (m) {
      cu.uLightDirVS.value
        .set(lightDir[0], lightDir[1], lightDir[2])
        .normalize()
        .transformDirection(m);

      // flow direction: GUI angle -> world XZ -> view space xy. The screen-
      // space height field lives in view-space xy/z, so only the in-plane
      // (xy) components drive the wave advection.
      const ang = (state.flowAngleDeg ?? 0) * Math.PI / 180;
      _flowWorld.set(Math.cos(ang), 0, Math.sin(ang)).transformDirection(m);
      cu.uFlowDirVS.value.set(_flowWorld.x, _flowWorld.y);
    }

    // accumulate wall-clock time for advected detail waves; clamp dt so a
    // backgrounded tab does not jump the waves forward on return. The same
    // clamped delta doubles as the hitch signal for spec anti-shimmer.
    const now = performance.now() * 0.001;
    let frameDt = 1 / 60;
    if (_lastNow > 0) {
      frameDt = Math.min(Math.max(now - _lastNow, 0.0), 0.05);
      cu.uTime.value += frameDt;
    }
    _lastNow = now;
    cu.uFrameDt.value = frameDt;

    // velocity-adaptive temporal alpha (water-motion estimate)
    sampleWaterMotion();
  }

  /** Copy sim positions into the GPU attribute (no allocation). */
  function uploadPositions() {
    const n = sim.count | 0;
    if (n <= 0) return;
    // grow the attribute if the sim was resized beyond initial capacity
    const need = sim.pos.length;
    if (need > posAttr.array.length) {
      posAttr.array = new Float32Array(need);
      posAttr.needsUpdate = true;
    }
    posAttr.array.set(n * 3 <= posAttr.array.length
      ? sim.pos.subarray(0, n * 3)
      : sim.pos.subarray(0, posAttr.array.length));
    posAttr.needsUpdate = true;
    pointsGeo.setDrawRange(0, Math.min(n, capacity === 0 ? n : posAttr.array.length / 3 | 0));
  }

  /* ---- full pipeline -------------------------------------------------- */
  function renderWater(camera) {
    if (!state.enabled || !sim.count) return;
    if (colorTarget) compMat.uniforms.tScene.value = colorTarget.texture;
    if (depthTexture) compMat.uniforms.tSceneDepth.value = depthTexture;
    if (camera) update(camera);

    // DEBUG: force composite to draw an opaque red tint everywhere water depth
    // exists at all — isolates "points not rasterized" from "shading math wrong"
    if (window.__waterDebugTint) {
      blit(compMat, null);
      return;
    }

    const r = renderer;
    _prevAutoClear = r.autoClear;
    r.getClearColor(_prevClear);
    _prevClearAlpha = r.getClearAlpha();
    _prevRT.value = r.getRenderTarget();
    r.autoClear = false;

    try {
      uploadPositions();

      // 1. particle front-cap depth (MAX blending) + thickness (additive)
      pointsScene.children.forEach((p) => { p.visible = false; });
      depthPoints.visible = true;
      clearTo(0, 0, 0, 1); // empty sentinel is 0 (see DEPTH_FS note); MAX blend fills real depths
      r.setRenderTarget(rtDepth);
      r.clear(true, false, false);
      r.render(pointsScene, camera);

      thickPoints.visible = true;
      depthPoints.visible = false;
      clearTo(0, 0, 0, 1);
      r.setRenderTarget(rtThick);
      r.clear(true, false, false);
      r.render(pointsScene, camera);
      thickPoints.visible = false;

      // capture per-pass stats for debugProbe (renderer.info with autoReset
      // resets after each .render(), so read right after the depth pass)
      _dbg.pointsCalls = r.info.render.calls;
      _dbg.pointsPoints = r.info.render.points;

      // 2. separable bilateral smoothing (H then V). The V-pass blends with
      //    last frame's smoothed depth (temporal anti-jitter); the blend
      //    weight drops to 0 for one frame after a camera jump so the
      //    history never smears across a cut.
      let tAlpha = state.temporalAlpha ?? 0;
      if (camera) {
        if (_haveCamPose) {
          // Combined screen-space motion estimate: translation + the arc the
          // scene sweeps when the camera rotates about its pivot. A slow pan
          // or orbit can keep |Δpos| tiny while rotation slides every water
          // pixel far from where last frame's history was sampled — so the
          // rotational sweep must count toward the reset threshold too.
          const posDelta = _camPos.distanceTo(camera.position);
          let motion = posDelta;
          if (camera.quaternion) {
            const rotRad = _camQuat.angleTo(camera.quaternion); // 0..π
            // pivot: explicit orbit target if the app provides one, else the
            // world origin (typical orbit-controls focus). Only used to scale
            // the sweep — a rough estimate is fine.
            const tgt = camera.target;
            if (tgt?.isVector3) _pivot.copy(tgt);
            else if (tgt?.position) _pivot.copy(tgt.position);
            else _pivot.set(0, 0, 0);
            motion += rotRad * camera.position.distanceTo(_pivot);
          }
          if (motion >= CAM_JUMP_DIST) {
            tAlpha = 0;                       // hard reset: never smear across a cut
          } else {
            // smooth falloff: fast-ish motion blends less history, proportionally
            tAlpha *= Math.max(0, 1 - motion / CAM_JUMP_DIST);
          }
        }
        _camPos.copy(camera.position);
        if (camera.quaternion) _camQuat.copy(camera.quaternion);
        _haveCamPose = true;
      }
      // Velocity-adaptive factor from WATER motion (sim rms speed).
      // Composes with the camera-motion falloff above — a camera jump still
      // hard-resets to 0; fast splashes without camera input cut the blend.
      tAlpha *= _waterMotionMult;
      if (_historyStale) { tAlpha = 0; _historyStale = false; }
      // H-pass reads the raw 1/z depth; no temporal blend (tPrev unbound
      // and alpha 0 for this pass).
      blurMat.uniforms.uTex.value = rtDepth.texture;
      blurMat.uniforms.uRecip.value = 1;
      blurMat.uniforms.uTemporalAlpha.value = 0;
      blurMat.uniforms.tPrev.value = null;
      blurMat.uniforms.uDir.value.set(1, 0);
      blit(blurMat, rtTemp);
      blurMat.uniforms.uTex.value = rtTemp.texture;
      blurMat.uniforms.uRecip.value = 0;
      blurMat.uniforms.uTemporalAlpha.value = tAlpha;
      blurMat.uniforms.tPrev.value = rtSmoothHist.texture;
      blurMat.uniforms.uDir.value.set(0, 1);
      blit(blurMat, rtSmoothOut);
      // ping-pong: what we just wrote becomes next frame's history
      const _swap = rtSmoothOut; rtSmoothOut = rtSmoothHist; rtSmoothHist = _swap;

      // 3. view-space normals from smoothed depth
      normalMat.uniforms.uTex.value = rtSmoothOut.texture;
      blit(normalMat, rtNormal);

      // 4. composite over the backdrop already on the canvas
      compMat.uniforms.tWaterDepth.value = rtSmoothOut.texture;
      blit(compMat, null);
    } finally {
      r.setRenderTarget(_prevRT.value);
      r.setClearColor(_prevClear, _prevClearAlpha);
      r.autoClear = _prevAutoClear;
    }
  }

  /* ---- gui ------------------------------------------------------------ */
  function addGui(gui) {
    if (!gui || typeof gui.addFolder !== 'function') return null;
    const f = gui.addFolder('Screen Fluid');
    f.add(state, 'enabled').name('enabled');
    f.add(commonPointUniforms.uRadius, 'value', 0.02, 0.5, 0.005).name('radius');
    f.add(commonPointUniforms.uSpriteScale, 'value', 1, 3, 0.05).name('sprite scale');
    f.add(compMat.uniforms.uRefract, 'value', 0, 0.15, 0.005).name('refract');
    f.add(compMat.uniforms.uReflect, 'value', 0, 0.15, 0.005).name('reflect');
    f.add(compMat.uniforms.uSpecular, 'value', 0, 3, 0.05).name('specular');
    f.add(compMat.uniforms.uShininess, 'value', 8, 400, 1).name('shininess');
    f.add(compMat.uniforms.uEdgeSoft, 'value', 0.005, 0.5, 0.005).name('edge soft');
    f.add(compMat.uniforms.uThickScale, 'value', 0.01, 1, 0.01).name('thick scale');
    f.add(compMat.uniforms.uThinCut, 'value', 0, 2, 0.01).name('thin cutoff');
    f.add(state, 'blurSigma', 0.1, 3, 0.05).name('blur sigma ×r');
    f.add(state, 'temporalAlpha', 0, 0.7, 0.01).name('temporal smooth');
    f.add(state, 'flowAngleDeg', 0, 360, 1).name('flow dir angle');
    f.add(compMat.uniforms.uFlowSpeed, 'value', 0, 4, 0.05).name('flow speed');
    f.add(compMat.uniforms.uFlowDetail, 'value', 0, 1.5, 0.01).name('detail strength');
    f.add(compMat.uniforms.uSpecStretch, 'value', 0.05, 1, 0.01).name('spec stretch');
    f.add(compMat.uniforms.uTintDepth, 'value', 0.2, 4, 0.05).name('tint depth');
    f.add(compMat.uniforms.uShallowClarity, 'value', 0, 1, 0.01).name('shallow clarity');
    const shallowHex = { hex: 0x78855e }; // ≈ shallowColor [0.47, 0.52, 0.37]
    f.addColor(shallowHex, 'hex').name('shallow color').onChange((v) => {
      const num = typeof v === 'number' ? v : parseInt(String(v).replace('#', ''), 16);
      if (!Number.isNaN(num)) compMat.uniforms.uShallowColor.value.setHex(num);
    });
    f.add(compMat.uniforms.uFoamEdge, 'value', 0, 1, 0.01).name('foam edge');
    f.add(compMat.uniforms.uFoamDamp, 'value', 0, 1, 0.01).name('rim damp');
    f.add(compMat.uniforms.uFoamBreak, 'value', 0, 1, 0.01).name('foam breakup');
    f.add(compMat.uniforms.uFoamStreaks, 'value', 0, 1, 0.01).name('foam streaks');
    f.add(compMat.uniforms.uShimmer, 'value', 0, 1, 0.01).name('shimmer');
    f.add(compMat.uniforms.uFoamSurface, 'value', 0, 1, 0.01).name('surface foam');
    f.add(compMat.uniforms.uRippleDetail, 'value', 0, 2, 0.01).name('ripple detail');
    f.add(compMat.uniforms.uSheen, 'value', 0, 1, 0.01).name('spec sheen');
    f.add(compMat.uniforms.uCaustics, 'value', 0, 1, 0.01).name('caustics');
    // --- Lane D shader polish ---
    f.add(state, 'adaptiveTemporal').name('temporal adapt');
    f.add(compMat.uniforms.uFoamSoft, 'value', 0, 1, 0.01).name('foam soften');
    f.add(compMat.uniforms.uSpecSteady, 'value', 0, 1, 0.01).name('spec steady');
    f.add(compMat.uniforms.uLobeWiden, 'value', 0.5, 1, 0.01).name('spec lobe widen');
    f.close();
    return f;
  }

  /* ---- teardown -------------------------------------------------------- */
  function dispose() {
    for (const rt of [rtDepth, rtTemp, rtSmoothA, rtSmoothB, rtNormal, rtThick]) rt.dispose();
    quadGeo.dispose();
    pointsGeo.dispose();
    for (const m of [depthMat, thickMat, blurMat, normalMat, compMat]) m.dispose();
    pointsScene.clear();
    quadScene.clear();
  }

  /* ---- debug probe (used by agent diagnostics; harmless in prod) ------ */
  function debugProbe() {
    try {
      const r = renderer;
      const HWr = rtDepth.width, HHr = rtDepth.height;
    const depthPx = new Float32Array(4);
    // sample center + a few spread points of the smoothed depth RT
    // (rtSmoothOut = latest ping-pong output)
    const samples = [];
    for (const [fx, fy] of [[0.5, 0.5], [0.45, 0.45], [0.55, 0.55], [0.5, 0.6], [0.25, 0.5]]) {
      r.readRenderTargetPixels(rtSmoothOut, Math.floor(fx * (HWr - 1)), Math.floor((1 - fy) * (HHr - 1)), 1, 1, depthPx);
      samples.push(+depthPx[0].toFixed(2));
    }
    const thickPx = new Float32Array(4);
    r.readRenderTargetPixels(rtThick, HWr >> 1, HHr >> 1, 1, 1, thickPx);
    // raw depth RT samples (pre-blur) — distinguishes "points never wrote"
    // from "blur destroyed the data"
    const rawDepth = [];
    for (const [fx, fy] of [[0.5, 0.5], [0.45, 0.45], [0.55, 0.55], [0.02, 0.02]]) {
      r.readRenderTargetPixels(rtDepth, Math.floor(fx * (HWr - 1)), Math.floor((1 - fy) * (HHr - 1)), 1, 1, depthPx);
      rawDepth.push(depthPx[0] > 0 ? +(1 / depthPx[0]).toFixed(2) : 0); // stored as 1/z
    }
    return {
      size: [HWr, HHr],
      smoothDepth: samples,          // view depth z; 0 means "no water written"
      rawDepth,
      thicknessCenter: +thickPx[0].toFixed(4),
      drawRange: pointsGeo.drawRange.count,
      pointsCalls: _dbg.pointsCalls,
      pointsPoints: _dbg.pointsPoints,
      uProj11: commonPointUniforms.uProj11.value,
      uViewportH: commonPointUniforms.uViewportH.value,
      radius: commonPointUniforms.uRadius.value,
      // adaptive-temporal diagnostics (rms particle speed, m/s)
      waterMotionEMA: +_waterMotionEMA.toFixed(4),
      temporalMult: +_waterMotionMult.toFixed(3),
    };
    } catch (e) {
      return { error: String(e?.message ?? e) };
    }
  }

  return { state, update, renderWater, addGui, dispose, debugProbe,
           get _compMat() { return compMat; },
           get _pointUniforms() { return commonPointUniforms; } };
}

export default createScreenFluid;
