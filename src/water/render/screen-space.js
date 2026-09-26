// water/render/screen-space.js — screen-space fluid rendering for the water pack.
//
// Surface from particles, no meshing (van der Laan et al. 2009):
//   1. depth splat    particles as point sprites; each writes 1/z of its
//                     sphere cap with MAX blending → nearest surface per pixel
//   2. thickness      additive gaussian splats, converted to metres of water
//   3. smoothing      separable bilateral blur of view depth (footprint and
//                     range scale with the particle size; empty texels
//                     excluded) + a light temporal blend
//   4. normals        from view-space positions, one-sided at silhouettes
//   5. composite      one full-screen pass over the scene colour: Fresnel
//                     (Schlick) mix of environment reflection and refracted
//                     scene, Beer–Lambert absorption through the measured
//                     thickness, sun specular, occlusion against the scene
//                     depth buffer, anti-aliased silhouettes; applies the
//                     renderer's tone mapping and output colour space.
// Steps 1–4 run at `resolution` × the drawing buffer (default 0.5).
//
//   const r = createScreenSpaceRenderer({ renderer, particleRadius, spacing });
//   r.setParticles(positions, count);      // world positions (interpolated)
//   r.render(scene, camera, target?);      // replaces renderer.render(scene, camera)
//
// Perspective cameras only.

import * as THREE from 'three';

/* ------------------------------ shaders ------------------------------ */

const QUAD_VS = /* glsl */`
precision highp float;
attribute vec3 position;
varying vec2 vUv;
void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}`;

const POINTS_VS = /* glsl */`
precision highp float;
attribute vec3 position;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform float uRadius;      // depth relief radius (m)
uniform float uFootprint;   // sprite footprint radius (m)
uniform float uProj11;
uniform float uViewportH;   // height (px) of the target the sprites render into
varying float vDist;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vDist = max(0.05, -mv.z);
  gl_PointSize = clamp(uFootprint * uProj11 * uViewportH / vDist, 1.0, 512.0);
  gl_Position = projectionMatrix * mv;
}`;

// 1/z of the sphere cap in front; MAX blending keeps the nearest surface.
const DEPTH_FS = /* glsl */`
precision highp float;
uniform float uRadius;
varying float vDist;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(p, p);
  if (r2 > 1.0) discard;
  gl_FragColor = vec4(1.0 / max(0.05, vDist - uRadius * sqrt(1.0 - r2)), 0.0, 0.0, 1.0);
}`;

const THICKNESS_FS = /* glsl */`
precision highp float;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(p, p);
  if (r2 > 1.0) discard;
  gl_FragColor = vec4(exp(-3.0 * r2) * sqrt(1.0 - r2), 0.0, 0.0, 1.0);
}`;

const BLUR_FS = /* glsl */`
precision highp float;
uniform sampler2D uTex;
uniform sampler2D tPrev;
uniform float uTemporal;
uniform vec2 uTexel;
uniform vec2 uDir;
uniform float uSigmaR;     // depth range sigma (m)
uniform float uRecip;      // 1: input holds 1/z
uniform float uRadiusPx;   // footprint radius in texels at z = 1 m
varying vec2 vUv;
float readZ(vec2 uv) {
  float v = texture2D(uTex, uv).r;
  return uRecip > 0.5 ? (v > 0.0 ? 1.0 / v : 0.0) : v;
}
void main() {
  float d0 = readZ(vUv);
  if (d0 <= 0.0) { gl_FragColor = vec4(0.0); return; }
  float stp = floor(clamp(uRadiusPx / d0 * 0.3, 1.0, 4.0));
  float w[5];
  w[0] = 0.227027; w[1] = 0.194594; w[2] = 0.121621; w[3] = 0.054054; w[4] = 0.016216;
  float inv2s2 = 1.0 / (2.0 * uSigmaR * uSigmaR);
  float sum = d0 * w[0], wsum = w[0];
  for (int i = 1; i <= 4; i++) {
    vec2 off = uDir * uTexel * float(i) * stp;
    float a = readZ(vUv + off), b = readZ(vUv - off);
    float wa = a > 0.0 ? w[i] * exp(-(a - d0) * (a - d0) * inv2s2) : 0.0;
    float wb = b > 0.0 ? w[i] * exp(-(b - d0) * (b - d0) * inv2s2) : 0.0;
    sum += a * wa + b * wb; wsum += wa + wb;
  }
  float d = sum / wsum;
  float pv = texture2D(tPrev, vUv).r;
  pv = (pv > 0.0 && abs(pv - d) < 4.0 * uSigmaR) ? pv : d;
  gl_FragColor = vec4(mix(d, pv, uTemporal), 0.0, 0.0, 1.0);
}`;

const NORMAL_FS = /* glsl */`
precision highp float;
uniform sampler2D uTex;
uniform vec2 uTexel;
uniform vec2 uInvProj;
varying vec2 vUv;
vec3 viewPos(vec2 uv, float z) { return vec3((uv * 2.0 - 1.0) * uInvProj * z, -z); }
void main() {
  float z0 = texture2D(uTex, vUv).r;
  if (z0 <= 0.0) { gl_FragColor = vec4(0.0, 0.0, 1.0, 0.0); return; }
  vec2 ox = vec2(uTexel.x, 0.0), oy = vec2(0.0, uTexel.y);
  float zl = texture2D(uTex, vUv - ox).r, zr = texture2D(uTex, vUv + ox).r;
  float zb = texture2D(uTex, vUv - oy).r, zt = texture2D(uTex, vUv + oy).r;
  vec3 p0 = viewPos(vUv, z0);
  vec3 dx = vec3(2.0 * uTexel.x * uInvProj.x * z0, 0.0, 0.0);
  if (zr > 0.0 && (zl <= 0.0 || abs(zr - z0) < abs(z0 - zl))) dx = viewPos(vUv + ox, zr) - p0;
  else if (zl > 0.0) dx = p0 - viewPos(vUv - ox, zl);
  vec3 dy = vec3(0.0, 2.0 * uTexel.y * uInvProj.y * z0, 0.0);
  if (zt > 0.0 && (zb <= 0.0 || abs(zt - z0) < abs(z0 - zb))) dy = viewPos(vUv + oy, zt) - p0;
  else if (zb > 0.0) dy = p0 - viewPos(vUv - oy, zb);
  gl_FragColor = vec4(normalize(cross(dx, dy)), 1.0);
}`;

// Composite (ShaderMaterial: three prepends tone mapping + colour space code).
const COMPOSITE_VS = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}`;

const COMPOSITE_FS = /* glsl */`
uniform sampler2D tScene;
uniform sampler2D tSceneDepth;
uniform sampler2D tDepth;       // smoothed view depth (0 = no water)
uniform sampler2D tNormal;
uniform sampler2D tThick;
uniform vec2 uTexel;            // water RT texel
uniform vec2 uInvProj;
uniform float uNear, uFar;
uniform mat3 uViewToWorld;
uniform float uThickScale;      // raw thickness → metres
uniform vec3 uAbsorption;       // 1/m per channel
uniform vec3 uScatterColor;     // in-scattered colour (turbid water)
uniform float uScatter;         // 0 clear … 1 opaque turbid
uniform float uRefraction;      // screen-space refraction strength
uniform float uF0;
uniform vec3 uSkyTop, uSkyHorizon;
uniform float uEnvIntensity;
uniform float uRoughness;
uniform vec3 uSunDirView;       // towards the light, view space
uniform vec3 uSunColor;         // colour × intensity
uniform float uEdgeSoft;
uniform float uThinCut;
#ifdef ENVMAP_TYPE_CUBE_UV
uniform sampler2D envMap;
#include <cube_uv_reflection_fragment>
#endif
varying vec2 vUv;

float sceneDepth(vec2 uv) {
  float z = texture2D(tSceneDepth, uv).x;
  if (z >= 1.0) return 1e6;
  float ndc = z * 2.0 - 1.0;
  return (2.0 * uNear * uFar) / (uFar + uNear - ndc * (uFar - uNear));
}

void tapWater(vec2 uv, float w, inout float z, inout vec3 n, inout float cov) {
  float d = texture2D(tDepth, uv).r;
  float k = d > 0.0 ? w : 0.0;
  z += d * k; n += texture2D(tNormal, uv).xyz * k; cov += k;
}

vec3 environment(vec3 dirWorld) {
#ifdef ENVMAP_TYPE_CUBE_UV
  return textureCubeUV(envMap, dirWorld, uRoughness).rgb * uEnvIntensity;
#else
  float t = clamp(dirWorld.y * 0.5 + 0.5, 0.0, 1.0);
  return mix(uSkyHorizon, uSkyTop, t) * uEnvIntensity;
#endif
}

void main() {
  vec3 scene = texture2D(tScene, vUv).rgb;
  // validity-weighted bilinear upsample of depth + normal
  vec2 st = vUv / uTexel - 0.5;
  vec2 f = fract(st);
  vec2 uv0 = (floor(st) + 0.5) * uTexel;
  float wz = 0.0, cov = 0.0; vec3 nsum = vec3(0.0);
  tapWater(uv0, (1.0 - f.x) * (1.0 - f.y), wz, nsum, cov);
  tapWater(uv0 + vec2(uTexel.x, 0.0), f.x * (1.0 - f.y), wz, nsum, cov);
  tapWater(uv0 + vec2(0.0, uTexel.y), (1.0 - f.x) * f.y, wz, nsum, cov);
  tapWater(uv0 + uTexel, f.x * f.y, wz, nsum, cov);
  vec3 color = scene;
  if (cov > 0.02) {
    wz /= cov;
    vec3 n = normalize(nsum);
    vec3 V = -normalize(vec3((vUv * 2.0 - 1.0) * uInvProj, -1.0));
    float sd = sceneDepth(vUv);
    float thickRaw = texture2D(tThick, vUv).r;
    // metres of water along the view ray: splat thickness, but never more
    // than the distance to the geometry behind the water surface
    float L = min(thickRaw * uThickScale, max(sd - wz, 0.0));
    // refraction: bend the view ray by the surface normal, scaled by depth
    vec2 ruv = clamp(vUv + n.xy * uRefraction * min(L, 1.0), vec2(0.002), vec2(0.998));
    // do not pull colour from geometry in FRONT of the water
    if (sceneDepth(ruv) < wz) ruv = vUv;
    vec3 T = exp(-uAbsorption * L);
    vec3 below = texture2D(tScene, ruv).rgb * T;
    below = mix(below, uScatterColor, (1.0 - (T.r + T.g + T.b) / 3.0) * uScatter);
    float cosT = clamp(dot(n, V), 0.0, 1.0);
    float fres = uF0 + (1.0 - uF0) * pow(1.0 - cosT, 5.0);
    vec3 R = uViewToWorld * reflect(-V, n);
    vec3 refl = environment(normalize(R));
    vec3 H = normalize(uSunDirView + V);
    float shin = 2.0 / max(uRoughness * uRoughness, 1e-4) - 2.0;
    float spec = pow(max(dot(n, H), 0.0), shin) * (shin + 8.0) / 25.13;
    vec3 water = mix(below, refl, fres) + uSunColor * spec * fres * max(dot(n, uSunDirView), 0.0);
    float occl = clamp((sd - wz) / (0.03 + 0.004 * wz) + 1.0, 0.0, 1.0);
    float edge = max(smoothstep(0.0, uEdgeSoft, sd - wz), 0.85);
    float solid = smoothstep(0.0, uThinCut, thickRaw);
    float a = occl * edge * solid * smoothstep(0.02, 0.5, cov);
    color = mix(scene, water, a);
  }
  gl_FragColor = vec4(color, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

/* ------------------------------ renderer ------------------------------ */

const DEFAULT_LOOK = {
  resolution: 0.5,
  footprintScale: 2.0,       // sprite footprint = footprintScale × particle radius
  blurSigma: 1.0,            // depth range sigma, × particle radius
  temporal: 0.15,
  absorption: [0.45, 0.09, 0.06], // pure water, 1/m (red absorbed first)
  scatterColor: [0.02, 0.12, 0.14],
  scatter: 0.35,
  refraction: 0.04,
  f0: 0.02,
  roughness: 0.08,
  envIntensity: 1.0,
  skyTop: [0.35, 0.55, 0.9],
  skyHorizon: [0.75, 0.82, 0.9],
  sunIntensity: 1.0,
  edgeSoft: 0.06,
  thinCut: 0.35,
};

/**
 * @param {object} o
 *   renderer        THREE.WebGLRenderer
 *   particleRadius  depth-relief radius per particle (m)
 *   spacing         particle spacing (m) — calibrates thickness to metres
 *   capacity        max particles
 *   look            DEFAULT_LOOK overrides
 */
export function createScreenSpaceRenderer({ renderer, particleRadius, spacing, capacity, look = {} }) {
  const L = { ...DEFAULT_LOOK, ...look };
  const size = renderer.getDrawingBufferSize(new THREE.Vector2());
  const cur = new THREE.Vector2();
  let W = 1, Hh = 1;

  const halfOpts = {
    type: THREE.HalfFloatType, format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
    depthBuffer: false, stencilBuffer: false, generateMipmaps: false,
  };
  const rtDepth = new THREE.WebGLRenderTarget(1, 1, halfOpts);
  const rtThick = new THREE.WebGLRenderTarget(1, 1, halfOpts);
  const rtTemp = new THREE.WebGLRenderTarget(1, 1, halfOpts);
  let rtSmooth = new THREE.WebGLRenderTarget(1, 1, halfOpts);
  let rtHist = new THREE.WebGLRenderTarget(1, 1, halfOpts);
  const rtNormal = new THREE.WebGLRenderTarget(1, 1, halfOpts);
  // full-resolution linear HDR scene colour + depth
  const sceneRT = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, depthBuffer: true, samples: 0 });
  sceneRT.depthTexture = new THREE.DepthTexture(1, 1);

  const footprint = particleRadius * L.footprintScale;
  const pointUniforms = {
    uRadius: { value: particleRadius },
    uFootprint: { value: footprint },
    uProj11: { value: 1 },
    uViewportH: { value: 1 },
  };
  const positions = new THREE.BufferAttribute(new Float32Array(capacity * 3), 3);
  positions.setUsage(THREE.DynamicDrawUsage);
  const pointsGeo = new THREE.BufferGeometry();
  pointsGeo.setAttribute('position', positions);
  pointsGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Infinity);
  const pointMat = (fs, blending) => new THREE.RawShaderMaterial({
    vertexShader: POINTS_VS, fragmentShader: fs, uniforms: pointUniforms,
    blending: THREE.CustomBlending, blendEquation: blending, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
    depthTest: false, depthWrite: false, transparent: true,
  });
  const depthPoints = new THREE.Points(pointsGeo, pointMat(DEPTH_FS, THREE.MaxEquation));
  const thickPoints = new THREE.Points(pointsGeo, pointMat(THICKNESS_FS, THREE.AddEquation));
  depthPoints.frustumCulled = thickPoints.frustumCulled = false;
  const pointScene = new THREE.Scene();
  pointScene.add(depthPoints, thickPoints);

  const texel = new THREE.Vector2();
  const invProj = new THREE.Vector2();
  const blurMat = new THREE.RawShaderMaterial({
    vertexShader: QUAD_VS, fragmentShader: BLUR_FS, depthTest: false, depthWrite: false,
    uniforms: {
      uTex: { value: null }, tPrev: { value: null }, uTemporal: { value: 0 }, uTexel: { value: texel },
      uDir: { value: new THREE.Vector2() }, uSigmaR: { value: L.blurSigma * particleRadius },
      uRecip: { value: 1 }, uRadiusPx: { value: 1 },
    },
  });
  const normalMat = new THREE.RawShaderMaterial({
    vertexShader: QUAD_VS, fragmentShader: NORMAL_FS, depthTest: false, depthWrite: false,
    uniforms: { uTex: { value: null }, uTexel: { value: texel }, uInvProj: { value: invProj } },
  });
  // thickness calibration: a slab L metres thick accumulates
  // L/s³ · π·F² · w̄ of splat weight (F = footprint, w̄ = mean splat weight)
  const meanSplat = 0.2656; // ∫ e^{-3r²}·√(1−r²)·2r dr over the unit disk
  const thickScale = (spacing ** 3) / (meanSplat * Math.PI * footprint * footprint);
  const compMat = new THREE.ShaderMaterial({
    vertexShader: COMPOSITE_VS, fragmentShader: COMPOSITE_FS, depthTest: false, depthWrite: false,
    uniforms: {
      tScene: { value: sceneRT.texture }, tSceneDepth: { value: sceneRT.depthTexture },
      tDepth: { value: null }, tNormal: { value: rtNormal.texture }, tThick: { value: rtThick.texture },
      uTexel: { value: texel }, uInvProj: { value: invProj }, uNear: { value: 0.1 }, uFar: { value: 1000 },
      uViewToWorld: { value: new THREE.Matrix3() },
      uThickScale: { value: thickScale },
      uAbsorption: { value: new THREE.Vector3(...L.absorption) },
      uScatterColor: { value: new THREE.Color(...L.scatterColor) },
      uScatter: { value: L.scatter }, uRefraction: { value: L.refraction }, uF0: { value: L.f0 },
      uSkyTop: { value: new THREE.Color(...L.skyTop) }, uSkyHorizon: { value: new THREE.Color(...L.skyHorizon) },
      uEnvIntensity: { value: L.envIntensity }, uRoughness: { value: L.roughness },
      uSunDirView: { value: new THREE.Vector3(0, 1, 0) }, uSunColor: { value: new THREE.Color(0, 0, 0) },
      uEdgeSoft: { value: L.edgeSoft }, uThinCut: { value: L.thinCut },
      envMap: { value: null },
    },
  });
  compMat.toneMapped = true;

  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
  quad.frustumCulled = false;
  const quadScene = new THREE.Scene();
  quadScene.add(quad);
  const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  let count = 0;
  let envSource = null, envTexture = null, pmrem = null, pmremTarget = null;
  const clear = new THREE.Color();
  const prevClear = new THREE.Color();
  const _sun = new THREE.Vector3(), _sunTarget = new THREE.Vector3();
  let haveHistory = false;
  const lastCam = new THREE.Matrix4();

  function resize() {
    renderer.getDrawingBufferSize(cur);
    if (cur.equals(size) && W > 1) return;
    size.copy(cur);
    W = Math.max(4, Math.floor(size.x * L.resolution));
    Hh = Math.max(4, Math.floor(size.y * L.resolution));
    for (const rt of [rtDepth, rtThick, rtTemp, rtSmooth, rtHist, rtNormal]) rt.setSize(W, Hh);
    sceneRT.setSize(size.x, size.y);
    texel.set(1 / W, 1 / Hh);
    haveHistory = false;
  }

  // scene.environment → PMREM (cube-UV) texture for rough reflections
  function syncEnvironment(scene) {
    const src = scene.environment ?? null;
    if (src === envSource) return;
    envSource = src;
    pmremTarget?.dispose(); pmremTarget = null;
    envTexture = null;
    if (src) {
      if (src.mapping === THREE.CubeUVReflectionMapping) envTexture = src;
      else {
        pmrem ??= new THREE.PMREMGenerator(renderer);
        pmremTarget = src.isCubeTexture ? pmrem.fromCubemap(src) : pmrem.fromEquirectangular(src);
        envTexture = pmremTarget.texture;
      }
    }
    compMat.uniforms.envMap.value = envTexture;
    for (const k of ['ENVMAP_TYPE_CUBE_UV', 'CUBEUV_TEXEL_WIDTH', 'CUBEUV_TEXEL_HEIGHT', 'CUBEUV_MAX_MIP']) delete compMat.defines[k];
    if (envTexture) {
      // same constants three derives for its own materials (WebGLProgram)
      const h = envTexture.image.height;
      const maxMip = Math.log2(h) - 2;
      compMat.defines.ENVMAP_TYPE_CUBE_UV = '';
      compMat.defines.CUBEUV_TEXEL_WIDTH = 1 / (3 * Math.max(2 ** maxMip, 7 * 16));
      compMat.defines.CUBEUV_TEXEL_HEIGHT = 1 / h;
      compMat.defines.CUBEUV_MAX_MIP = `${maxMip}.0`;
    }
    compMat.needsUpdate = true;
  }

  function syncSun(scene, camera) {
    let light = null;
    scene.traverseVisible((o) => { if (!light && o.isDirectionalLight) light = o; });
    const u = compMat.uniforms;
    if (!light) { u.uSunColor.value.setRGB(0, 0, 0); return; }
    light.getWorldPosition(_sun);
    light.target.getWorldPosition(_sunTarget);
    u.uSunDirView.value.copy(_sun.sub(_sunTarget).normalize()).transformDirection(camera.matrixWorldInverse);
    u.uSunColor.value.copy(light.color).multiplyScalar(light.intensity * L.sunIntensity);
  }

  function blit(material, target) {
    quad.material = material;
    renderer.setRenderTarget(target);
    renderer.render(quadScene, quadCam);
  }

  function setLook(patch) {
    Object.assign(L, patch);
    const u = compMat.uniforms;
    u.uAbsorption.value.set(...L.absorption);
    u.uScatterColor.value.setRGB(...L.scatterColor);
    u.uSkyTop.value.setRGB(...L.skyTop);
    u.uSkyHorizon.value.setRGB(...L.skyHorizon);
    u.uScatter.value = L.scatter; u.uRefraction.value = L.refraction; u.uF0.value = L.f0;
    u.uRoughness.value = L.roughness; u.uEnvIntensity.value = L.envIntensity;
    u.uEdgeSoft.value = L.edgeSoft; u.uThinCut.value = L.thinCut;
    blurMat.uniforms.uSigmaR.value = L.blurSigma * particleRadius;
    if ('resolution' in patch) W = 1; // re-allocate targets on the next render
  }

  return {
    look: L,
    /** Change look parameters (see DEFAULT_LOOK) at runtime. */
    setLook,
    /** Particle positions for this frame (world space, e.g. interpolated). */
    setParticles(pos, n) {
      count = Math.min(n, capacity);
      positions.array.set(pos.subarray(0, count * 3));
      positions.clearUpdateRanges();
      positions.addUpdateRange(0, count * 3);
      positions.needsUpdate = true;
      pointsGeo.setDrawRange(0, count);
    },
    /** Render `scene` with water to `target` (default: the canvas). */
    render(scene, camera, target = null) {
      resize();
      const prevTarget = renderer.getRenderTarget();
      const prevAuto = renderer.autoClear;
      renderer.getClearColor(prevClear);
      const prevAlpha = renderer.getClearAlpha();
      camera.updateMatrixWorld();
      // scene → linear HDR colour + depth
      renderer.setRenderTarget(sceneRT);
      renderer.render(scene, camera);
      const u = compMat.uniforms;
      const e = camera.projectionMatrix.elements;
      invProj.set(1 / e[0], 1 / e[5]);
      u.uNear.value = camera.near; u.uFar.value = camera.far;
      u.uViewToWorld.value.setFromMatrix4(camera.matrixWorld);
      syncEnvironment(scene);
      syncSun(scene, camera);
      let depthTex = null;
      if (count > 0) {
        renderer.autoClear = false;
        renderer.setClearColor(clear.setRGB(0, 0, 0), 0);
        pointUniforms.uProj11.value = e[5];
        pointUniforms.uViewportH.value = Hh;
        depthPoints.visible = true; thickPoints.visible = false;
        renderer.setRenderTarget(rtDepth); renderer.clear(true, false, false); renderer.render(pointScene, camera);
        depthPoints.visible = false; thickPoints.visible = true;
        renderer.setRenderTarget(rtThick); renderer.clear(true, false, false); renderer.render(pointScene, camera);
        // bilateral smoothing; temporal blend only while the camera is still
        const still = haveHistory && lastCam.equals(camera.matrixWorld);
        lastCam.copy(camera.matrixWorld);
        const bu = blurMat.uniforms;
        bu.uRadiusPx.value = footprint * e[5] * Hh * 0.5;
        bu.uTex.value = rtDepth.texture; bu.uRecip.value = 1; bu.uTemporal.value = 0;
        bu.tPrev.value = rtHist.texture; bu.uDir.value.set(1, 0);
        blit(blurMat, rtTemp);
        bu.uTex.value = rtTemp.texture; bu.uRecip.value = 0; bu.uTemporal.value = still ? L.temporal : 0;
        bu.uDir.value.set(0, 1);
        blit(blurMat, rtSmooth);
        [rtSmooth, rtHist] = [rtHist, rtSmooth]; // rtHist now holds this frame
        haveHistory = true;
        normalMat.uniforms.uTex.value = rtHist.texture;
        blit(normalMat, rtNormal);
        depthTex = rtHist.texture;
      }
      u.tDepth.value = depthTex ?? rtDepth.texture;
      if (!depthTex) { renderer.setRenderTarget(rtDepth); renderer.setClearColor(clear.setRGB(0, 0, 0), 0); renderer.clear(true, false, false); }
      blit(compMat, target);
      renderer.setRenderTarget(prevTarget);
      renderer.autoClear = prevAuto;
      renderer.setClearColor(prevClear, prevAlpha);
    },
    dispose() {
      for (const rt of [rtDepth, rtThick, rtTemp, rtSmooth, rtHist, rtNormal, sceneRT]) rt.dispose();
      pmremTarget?.dispose(); pmrem?.dispose();
      pointsGeo.dispose(); quad.geometry.dispose();
      for (const m of [depthPoints.material, thickPoints.material, blurMat, normalMat, compMat]) m.dispose();
    },
  };
}
