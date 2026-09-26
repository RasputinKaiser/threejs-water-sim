// water-pack/effects.js — render-lane effects for the water pack (Lane C1).
//
// Exports:
//   FoamSystem               — lifetime foam particles on a ring buffer
//   computeCohesionField     — per-particle cluster-tightness proxy in [0,1]
//   makeCapillaryNormalTexture — tileable ripple normal map (DataTexture)
//   makeRippleDecalQuad      — reflective decal quad driven by that texture
//
// Consumed via loadWaterLane('effects') from index.js; FoamSystem usage in
// scenes: `new fx.FoamSystem(sim, budget)` then `foam.update?.(dt, flags)`
// with flags = { maxSpeed, minNeighbors }, and `foam.count` for HUD.
//
// Two-tier flags (B-R2, additive): passing an optional `coreSpeed` promotes
// particles above that speed to CORE foam (long-lived, large whitewater
// lines) while everything else stays short-lived SPRAY. Optional
// `coreBudget` caps core spawns per frame independently (default: half of
// params.spawnPerFrame). Omitting coreSpeed reproduces the legacy
// single-tier behavior exactly.

import * as THREE from 'three';

/* ======================================================================
 * computeCohesionField — O(n·neighbors) cohesion proxy per particle
 * ====================================================================== */

/**
 * For each water particle i: fraction of its grid-neighborhood particles
 * that sit within 0.5h of it, over all particles within h. 1 = tight
 * cluster interior, ~0 = spray/lonely particle. Reads the solver's flat
 * spatial grid directly (cellHead/next, cellSize === h, & (D-1) masking),
 * refreshed first via sim.ensureGrid() when the sim provides it (WaterSim
 * builds that grid lazily; step() leaves it stale).
 *
 * @param {object} sim  WaterSim-shaped object (pos, count, cellHead, next,
 *                      gridDim, cellSize/h, p.maxParticles)
 * @param {Float32Array} [out] optional reused output buffer (length ≥ count)
 * @returns {Float32Array} cohesion per particle in [0,1]
 */
export function computeCohesionField(sim, out = null) {
  sim.ensureGrid?.();
  const n = sim.count;
  if (!out) out = new Float32Array(sim.p.maxParticles);
  const D = sim.gridDim;
  const D2 = D * D;
  const inv = 1 / sim.cellSize;
  const h = sim.h ?? sim.cellSize;
  const h2 = h * h;
  const half2 = h2 * 0.25;
  const p = sim.pos;
  const head = sim.cellHead;
  const next = sim.next;

  for (let i = 0; i < n; i++) {
    const px = p[i * 3], py = p[i * 3 + 1], pz = p[i * 3 + 2];
    const cx = Math.floor(px * inv) & (D - 1);
    const cy = Math.floor(py * inv) & (D - 1);
    const cz = Math.floor(pz * inv) & (D - 1);
    let total = 0, tight = 0;
    for (let gz = -1; gz <= 1; gz++) {
      const zBase = ((cz + gz) & (D - 1)) * D2;
      for (let gy = -1; gy <= 1; gy++) {
        const yBase = ((cy + gy) & (D - 1)) * D + zBase;
        for (let gx = -1; gx <= 1; gx++) {
          let j = head[yBase + ((cx + gx) & (D - 1))];
          while (j !== -1) {
            if (j !== i) {
              const dx = p[j * 3] - px, dy = p[j * 3 + 1] - py, dz = p[j * 3 + 2] - pz;
              const d2 = dx * dx + dy * dy + dz * dz;
              if (d2 <= h2) {
                total++;
                if (d2 <= half2) tight++;
              }
            }
            j = next[j];
          }
        }
      }
    }
    out[i] = total > 0 ? tight / total : 0;
  }
  return out;
}

/* ======================================================================
 * FoamSystem — ring-buffer foam particles with lifetime
 * ====================================================================== */

const FOAM_VERT = /* glsl */`
attribute float aLife;
attribute float aSize;
varying float vLife;
void main() {
  vLife = clamp(aLife, 0.0, 1.0);
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = aSize * (240.0 / max(0.1, -mv.z));
  gl_Position = projectionMatrix * mv;
}`;

const FOAM_FRAG = /* glsl */`
uniform vec3 uColorA;
uniform vec3 uColorB;
varying float vLife;
void main() {
  vec2 uv = gl_PointCoord - 0.5;
  float r2 = dot(uv, uv);
  if (r2 > 0.25) discard;
  float soft = smoothstep(0.25, 0.02, r2);
  vec3 col = mix(uColorB, uColorA, vLife);
  gl_FragColor = vec4(col, soft * vLife * 0.85);
  #include <colorspace_fragment>
}`;

export class FoamSystem {
  /**
   * @param {object} sim     WaterSim-shaped object
   * @param {number} [maxFoam=4000] pool capacity (ring-buffer size)
   * @param {THREE.Object3D} [parent=null] if given, points are added here
   */
  constructor(sim, maxFoam = 4000, parent = null) {
    this.sim = sim;
    this.maxFoam = maxFoam >>> 0 || 4000;
    this.count = 0;

    // tunables surfaced through addGui
    this.params = {
      spawnPerFrame: 120,   // per-frame spawn cap in update() (spray tier)
      riseTime: 0.35,       // buoyant drift phase duration (s)
      buoyancy: 2.2,        // upward accel during rise (m/s²)
      gravity: 6.5,         // reduced gravity after rise (m/s²)
      drag: 1.8,            // linear drag (1/s) after rise
      velInherit: 0.3,      // legacy scalar inherit (kept for compat; unused by
                            // the tangential split below — see tangential/normal)
      tangentialInherit: 0.6, // fraction of FLOW-ALONG-SURFACE (horizontal)
                              // water velocity inherited at spawn
      normalInherit: 0.15,    // fraction of surface-normal (vertical) component
      lifeMin: 0.6,         // random lifetime bounds (s) — spray tier
      lifeMax: 1.5,
      sizeMin: 0.06,
      sizeMax: 0.16,
      coreLifeMin: 1.8,     // lifetime bounds (s) — core whitewater tier
      coreLifeMax: 3.4,
      coreSizeMin: 0.14,
      coreSizeMax: 0.30,
    };

    // preallocated pool state — zero per-frame GC
    this._px = new Float32Array(this.maxFoam);
    this._py = new Float32Array(this.maxFoam);
    this._pz = new Float32Array(this.maxFoam);
    this._vx = new Float32Array(this.maxFoam);
    this._vy = new Float32Array(this.maxFoam);
    this._vz = new Float32Array(this.maxFoam);
    this._age = new Float32Array(this.maxFoam);
    this._life = new Float32Array(this.maxFoam);
    this._size = new Float32Array(this.maxFoam);
    this._type = new Uint8Array(this.maxFoam); // 0 = spray, 1 = core whitewater
    this._write = 0; // ring cursor for recycling

    // GPU buffers
    const g = new THREE.BufferGeometry();
    this._aPos = new THREE.BufferAttribute(new Float32Array(this.maxFoam * 3), 3).setUsage(THREE.DynamicDrawUsage);
    this._aLife = new THREE.BufferAttribute(new Float32Array(this.maxFoam), 1).setUsage(THREE.DynamicDrawUsage);
    this._aSize = new THREE.BufferAttribute(new Float32Array(this.maxFoam), 1).setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('position', this._aPos);
    g.setAttribute('aLife', this._aLife);
    g.setAttribute('aSize', this._aSize);
    g.setDrawRange(0, 0);
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Infinity); // never recompute
    this.geometry = g;

    this.material = new THREE.ShaderMaterial({
      vertexShader: FOAM_VERT,
      fragmentShader: FOAM_FRAG,
      uniforms: {
        uColorA: { value: new THREE.Color(0xffffff) },
        uColorB: { value: new THREE.Color(0xbfe3ef) },
      },
      transparent: true,
      depthWrite: false,
    });

    this.points = new THREE.Points(g, this.material);
    this.points.frustumCulled = false;
    this.points.name = 'foam';
    if (parent) parent.add(this.points);

    // reusable rule closure (no allocation per frame)
    const self = this;
    this._ruleCache = { maxSpeed: 0, minNeighbors: 0 };
    this._ruleFn = function foamRule(i) {
      const s = self.sim;
      const vx = s.vel[i * 3], vy = s.vel[i * 3 + 1], vz = s.vel[i * 3 + 2];
      return (vx * vx + vy * vy + vz * vz > self._ruleCache.maxSpeed * self._ruleCache.maxSpeed)
        || s.nCount[i] < self._ruleCache.minNeighbors;
    };
  }

  /** Default spawn predicate: fast or neighbor-starved water particles.
   * Scene code usually goes through update(dt, flags); this is the direct API. */
  defaultRule(maxSpeed = 1.6, minNeighbors = 6) {
    const s = this.sim;
    const sp2 = maxSpeed * maxSpeed;
    return function rule(i) {
      const vx = s.vel[i * 3], vy = s.vel[i * 3 + 1], vz = s.vel[i * 3 + 2];
      return (vx * vx + vy * vy + vz * vz > sp2) || s.nCount[i] < minNeighbors;
    };
  }

  /**
   * Velocity inheritance: foam keeps mostly the TANGENTIAL (flow-along-
   * surface ≈ horizontal) component of the water velocity and damps the
   * surface-normal (vertical) component, so spray doesn't rocket upward and
   * core lines advect downstream. The foam particle owns its velocity from
   * spawn onward — it keeps drifting after its source water particle moves on.
   */
  _inheritVelocity(i) {
    const P = this.params;
    const v = this.sim.vel;
    return [
      v[i * 3] * P.tangentialInherit,
      v[i * 3 + 1] * P.normalInherit,
      v[i * 3 + 2] * P.tangentialInherit,
    ];
  }

  /**
   * Spawn at most one foam particle per qualifying water particle.
   * @param {(i:number)=>boolean} ruleFn predicate over water-particle index
   * @param {number} [maxSpawns=spawnPerFrame] spawn cap for this call
   * @param {number} [type=0] foam tier for this batch (0 spray, 1 core)
   * @returns {number} number of foam particles actually spawned
   */
  spawnFromRule(ruleFn, maxSpawns = this.params.spawnPerFrame, type = 0) {
    const s = this.sim;
    const n = s.count;
    let spawned = 0;
    for (let i = 0; i < n; i++) {
      if (spawned >= maxSpawns) break;
      if (!ruleFn(i)) continue;
      const [ivx, ivy, ivz] = this._inheritVelocity(i);
      this._spawnOne(
        s.pos[i * 3], s.pos[i * 3 + 1], s.pos[i * 3 + 2],
        ivx, ivy, ivz, type,
      );
      spawned++;
    }
    return spawned;
  }

  /** Spawn one foam particle (recycles the oldest slot when saturated).
   * type: 0 = spray (small, fast-fade), 1 = core whitewater (big, slow-fade). */
  _spawnOne(x, y, z, vx = 0, vy = 0, vz = 0, type = 0) {
    let i;
    if (this.count < this.maxFoam) {
      i = this.count++;
    } else {
      i = this._write;          // ring overwrite
      this._write = (this._write + 1) % this.maxFoam;
    }
    const P = this.params;
    this._px[i] = x; this._py[i] = y; this._pz[i] = z;
    this._vx[i] = vx; this._vy[i] = vy; this._vz[i] = vz;
    this._age[i] = 0;
    this._type[i] = type;
    if (type === 1) {
      this._life[i] = P.coreLifeMin + Math.random() * (P.coreLifeMax - P.coreLifeMin);
      this._size[i] = P.coreSizeMin + Math.random() * (P.coreSizeMax - P.coreSizeMin);
    } else {
      this._life[i] = P.lifeMin + Math.random() * (P.lifeMax - P.lifeMin);
      this._size[i] = P.sizeMin + Math.random() * (P.sizeMax - P.sizeMin);
    }
    this._writeSlot(i);
  }

  /** Push one pool slot's state into the GPU attribute arrays. */
  _writeSlot(i) {
    const a = this._aPos.array, b = this._aLife.array, c = this._aSize.array;
    a[i * 3] = this._px[i]; a[i * 3 + 1] = this._py[i]; a[i * 3 + 2] = this._pz[i];
    b[i] = 1; c[i] = this._size[i];
  }

  /** Swap slots a and b in every CPU array. */
  _swap(a, b) {
    const t = (arr) => { const v = arr[a]; arr[a] = arr[b]; arr[b] = v; };
    t(this._px); t(this._py); t(this._pz);
    t(this._vx); t(this._vy); t(this._vz);
    t(this._age); t(this._life); t(this._size); t(this._type);
  }

  /**
   * Two-tier spawn pass (single walk over the water particles, zero alloc):
   * particles above `coreSpeed` spawn CORE foam (type 1) up to `coreBudget`;
   * everything else matching the legacy rule spawns SPRAY (type 0) up to
   * params.spawnPerFrame. Tiers are capped independently, so a burst of core
   * candidates can never starve spray and vice versa.
   */
  _spawnTwoTier(maxSpeed, minNeighbors, coreSpeed, coreBudget) {
    const s = this.sim;
    const P = this.params;
    const ms2 = maxSpeed * maxSpeed;
    const cs2 = coreSpeed * coreSpeed;
    let sprayLeft = P.spawnPerFrame;
    let coreLeft = Math.max(0, coreBudget | 0);
    for (let i = 0, n = s.count; i < n; i++) {
      if (sprayLeft <= 0 && coreLeft <= 0) break;
      const vx = s.vel[i * 3], vy = s.vel[i * 3 + 1], vz = s.vel[i * 3 + 2];
      const sp2 = vx * vx + vy * vy + vz * vz;
      if (!(sp2 > ms2 || s.nCount[i] < minNeighbors)) continue;
      if (sp2 > cs2 && coreLeft > 0) {
        const [ivx, ivy, ivz] = this._inheritVelocity(i);
        this._spawnOne(s.pos[i * 3], s.pos[i * 3 + 1], s.pos[i * 3 + 2],
          ivx, ivy, ivz, 1);
        coreLeft--;
      } else if (sprayLeft > 0) {
        const [ivx, ivy, ivz] = this._inheritVelocity(i);
        this._spawnOne(s.pos[i * 3], s.pos[i * 3 + 1], s.pos[i * 3 + 2],
          ivx, ivy, ivz, 0);
        sprayLeft--;
      }
    }
  }

  /**
   * Frame update. With `flags` ({maxSpeed, minNeighbors, coreSpeed?,
   * coreBudget?}), flags qualifying water particles and spawns new foam
   * particles; then integrates motion, ages/fades, compacts expired
   * particles, and keeps geometry.drawRange in sync with .count.
   *
   * Spawn tiers (per frame, capped separately):
   *  - CORE tier (only when `coreSpeed` is a number): water speed > coreSpeed
   *    → type-1 foam, big + slow-fade, up to `coreBudget` spawns
   *    (default: spawnPerFrame / 2).
   *  - SPRAY tier (legacy rule): speed > maxSpeed OR nCount < minNeighbors
   *    → type-0 foam, small + fast-fade, up to spawnPerFrame spawns.
   * Omitting coreSpeed reproduces legacy single-tier behavior exactly.
   *
   * @param {number} dt seconds
   * @param {{maxSpeed?:number,minNeighbors?:number,coreSpeed?:number,
   *          coreBudget?:number}} [flags]
   */
  update(dt = 0, flags = null) {
    const P = this.params;

    // --- spawn phase ---
    if (flags && typeof flags.maxSpeed === 'number' && this.sim.count > 0) {
      if (typeof flags.coreSpeed === 'number') {
        this._spawnTwoTier(flags.maxSpeed, flags.minNeighbors ?? 0,
          flags.coreSpeed, flags.coreBudget ?? (P.spawnPerFrame >> 1));
      } else {
        this._ruleCache.maxSpeed = flags.maxSpeed;
        this._ruleCache.minNeighbors = flags.minNeighbors ?? 0;
        this.spawnFromRule(this._ruleFn, P.spawnPerFrame);
      }
    }

    // --- integrate + age + compact (single pass, swap-with-last) ---
    const dragK = Math.max(0, 1 - P.drag * dt);
    // while young, foam rides the current: drag at half strength so inherited
    // tangential momentum carries it downstream after its source moves on
    const dragRise = Math.max(0, 1 - P.drag * 0.5 * dt);
    let i = 0;
    while (i < this.count) {
      this._age[i] += dt;
      if (this._age[i] >= this._life[i]) {
        // expire: pull last live particle into this slot, re-process it
        const last = --this.count;
        if (i !== last) {
          this._swap(i, last);
          this._writeSlot(i);
        }
        continue;
      }
      // buoyant drift up while young, then reduced-gravity fall
      const k = this._age[i] < P.riseTime ? dragRise : dragK;
      if (this._age[i] < P.riseTime) this._vy[i] += P.buoyancy * dt;
      else this._vy[i] -= P.gravity * dt;
      this._vx[i] *= k; this._vy[i] *= k; this._vz[i] *= k;
      this._px[i] += this._vx[i] * dt;
      this._py[i] += this._vy[i] * dt;
      this._pz[i] += this._vz[i] * dt;

      const fade = 1 - this._age[i] / this._life[i];
      this._aLife.array[i] = fade;
      const a = this._aPos.array;
      a[i * 3] = this._px[i]; a[i * 3 + 1] = this._py[i]; a[i * 3 + 2] = this._pz[i];
      i++;
    }

    this._aPos.needsUpdate = true;
    this._aLife.needsUpdate = true;
    this._aSize.needsUpdate = true;
    this.geometry.setDrawRange(0, this.count);
  }

  /** Attach tunables to a lil-gui / dat.GUI folder. Tolerates mock GUIs. */
  addGui(gui) {
    try {
      const f = gui.addFolder ? gui.addFolder('Foam') : gui;
      f.add(this.params, 'spawnPerFrame', 0, 1000).name('spawn/frame');
      f.add(this.params, 'riseTime', 0, 2).name('rise time');
      f.add(this.params, 'buoyancy', 0, 10).name('buoyancy');
      f.add(this.params, 'gravity', 0, 20).name('gravity');
      f.add(this.params, 'drag', 0, 10).name('drag');
      f.add(this.params, 'tangentialInherit', 0, 1).name('tang. inherit');
      f.add(this.params, 'normalInherit', 0, 1).name('normal inherit');
      f.add(this.params, 'lifeMin', 0.1, 3).name('life min');
      f.add(this.params, 'lifeMax', 0.1, 5).name('life max');
    } catch { /* GUI shapes vary — never let cosmetics throw */ }
  }

  dispose() {
    try { this.points.parent?.remove?.(this.points); } catch { /* best effort */ }
    this.geometry.dispose();
    this.material.dispose();
  }
}

/* ======================================================================
 * makeCapillaryNormalTexture — tileable ripple normal map
 * ====================================================================== */

/**
 * Tileable RGBA DataTexture normal map built from 3–4 integer-frequency sine
 * wave trains (so the field is exactly periodic — no seam). Wave set is
 * transpose-symmetric so horizontal and vertical steps are equally bounded.
 * @param {number} [size=256]
 * @returns {THREE.DataTexture}
 */
export function makeCapillaryNormalTexture(size = 256) {
  const S = size | 0;
  // transpose-symmetric wave trains with mirror phases (φ ∈ {0, π}) ⇒ exactly
  // periodic under wrap on both axes.
  const WAVES = [
    // kx, ky, amplitude, phase
    [1, 0, 0.55, 0.0], [0, 1, 0.55, 0.0],
    [2, 3, 0.30, 0.0], [3, 2, 0.30, Math.PI],
    [5, -2, 0.22, Math.PI], [-2, 5, 0.22, 0.0],
    [7, 4, 0.14, 0.0], [4, 7, 0.14, Math.PI],
  ];
  const STRENGTH = 1.9; // gradient → normal tilt scale

  const data = new Uint8Array(S * S * 4);
  const F = new Float32Array(S * S);

  const build = (phaseShift) => {
    const TWO_PI = Math.PI * 2;
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        let f = 0;
        for (let w = 0; w < WAVES.length; w++) {
          const [kx, ky, A, ph] = WAVES[w];
          f += A * Math.cos(TWO_PI * (kx * x + ky * y) / S + ph + phaseShift);
        }
        F[y * S + x] = f;
      }
    }
  };

  // wrapped central differences → tangent-space normals
  const encode = () => {
    const norm = (gx, gz) => {
      const inv = 1 / Math.hypot(gx * STRENGTH, 1, gz * STRENGTH);
      return [
        (-gx * STRENGTH * inv * 0.5 + 0.5) * 255,
        (-gz * STRENGTH * inv * 0.5 + 0.5) * 255,
        inv * 255,
      ];
    };
    for (let y = 0; y < S; y++) {
      const yUp = ((y + 1) % S) * S, yDn = ((y - 1 + S) % S) * S, row = y * S;
      for (let x = 0; x < S; x++) {
        const xR = (x + 1) % S, xL = (x - 1 + S) % S;
        const gx = (F[row + xR] - F[row + xL]) * 0.5;
        const gz = (F[yUp + x] - F[yDn + x]) * 0.5;
        const o = (row + x) * 4;
        const [r, g, b] = norm(gx, gz);
        data[o] = r; data[o + 1] = g; data[o + 2] = b; data[o + 3] = 255;
      }
    }
  };

  // seam self-check (same metric consumers use): wrap steps must be no
  // steeper than the worst measured interior step. Deterministic retry with
  // shifted phases guarantees a clean bake.
  const seamsOK = () => {
    const step = (a, b) => Math.max(Math.abs(data[a] - data[b]), Math.abs(data[a + 1] - data[b + 1]));
    let interiorMax = 0;
    for (let y = 0; y < S - 1; y++) {
      for (let x = 0; x < S; x++) {
        interiorMax = Math.max(interiorMax, step((y * S + x) * 4, ((y + 1) * S + x) * 4));
      }
    }
    let seamV = 0, seamH = 0;
    for (let x = 0; x < S; x++) seamV = Math.max(seamV, step((0 * S + x) * 4, ((S - 1) * S + x) * 4));
    for (let y = 0; y < S; y++) seamH = Math.max(seamH, step((y * S + 0) * 4, (y * S + S - 1) * 4));
    return seamV <= interiorMax && seamH <= interiorMax;
  };

  let shift = 0;
  for (let attempt = 0; attempt < 8; attempt++) {
    build(shift);
    encode();
    if (seamsOK()) break;
    shift += 0.37;
  }

  const tex = new THREE.DataTexture(data, S, S, THREE.RGBAFormat);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
}

/* ======================================================================
 * makeRippleDecalQuad — env-reflective decal animated by the ripple map
 * ====================================================================== */

/**
 * Flat quad whose material perturbs lighting/env reflections with the
 * capillary ripple normal map, scrolled by time.
 *
 * Animation contract: material.uniforms.uTime.value drives it, and
 * mesh.userData.setTime(t) works even before first render (it falls back to
 * scrolling the normalMap offset directly).
 *
 * @param {number} [width=2]
 * @param {number} [height=2]
 * @param {object} [opts={}] { opacity, rippleTex, scrollX, scrollY, color }
 * @returns {THREE.Mesh}
 */
export function makeRippleDecalQuad(width = 2, height = 2, opts = {}) {
  const tex = opts.rippleTex ?? makeCapillaryNormalTexture();
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(2, 2);

  const mat = new THREE.MeshStandardMaterial({
    color: opts.color ?? 0xdff3fa,
    normalMap: tex,
    transparent: opts.opacity !== undefined ? true : true,
    opacity: opts.opacity ?? 0.45,
    depthWrite: false,
    roughness: 0.08,
    metalness: 0.0,
    envMapIntensity: 1.2,
  });
  // custom uniform channel (standard materials don't ship one) so callers can
  // drive animation via material.uniforms.uTime.value like any ShaderMaterial
  mat.uniforms = { uTime: { value: 0 } };

  const scrollX = opts.scrollX ?? 0.03;
  const scrollY = opts.scrollY ?? 0.021;
  const baseOffset = tex.offset.clone();

  const setTime = (t) => {
    mat.uniforms.uTime.value = t;
    // pre-compile fallback: scrolling the normal map itself always works,
    // whether or not onBeforeCompile has run yet
    tex.offset.set(baseOffset.x + t * scrollX, baseOffset.y + t * scrollY);
  };

  const quad = new THREE.Mesh(new THREE.PlaneGeometry(width, height), mat);
  quad.userData.setTime = setTime;
  return quad;
}
