// water-pack/surface.js — three.js rendering for the pack's WaterSim.
// Metaball surface (three.js MarchingCubes) + foam/spray overlay + optional
// waterline marker. Zero scene assumptions: pass a THREE scene, get a group back.
// All visual state lives in `state` — wire it to lil-gui yourself or use
// addGui(gui) if you have one.
//
// Lane P1 performance notes (the MC field build was 64ms/frame @ res 156):
//
// 1. DYNAMIC VOLUME FITTING — the MC volume tracks the particle AABB (+margin,
//    quantized to cell multiples, shrink-hysteresis) instead of the static
//    scene bounds. Water often occupies a small fraction of the bounds box, so
//    this concentrates grid cells on actual water and lets auto-resolution
//    drop accordingly. The auto-res rule (cell <= ballDiameter/1.5 per axis)
//    is evaluated against the FITTED volume every frame.
//
// 2. AXIS SLAB SPLITTING — the MarchingCubes addon forces a cubic res³ grid
//    and its update() polygonizes every cube, so cost ~ res³ regardless of how
//    little water there is. Splitting the fitted volume into k equal slabs
//    along its longest axis (each its own MarchingCubes at res k-times
//    smaller) cuts reset+polygonize cost ~k². The scalar field is additive,
//    so a ball straddling a seam is added to BOTH slabs and the combined
//    field is identical to a single global grid — the surface is seamless and
//    normals at seams are valid (seam cells are interior to their grid).
//
// 3. SATURATED-CELL CULLING — before splatting a particle, read back the 27
//    field cells around its center (the .field arrays we write directly); if
//    ALL already sit above isolation+cullMargin, the particle is deep interior
//    and its tail contributes nothing the surface needs — skip the splat
//    entirely. Only surface-layer balls are ever written. Order-dependent but
//    stable: the first particles through saturate a region, later ones get
//    culled.
//
// 4. INPUT STRIDE (downsample) — derived, self-gating. A lone ball crosses
//    isolation out to exactly rN = 0.55h/boundsExtent (normalized); its field
//    reaches zero at R0 ≈ sqrt(1+iso/sub)·rN. Skipping every Nth particle is
//    overlap-safe only while N·spacing stays inside that reach on EVERY axis:
//    N = floor(min_a worldReach_a / particleSpacing). On anisotropic scenes
//    (e.g. creek: y-extent 9 vs x 40) the vertical reach is BELOW one
//    particle spacing, so the derivation yields N=1 (no-op) and culling does
//    the work instead; near-cubic scenes derive N≈2.
//
// Lane M1 incremental perf notes (this revision):
//
// 5. PER-FRAME FIELD CLEAR, DONE CHEAPLY — the previous revision NEVER cleared
//    the MC grids between frames (the addon's reset() was only called by the
//    old water-render.js). Fields accumulated frame over frame, saturated-cell
//    culling then kicked in everywhere within a few frames and the surface
//    froze into a stale blob. We restore correct per-frame semantics without
//    paying the addon's full-grid reset(): every splat records the dirty cell
//    box it touched per slab, and the NEXT frame clears just those rows
//    (field + normal_cache; palette is never written — see #6). On flat water
//    this clears ~10% of each grid instead of 100%.
//
// 6. LEAN SPLAT — we write the field ourselves instead of calling the addon's
//    addBall(): identical math (val = strength/(1e-6 + d²) − sub, val>0 guard,
//    same [1,res−1] clamps) minus the per-cell palette work (a sqrt + 3 adds
//    that the addon always does even with enableColors=false). Field values —
//    and therefore the rendered surface — are bit-identical to addBall().
//
// 7. KERNEL NARROWING (state.kernelSub) — the ball kernel is
//    val(d) = rN²(iso+sub)/d² − sub with lone-ball iso radius FIXED at
//    rN = 0.55h for any sub (strength compensates). Its ZERO radius however is
//    R0 = sqrt(1+iso/sub)·rN: 2.80·rN at the legacy sub=12, i.e. every ball
//    writes ~(2·2.8rN)³ cells although cells beyond ~1 cell past the iso
//    surface barely move it. Raising sub narrows the written footprint
//    ((2·1.73rN)³ at sub=40, ≈4.5× fewer cells) while the iso radius stays
//    rN. Neighbor-tail blending shrinks slightly — bench-measured enclosed-
//    volume delta vs sub=12 is the parity gate (see /tmp/mc-perf-bench.mjs);
//    the shipped default keeps that delta under ~0.5%. state.cullMargin
//    headroom is unaffected (it scales with iso, not sub).
//
// 8. TEMPORAL REUSE (state.temporalReuse, default OFF) — the addon CANNOT
//    update incrementally: update() polygonizes all res³ cells unconditionally
//    and reset() wipes whole grids; there is no dirty-cell API (forking the
//    addon is M2 territory). The only reuse available at this layer is
//    whole-frame skip: if the fitted volume didn't change and NO particle
//    moved more than half a cell since the last built frame, keep yesterday's
//    mesh untouched (it is still exactly what the field would produce to
//    within sub-cell detail). Off by default — it halves surface refresh on
//    near-static pools; opt in for very calm scenes.

import * as THREE from 'three';
import { MarchingCubes } from 'three/addons/objects/MarchingCubes.js';

export function createWaterSurface(sim, scene, bounds, opts = {}) {
  const state = {
    mode: 'metaballs',                 // 'metaballs' | 'particles'
    resolution: opts.resolution ?? 0,  // 0 = AUTO from ball size (see layout())
    isolation: 82,
    opacity: 0.94,
    color: opts.color ?? 0x1f7ad0,
    foam: true,
    foamMaxSpeed: 6.5,
    foamMaxNeighbors: 14,
    foamSize: 0.16,
    waterline: opts.waterline ?? null, // {min:[x,z], max:[x,z]} region → level marker
    slices: 0,                         // MC axis slabs: 0 = auto (1–4), else forced
    inputStride: 0,                    // ball-input downsample: 0 = derived auto
    cull: true,                        // saturated-cell culling of interior balls
    cullMargin: 12,                    // field headroom required to cull (~subtract)
    kernelSub: opts.kernelSub ?? 12,   // ball kernel subtract: higher = narrower reach
    temporalReuse: opts.temporalReuse ?? false, // skip rebuild when nothing moved
    _legacyBuild: opts._legacyBuild ?? false,   // bench fallback: addon reset()+addBall()
  };

  const group = new THREE.Group();
  scene.add(group);

  const material = new THREE.MeshPhysicalMaterial({
    color: state.color,
    transparent: true,
    opacity: state.opacity,
    roughness: 0.05,
    metalness: 0.0,
    transmission: 0.0,
    ior: 1.33,
    clearcoat: 0.9,
    clearcoatRoughness: 0.06,
    reflectivity: 1.0,
    envMapIntensity: 1.6,
  });

  // ---- geometry constants -------------------------------------------------
  // Ball diameter reference (matches the historical strength convention:
  // lone-ball iso radius = 0.55h world on the split axis ⇒ visual diameter
  // ≈ 2·1.54h once the kernel's falloff between neighbors is included).
  const dWorld = () => 2 * sim.p.h * 0.55;
  const baseCell = () => dWorld() / 1.5;   // auto-res rule: cell ≤ diameter/1.5

  // ---- MC slab pool -------------------------------------------------------
  let mcPool = [];
  let slabDirty = [];                      // per-slab written-cell box (last frame)
  let mcTriCount = 0;
  let poolKey = '';
  let lastRebuildFrame = -1e9;
  let frameNo = 0;
  let warnedBudget = false;

  // fitted volume (world AABB, quantized)
  const volMin = [0, 0, 0], volMax = [0, 0, 0];
  let volValid = false;
  let volChanged = false;
  let shrinkHold = 0;
  const SHRINK_FRAMES = 45;

  // temporal-reuse bookkeeping
  let reusePrev = null;                    // positions snapshot at last BUILT frame
  let reuseN = -1, reuseK = -1, reuseRes = -1;

  // last-frame stats (see stats getter)
  let lastStride = 1, lastK = 0, lastRes = 0, lastBalls = 0, lastCulled = 0, lastReused = false;

  function disposePool() {
    for (const m of mcPool) { group.remove(m); m.geometry.dispose(); }
    mcPool = [];
    slabDirty = [];
    poolKey = '';
  }

  function ensurePool(k, resSlab) {
    const key = k + '|' + resSlab;
    if (key === poolKey) return true;
    // micro-debounce: layout keys are quantized (volume snaps to cell multiples,
    // shrinks held 45 frames), so mismatches are discrete events — rebuilding
    // immediately costs a one-frame alloc spike, far better than freezing the
    // surface while deferring.
    if (mcPool.length && frameNo - lastRebuildFrame < 3) return false;
    disposePool();
    // Per-slab triangle budget: total area is roughly conserved when splitting,
    // so ~total/k each, plus slack for lumpy distribution (water piled in one
    // slab). Too small ⇒ three.js silently drops chunks ("bite" bug).
    const budget = Math.max(100000, Math.ceil((250000 * 1.3) / k));
    for (let i = 0; i < k; i++) {
      const mc = new MarchingCubes(resSlab, material, false, false, budget);
      mc.isolation = state.isolation;
      mc.enableUvs = false; mc.enableColors = false;
      mc.frustumCulled = false;
      group.add(mc);
      mcPool.push(mc);
      slabDirty.push({ x0: 0, x1: -1, y0: 0, y1: -1, z0: 0, z1: -1 });
    }
    poolKey = key;
    lastRebuildFrame = frameNo;
    return true;
  }

  // Clear only the cells a slab's splats wrote LAST frame (row-wise fills).
  // normal_cache must be cleared alongside: compNorm() trusts any non-zero
  // cached gradient, so stale entries would survive as wrong normals.
  function clearDirty(mc, d) {
    if (d.x1 < d.x0) return;
    const res = mc.resolution, size2 = res * res;
    const f = mc.field, nc = mc.normal_cache;
    for (let z = d.z0; z <= d.z1; z++) {
      const zoff = size2 * z;
      for (let y = d.y0; y <= d.y1; y++) {
        const base = zoff + res * y;
        f.fill(0, base + d.x0, base + d.x1 + 1);
        nc.fill(0, (base + d.x0) * 3, (base + d.x1 + 1) * 3);
      }
    }
    d.x1 = d.x0 - 1; // invalidate until the next splat
  }

  // Fit the MC volume to the particles (+margin), quantized to cell multiples.
  // Grow immediately (never clip water); shrink only after a sustained hold.
  function fitVolume(pos, n) {
    volChanged = false;
    if (n === 0) return;
    const q = baseCell();
    const margin = dWorld() + q;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
      for (let a = 0; a < 3; a++) {
        const v = pos[i * 3 + a];
        if (v < lo[a]) lo[a] = v;
        if (v > hi[a]) hi[a] = v;
      }
    }
    const candLo = [0, 0, 0], candHi = [0, 0, 0];
    let candVol = 1;
    for (let a = 0; a < 3; a++) {
      candLo[a] = Math.max(bounds.min[a], Math.floor((lo[a] - margin) / q) * q);
      candHi[a] = Math.min(bounds.min[a] + bounds.size[a], Math.ceil((hi[a] + margin) / q) * q);
      if (candHi[a] <= candLo[a]) { candHi[a] = candLo[a] + q; }
      candVol *= candHi[a] - candLo[a];
    }
    const assign = () => {
      for (let a = 0; a < 3; a++) { volMin[a] = candLo[a]; volMax[a] = candHi[a]; }
      volChanged = true; shrinkHold = 0;
    };
    if (!volValid) { assign(); volValid = true; return; }
    let grows = false;
    for (let a = 0; a < 3; a++) {
      if (candLo[a] < volMin[a] || candHi[a] > volMax[a]) { grows = true; break; }
    }
    if (grows) { assign(); return; }
    let curVol = 1;
    for (let a = 0; a < 3; a++) curVol *= volMax[a] - volMin[a];
    if (candVol < curVol * 0.9) {
      if (++shrinkHold >= SHRINK_FRAMES) assign();
    } else {
      shrinkHold = 0;
    }
  }

  function update(pos, vel, n) {
    frameNo++;
    points.visible = state.mode === 'particles';
    for (const m of mcPool) m.visible = state.mode === 'metaballs';

    if (points.visible) {
      for (let i = 0; i < n; i++) {
        ptPos[i * 3] = pos[i * 3]; ptPos[i * 3 + 1] = pos[i * 3 + 1]; ptPos[i * 3 + 2] = pos[i * 3 + 2];
        const sp = Math.hypot(vel[i * 3], vel[i * 3 + 1], vel[i * 3 + 2]);
        _c.copy(coldColor).lerp(hotColor, Math.min(sp / 8, 1));
        ptCol[i * 3] = _c.r; ptCol[i * 3 + 1] = _c.g; ptCol[i * 3 + 2] = _c.b;
      }
      ptGeo.setDrawRange(0, n);
      ptGeo.attributes.position.needsUpdate = true;
      ptGeo.attributes.color.needsUpdate = true;
    }

    if (state.mode === 'metaballs') {
      fitVolume(pos, n);
      if (!volValid) return;

      // ---- layout: split axis, slab count, slab resolution ----------------
      const volSize = [volMax[0] - volMin[0], volMax[1] - volMin[1], volMax[2] - volMin[2]];
      const cell = baseCell();
      // auto-res rule per axis against the FITTED volume: cells ≤ diameter/1.5
      const need = [
        Math.ceil(volSize[0] / cell),
        Math.ceil(volSize[1] / cell),
        Math.ceil(volSize[2] / cell),
      ];
      let resGlobal = Math.max(need[0], need[1], need[2]);
      resGlobal = Math.max(24, Math.min(state.resolution > 0 ? state.resolution : resGlobal, 160));
      let axis = 0;
      if (need[1] > need[axis]) axis = 1;
      if (need[2] > need[axis]) axis = 2;
      // Slab count: aim for resSlab ≈ 56, which keeps world cell sizes close to
      // what the old fixed-bounds single-grid build used at typical resolutions
      // (look parity — measured: k=3 @ res 52 → +8% tris vs baseline; coarser
      // slab grids fatten balls vertically, finer ones sharpen them). Bench
      // sweep (Lane M1) confirmed k≈resGlobal/56 minimizes total update time:
      // larger k shrinks polygonize but pays duplicated seam balls + per-slab
      // overhead; smaller k does the reverse.
      const k = Math.max(1, Math.min(state.slices > 0 ? state.slices : Math.round(resGlobal / 56), 4));
      // slab res must still satisfy the rule on the two non-split axes
      let resOther = 0;
      for (let a = 0; a < 3; a++) if (a !== axis) resOther = Math.max(resOther, need[a]);
      const resSlab = Math.max(Math.max(24, resOther), Math.ceil(resGlobal / k));
      if (!ensurePool(k, resSlab)) return; // 1-frame deferral, never a freeze

      // ---- per-slab placement ---------------------------------------------
      const slabLen = volSize[axis] / k;
      for (let s = 0; s < k; s++) {
        const mc = mcPool[s];
        mc.scale.set(volSize[0] / 2, volSize[1] / 2, volSize[2] / 2);
        mc.position.set(
          volMin[0] + volSize[0] / 2,
          volMin[1] + volSize[1] / 2,
          volMin[2] + volSize[2] / 2);
        mc.scale.setComponent(axis, slabLen / 2);
        mc.position.setComponent(axis, volMin[axis] + slabLen * (s + 0.5));
        mc.isolation = state.isolation;
      }

      // ---- ball params -----------------------------------------------------
      // Lone-ball ISO radius is pinned at 0.55h world (normalized rN on the
      // split axis) for ANY kernelSub: strength = rN²(iso+sub) makes the field
      // cross isolation exactly at rN. kernelSub only sets how far beyond rN
      // the kernel keeps writing (zero radius sqrt(1+iso/sub)·rN).
      const SUB = state.kernelSub > 0 ? state.kernelSub : 12;
      const rN = (sim.p.h * 0.55) / slabLen;
      const strength = rN * rN * (state.isolation + SUB);
      const R0n = Math.sqrt(strength / SUB);       // zero-field radius (norm.)

      // ---- derived input stride (see module header; usually 1 on flat scenes)
      const spacing = sim.p.h * (sim.p.spacingRatio ?? 0.55);
      let stride = state.inputStride > 0 ? state.inputStride : 1;
      if (state.inputStride === 0) {
        let ns = Infinity;
        for (let a = 0; a < 3; a++) {
          const reach = R0n * (a === axis ? slabLen : volSize[a]);
          ns = Math.min(ns, reach / spacing);
        }
        stride = Math.max(1, Math.floor(ns));
      }
      lastStride = stride; lastK = k; lastRes = resSlab;

      // ---- temporal reuse: skip the whole rebuild if nothing moved ---------
      let reused = false;
      if (state.temporalReuse && !volChanged && reusePrev && reuseN === n &&
          reuseK === k && reuseRes === resSlab) {
        let maxD = 0;
        const m3 = n * 3;
        for (let j = 0; j < m3; j += 3) {
          const dx = Math.abs(pos[j] - reusePrev[j]);
          const dy = Math.abs(pos[j + 1] - reusePrev[j + 1]);
          const dz = Math.abs(pos[j + 2] - reusePrev[j + 2]);
          if (dx > maxD) maxD = dx;
          if (dy > maxD) maxD = dy;
          if (dz > maxD) maxD = dz;
        }
        reused = maxD < cell * 0.5;
      }
      lastReused = reused;

      if (reused) {
        material.opacity = state.opacity;
      } else {
        if (state.temporalReuse) {
          if (!reusePrev || reusePrev.length < n * 3) reusePrev = new Float32Array(n * 3);
          reusePrev.set(pos.subarray(0, n * 3));
          reuseN = n; reuseK = k; reuseRes = resSlab;
        }

        // ---- build fields ---------------------------------------------------
        const res = resSlab, size2 = res * res;
        const cull = state.cull;
        const cullThr = state.isolation + state.cullMargin;
        let balls = 0, culled = 0;

        if (state._legacyBuild) {
          // bench/debug fallback: the addon's own full reset + addBall path
          // (semantically the pre-M1 intended behavior, full-grid clear price
          // included).
          for (const mc of mcPool) mc.reset();
          for (let i = 0; i < n; i += stride) {
            const px = pos[i * 3], py = pos[i * 3 + 1], pz = pos[i * 3 + 2];
            const nx_ = (px - volMin[0]) / volSize[0];
            const ny_ = (py - volMin[1]) / volSize[1];
            const nz_ = (pz - volMin[2]) / volSize[2];
            const t = axis === 0 ? nx_ : axis === 1 ? ny_ : nz_;
            if (t < -0.01 || t > 1.01) continue;
            let s0 = Math.ceil(t * k - 1 - R0n), s1 = Math.floor(t * k + R0n);
            if (s0 < 0) s0 = 0;
            if (s1 > k - 1) s1 = k - 1;
            for (let s = s0; s <= s1; s++) {
              const u = t * k - s;
              let bx, by, bz;
              if (axis === 0) { bx = u; by = ny_; bz = nz_; }
              else if (axis === 1) { bx = nx_; by = u; bz = nz_; }
              else { bx = nx_; by = ny_; bz = u; }
              if (bx < -R0n || bx > 1 + R0n || by < -R0n || by > 1 + R0n || bz < -R0n || bz > 1 + R0n) continue;
              mcPool[s].addBall(bx, by, bz, strength, SUB);
              balls++;
            }
          }
        } else {
          // clear what LAST frame's splats wrote (cheap partial reset)
          for (let s = 0; s < k; s++) clearDirty(mcPool[s], slabDirty[s]);

          for (let i = 0; i < n; i += stride) {
            const px = pos[i * 3], py = pos[i * 3 + 1], pz = pos[i * 3 + 2];
            // normalized coords in the fitted volume
            const nx_ = (px - volMin[0]) / volSize[0];
            const ny_ = (py - volMin[1]) / volSize[1];
            const nz_ = (pz - volMin[2]) / volSize[2];
            const t = axis === 0 ? nx_ : axis === 1 ? ny_ : nz_;
            if (t < -0.01 || t > 1.01) continue;
            // slabs this ball can touch (its zero-radius reach)
            let s0 = Math.ceil(t * k - 1 - R0n), s1 = Math.floor(t * k + R0n);
            if (s0 < 0) s0 = 0;
            if (s1 > k - 1) s1 = k - 1;
            if (s1 < s0) continue;

            // saturated-cell culling: 27 cells around the center, checked in EVERY
            // slab this ball touches. Culling based on the primary slab alone
            // starved the neighbour slab's field near seams → visible gaps and
            // straight clipped edges at slab boundaries (user-visible bug).
            // Fast necessary condition first: if the CENTER cell is below the
            // threshold its 27 neighbors can't all be above it — 1 read instead
            // of 27 for every ball that can't possibly cull (shallow water:
            // virtually all of them; measured culled=0 in the creek bench).
            const sp = Math.max(0, Math.min(k - 1, Math.floor(t * k)));
            if (cull && s0 <= sp && sp <= s1) {
              let sat = true;
              const uC = t * k - sp;
              const gxc = Math.floor((axis === 0 ? uC : nx_) * res);
              const gyc = Math.floor((axis === 1 ? uC : ny_) * res);
              const gzc = Math.floor((axis === 2 ? uC : nz_) * res);
              if (gxc <= 0 || gxc >= res - 1 || gyc <= 0 || gyc >= res - 1 || gzc <= 0 || gzc >= res - 1) sat = false;
              else if (mcPool[sp].field[size2 * gzc + res * gyc + gxc] < cullThr) sat = false;
              if (sat) {
              for (let s = s0; s <= s1 && sat; s++) {
                const field = mcPool[s].field;
                // local grid cell of the centre in THIS slab's frame
                const u = t * k - s;
                const gx = Math.floor((axis === 0 ? u : nx_) * res);
                const gy = Math.floor((axis === 1 ? u : ny_) * res);
                const gz = Math.floor((axis === 2 ? u : nz_) * res);
                if (gx <= 0 || gx >= res - 1 || gy <= 0 || gy >= res - 1 || gz <= 0 || gz >= res - 1) { sat = false; break; }
                for (let dz = -1; dz <= 1 && sat; dz++)
                  for (let dy = -1; dy <= 1 && sat; dy++) {
                    let idx = size2 * (gz + dz) + res * (gy + dy) + gx - 1;
                    for (let dx = -1; dx <= 1; dx++)
                      if (field[idx + dx] < cullThr) { sat = false; break; }
                  }
              }
              }
              if (sat) { culled++; continue; }
            }

            for (let s = s0; s <= s1; s++) {
              const u = t * k - s;
              let bx, by, bz;
              if (axis === 0) { bx = u; by = ny_; bz = nz_; }
              else if (axis === 1) { bx = nx_; by = u; bz = nz_; }
              else { bx = nx_; by = ny_; bz = u; }
              if (bx < -R0n || bx > 1 + R0n || by < -R0n || by > 1 + R0n || bz < -R0n || bz > 1 + R0n) continue;
              // lean splat: addBall math minus the palette work (colors off).
              const mc = mcPool[s];
              const rad = res * R0n;
              let x0 = Math.floor(bx * res - rad); if (x0 < 1) x0 = 1;
              let x1 = Math.floor(bx * res + rad); if (x1 > res - 1) x1 = res - 1;
              let y0 = Math.floor(by * res - rad); if (y0 < 1) y0 = 1;
              let y1 = Math.floor(by * res + rad); if (y1 > res - 1) y1 = res - 1;
              let z0 = Math.floor(bz * res - rad); if (z0 < 1) z0 = 1;
              let z1 = Math.floor(bz * res + rad); if (z1 > res - 1) z1 = res - 1;
              const field = mc.field, d = slabDirty[s];
              if (d.x1 < d.x0) { d.x0 = x0; d.x1 = x1; d.y0 = y0; d.y1 = y1; d.z0 = z0; d.z1 = z1; }
              else {
                if (x0 < d.x0) d.x0 = x0; if (x1 > d.x1) d.x1 = x1;
                if (y0 < d.y0) d.y0 = y0; if (y1 > d.y1) d.y1 = y1;
                if (z0 < d.z0) d.z0 = z0; if (z1 > d.z1) d.z1 = z1;
              }
              const eps = 1e-6;
              for (let z = z0; z < z1; z++) {
                const zoff = size2 * z;
                const fz = z / res - bz, fz2 = fz * fz;
                for (let y = y0; y < y1; y++) {
                  const yoff = zoff + res * y;
                  const fy = y / res - by, fy2 = fy * fy;
                  for (let x = x0; x < x1; x++) {
                    const fx = x / res - bx;
                    const val = strength / (eps + fx * fx + fy2 + fz2) - SUB;
                    if (val > 0) field[yoff + x] += val;
                  }
                }
              }
              balls++;
            }
          }
        }
        lastBalls = balls; lastCulled = culled;

        // ---- polygonize -------------------------------------------------------
        mcTriCount = 0;
        for (const mc of mcPool) {
          mc.update();
          mcTriCount += mc.count / 3;
          if (!warnedBudget && mc.count / 3 > mc.geometry.attributes.position.count / 3 - 1) {
            console.warn('[water-pack/surface] MC slab triangle budget exhausted — raise resolution or reduce slices.');
            warnedBudget = true;
          }
        }
        material.opacity = state.opacity;
      }
    }

    // foam
    foamPoints.visible = state.mode === 'metaballs' && state.foam;
    if (foamPoints.visible) {
      let fn = 0;
      for (let i = 0; i < n; i++) {
        const sp = Math.hypot(vel[i * 3], vel[i * 3 + 1], vel[i * 3 + 2]);
        if (sp > state.foamMaxSpeed || sim.nCount[i] < state.foamMaxNeighbors) {
          foamPos[fn * 3] = pos[i * 3]; foamPos[fn * 3 + 1] = pos[i * 3 + 1]; foamPos[fn * 3 + 2] = pos[i * 3 + 2];
          fn++;
        }
      }
      foamGeo.setDrawRange(0, fn);
      foamGeo.attributes.position.needsUpdate = true;
    }

    // waterline marker follows probe level
    if (waterlineMesh && sim.waterLevel !== undefined) {
      waterlineMesh.visible = Number.isFinite(sim.waterLevel);
      waterlineMesh.position.y = sim.waterLevel ?? 0;
    }
  }

  // particles (debug view)
  const ptPos = new Float32Array(sim.p.maxParticles * 3);
  const ptCol = new Float32Array(sim.p.maxParticles * 3);
  const ptGeo = new THREE.BufferGeometry();
  ptGeo.setAttribute('position', new THREE.BufferAttribute(ptPos, 3));
  ptGeo.setAttribute('color', new THREE.BufferAttribute(ptCol, 3));
  const points = new THREE.Points(ptGeo, new THREE.PointsMaterial({ size: 0.14, vertexColors: true, sizeAttenuation: true }));
  points.frustumCulled = false;
  scene.add(points);

  // foam overlay
  const foamPos = new Float32Array(sim.p.maxParticles * 3);
  const foamGeo = new THREE.BufferGeometry();
  foamGeo.setAttribute('position', new THREE.BufferAttribute(foamPos, 3));
  const foamMat = new THREE.PointsMaterial({
    color: 0xeef6ff, size: state.foamSize, sizeAttenuation: true,
    transparent: true, opacity: 0.85, depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const foamPoints = new THREE.Points(foamGeo, foamMat);
  foamPoints.frustumCulled = false;
  group.add(foamPoints);

  // waterline marker: a flat ring/quad at the measured level inside a region
  let waterlineMesh = null;
  if (state.waterline) {
    const w = state.waterline;
    const sx = w.max[0] - w.min[0], sz = w.max[2] - w.min[2];
    waterlineMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(sx, sz),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.25, side: THREE.DoubleSide, depthWrite: false }),
    );
    waterlineMesh.rotation.x = -Math.PI / 2;
    waterlineMesh.position.set((w.min[0] + w.max[0]) / 2, 0, (w.min[2] + w.max[2]) / 2);
    waterlineMesh.visible = false;
    scene.add(waterlineMesh);
  }

  const coldColor = new THREE.Color(0x1e5fbf);
  const hotColor = new THREE.Color(0x66d9ff);
  const _c = new THREE.Color();

  function publicUpdate() {
    update(sim.pos, sim.vel, sim.count);
  }

  function addGui(gui) {
    const f = gui.addFolder('💧 Water Render');
    f.add(state, 'mode', ['metaballs', 'particles']).name('render mode');
    f.add(state, 'resolution', 24, 96, 4).name('MC resolution').onFinishChange(() => { disposePool(); lastRebuildFrame = -1e9; });
    f.add(state, 'slices', 0, 4, 1).name('MC slices (0=auto)').onFinishChange(() => { disposePool(); lastRebuildFrame = -1e9; });
    f.add(state, 'cull').name('cull interior balls');
    f.add(state, 'isolation', 40, 200, 1).name('isolation');
    f.add(state, 'opacity', 0.3, 1).name('opacity');
    f.add(state, 'kernelSub', 12, 80, 1).name('kernel narrow (sub)');
    f.add(state, 'temporalReuse').name('temporal reuse');
    const ff = f.addFolder('Foam / spray');
    ff.add(state, 'foam').name('show foam');
    ff.add(state, 'foamMaxSpeed', 1, 12, 0.5).name('speed threshold');
    ff.add(state, 'foamMaxNeighbors', 2, 40, 1).name('density threshold');
    ff.add(state, 'foamSize', 0.05, 0.4, 0.01).name('sprite size').onChange(v => foamMat.size = v);
    ff.close();
    f.close();
    return f;
  }

  return {
    update: publicUpdate, addGui, state,
    get mcTris() { return mcTriCount; },
    get stats() {
      return { stride: lastStride, k: lastK, resSlab: lastRes, balls: lastBalls, culled: lastCulled, reused: lastReused };
    },
    group,
  };
}
