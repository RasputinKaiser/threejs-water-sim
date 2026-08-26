// water-render.js — three.js visualization of the PBF particle sim.
// Two modes:
//  - "metaballs": MarchingCubes isosurface → smooth, genuinely liquid-looking body
//  - "particles": point sprites colored by speed (debug)
// Plus a fill-level probe: samples a vertical column to report the settled
// water height inside the pool (used by the HUD + flatness metric).

import * as THREE from 'three';
import { MarchingCubes } from 'three/addons/objects/MarchingCubes.js';

export function createWaterRenderer(sim, scene, bounds) {
  // bounds: {min:[x,y,z], size:[x,y,z]} world AABB the marching cubes volume covers
  const state = {
    mode: 'metaballs',
    resolution: 56,
    isolation: 82,
    opacity: 0.92,
    colorBySpeed: true,
  };

  const group = new THREE.Group();
  scene.add(group);

  const material = new THREE.MeshPhysicalMaterial({
    color: 0x1f7ad0,
    transparent: true,
    opacity: 0.94,
    roughness: 0.05,        // mirror-smooth — env reflections do the "water" work
    metalness: 0.0,
    transmission: 0.0,      // off: expensive + flaky in software GL; env map instead
    ior: 1.33,
    clearcoat: 0.9,
    clearcoatRoughness: 0.06,
    reflectivity: 1.0,
    envMapIntensity: 1.6,   // strong Fresnel-ish highlights off the room env
  });

  let mc = null;
  let mcRes = 0;

  function ensureMC() {
    if (mc && mcRes === state.resolution) return;
    if (mc) { group.remove(mc); mc.geometry.dispose(); }
    // NB: first ctor arg is the grid resolution (field = res³), NOT a parent.
    // MC vertex space spans [-1,1] → scale = size/2 so the volume maps to `bounds`.
    mc = new MarchingCubes(state.resolution, material, false, false, 60000);
    mc.scale.set(bounds.size[0] / 2, bounds.size[1] / 2, bounds.size[2] / 2);
    mc.position.set(bounds.min[0] + bounds.size[0] / 2, bounds.min[1] + bounds.size[1] / 2, bounds.min[2] + bounds.size[2] / 2);
    mc.isolation = state.isolation;
    mc.enableUvs = false; mc.enableColors = false;
    mc.frustumCulled = false; // bounding sphere goes stale as the surface morphs
    mcRes = state.resolution;
    group.add(mc);
  }
  ensureMC();

  // particle points
  const MAXP = sim.p.maxParticles;
  const ptPos = new Float32Array(MAXP * 3);
  const ptCol = new Float32Array(MAXP * 3);
  const ptGeo = new THREE.BufferGeometry();
  ptGeo.setAttribute('position', new THREE.BufferAttribute(ptPos, 3));
  ptGeo.setAttribute('color', new THREE.BufferAttribute(ptCol, 3));
  const ptMat = new THREE.PointsMaterial({ size: 0.14, vertexColors: true, sizeAttenuation: true });
  const points = new THREE.Points(ptGeo, ptMat);
  points.frustumCulled = false;
  scene.add(points);

  const coldColor = new THREE.Color(0x1e5fbf);
  const hotColor = new THREE.Color(0x66d9ff);

  // foam/spray: low-density + fast particles drawn as additive white points
  const foamState = { show: true, maxSpeed: 6.5, maxNeighbors: 14, size: 0.16 };
  const foamPos = new Float32Array(sim.p.maxParticles * 3);
  const foamGeo = new THREE.BufferGeometry();
  foamGeo.setAttribute('position', new THREE.BufferAttribute(foamPos, 3));
  const foamMat = new THREE.PointsMaterial({
    color: 0xeef6ff, size: foamState.size, sizeAttenuation: true,
    transparent: true, opacity: 0.85, depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const foamPoints = new THREE.Points(foamGeo, foamMat);
  foamPoints.frustumCulled = false;
  group.add(foamPoints);

  let mcTriCount = 0;
  function update() {
    const n = sim.count;
    const pos = sim.pos, vel = sim.vel;
    const showPoints = state.mode === 'particles' || state.colorBySpeed === 'always';
    points.visible = state.mode === 'particles';
    mc.visible = state.mode === 'metaballs';

    if (points.visible) {
      for (let i = 0; i < n; i++) {
        ptPos[i * 3] = pos[i * 3]; ptPos[i * 3 + 1] = pos[i * 3 + 1]; ptPos[i * 3 + 2] = pos[i * 3 + 2];
        const sp = Math.hypot(vel[i * 3], vel[i * 3 + 1], vel[i * 3 + 2]);
        const t = Math.min(sp / 8, 1);
        const c = _tmpColor.copy(coldColor).lerp(hotColor, t);
        ptCol[i * 3] = c.r; ptCol[i * 3 + 1] = c.g; ptCol[i * 3 + 2] = c.b;
      }
      ptGeo.setDrawRange(0, n);
      ptGeo.attributes.position.needsUpdate = true;
      ptGeo.attributes.color.needsUpdate = true;
    }

    if (mc.visible) {
      ensureMC();
      mc.isolation = state.isolation;
      mc.reset();
      const res = mcRes;
      // ball strength derived from particle radius: MC visible radius (normalized)
      //   r_n = sqrt(strength / (isolation + subtract))  →  strength = r_n² · (iso+sub)
      const sub = 12;
      // lone-ball surface radius: r = sqrt(strength/(isolation+sub)) in normalized
      // coords. Want ≈ 0.55·h world → strength = r_n²·(iso+sub).
      const rN = (sim.h * 0.55) / bounds.size[0];
      const strength = rN * rN * (state.isolation + sub);
      // ball coords are [0,1] over the volume (cell = ball·res); mesh scale is
      // size/2 with vertex space [-1,1], which maps ball back to world exactly.
      for (let i = 0; i < n; i++) {
        const x = (pos[i * 3] - bounds.min[0]) / bounds.size[0];
        const y = (pos[i * 3 + 1] - bounds.min[1]) / bounds.size[1];
        const z = (pos[i * 3 + 2] - bounds.min[2]) / bounds.size[2];
        if (x < 0.02 || x > 0.98 || y < 0.02 || y > 0.98 || z < 0.02 || z > 0.98) continue;
        mc.addBall(x, y, z, strength, sub);
      }
      mc.update();
      material.opacity = state.opacity;
      mcTriCount = mc.count / 3; // triangles generated (for HUD)
    }

    // foam overlay (both modes)
    foamPoints.visible = state.mode === 'metaballs' && foamState.show;
    if (foamPoints.visible) {
      let fn = 0;
      for (let i = 0; i < n; i++) {
        const sp = Math.hypot(vel[i * 3], vel[i * 3 + 1], vel[i * 3 + 2]);
        if (sp > foamState.maxSpeed || sim.nCount[i] < foamState.maxNeighbors) {
          foamPos[fn * 3] = pos[i * 3]; foamPos[fn * 3 + 1] = pos[i * 3 + 1]; foamPos[fn * 3 + 2] = pos[i * 3 + 2];
          fn++;
        }
      }
      foamGeo.setDrawRange(0, fn);
      foamGeo.attributes.position.needsUpdate = true;
    }
  }

  function addGui(gui) {
    const f = gui.addFolder('💧 Water Render');
    f.add(state, 'mode', ['metaballs', 'particles']).name('render mode');
    f.add(state, 'resolution', 24, 96, 4).name('MC resolution').onFinishChange(() => ensureMC());
    f.add(state, 'isolation', 40, 200, 1).name('isolation');
    f.add(state, 'opacity', 0.3, 1).name('opacity');
    const ff = f.addFolder('Foam / spray');
    ff.add(foamState, 'show').name('show foam');
    ff.add(foamState, 'maxSpeed', 1, 12, 0.5).name('speed threshold');
    ff.add(foamState, 'maxNeighbors', 2, 40, 1).name('density threshold');
    ff.add(foamState, 'size', 0.05, 0.4, 0.01).name('sprite size').onChange(v => foamMat.size = v);
    ff.close();
    f.close();
    return f;
  }

  return { update, addGui, state, get mcTris() { return mcTriCount; }, group };
}

const _tmpColor = new THREE.Color();

// Fill probe: measures water level + flatness inside an axis-aligned pool region.
export function createFillProbe(sim, region) {
  // region: {min:[..], max:[..]} — pool interior in world space
  return {
    measure() {
      let sumY = 0, sumY2 = 0, nIn = 0, top = -Infinity;
      const p = sim.pos;
      for (let i = 0; i < sim.count; i++) {
        const x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
        if (x > region.min[0] && x < region.max[0] && z > region.min[2] && z < region.max[2] && y < region.max[1]) {
          sumY += y; sumY2 += y * y; nIn++;
          if (y > top) top = y;
        }
      }
      if (!nIn) return { count: 0, meanY: NaN, stdY: NaN, topY: NaN };
      const mean = sumY / nIn;
      const varr = sumY2 / nIn - mean * mean;
      return { count: nIn, meanY: mean, stdY: Math.sqrt(Math.max(varr, 0)), topY: top };
    },
  };
}
