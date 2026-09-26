// water/gpu/wgsl.js — the DFSPH solver of core/dfsph.js as WebGPU compute
// shaders (one module, one entry point per phase).
//
// Per particle state (vec4 arrays, sorted every substep):
//   pos   xyz, w = id (w < 0: dead — removed by the next sort)
//   vel   xyz, w = carried density-solve pressure (warm start)
//   prev  xyz = position at the start of the render frame, w = carried
//         divergence-solve pressure
// Spatial hash: counting sort by Teschner-hashed cell (atomic histogram,
// three-level exclusive scan, scatter) — dead particles go to a sentinel
// bucket past the table, so the live ones end up compacted at the front.
// Neighbor lists (≤ M per particle) are rebuilt every substep; kernel
// gradients are recomputed from positions in every solver pass.
// Atomic counters (i32/u32 in one buffer): see A_* below.

export const A = {
  COUNT: 0,        // live particles (after the sort; + appended spawns)
  COUNT_PRE: 1,    // particles before the sort (sort range)
  NEXT_ID: 2,
  ERR_SUM: 3,      // Σ residual / ρ0 × 1e5 (last density residual pass)
  ERR_MAX: 4,      // max residual / ρ0 as f32 bits (positive floats order as u32)
  VMAX: 5,         // max speed as f32 bits
  DRAINED: 6,
  LEAKED: 7,
  QUARANTINED: 8,
  OVERFLOW: 9,
  DIFF_COUNT: 10,  // whitewater particles
  DIFF_NEXT: 11,   // survivors appended during advection
  FRAME: 12,
  IMPULSES: 16,    // then nCol × 6 fixed-point impulses, then nCol × 4 contacts
};
export const IMPULSE_SCALE = 1e4;   // N·s → i32
export const CONTACT_SCALE = 1e3;   // m/s → i32
export const K_BND = 4;             // boundary slots per particle
export const STRIDE = 24;           // collider record (core/colliders.js)

export const WGSL = /* wgsl */`
struct Params {
  dt: f32, gx: f32, gy: f32, gz: f32,
  h: f32, invH: f32, sigma: f32, rho0: f32,
  spacing: f32, visc: f32, vort: f32, fric: f32,
  maxSpeed: f32, bndInv: f32, contactMin: f32, maxOver: f32,
  warmD: f32, warmV: f32, pmass: f32, minN: f32,
  bminX: f32, bminY: f32, bminZ: f32, boundsOn: f32,
  bmaxX: f32, bmaxY: f32, bmaxZ: f32, touch: f32,
  nCol: u32, mask: u32, M: u32, upper: u32,
  hfBase: u32, tabBase: u32, tabN: u32, nDrains: u32,
  frames: f32, nScan: u32, nBlocks: u32, D: u32,
  wwTa: f32, wwWc: f32, taLo: f32, taHi: f32,
  wcLo: f32, wcHi: f32, ekLo: f32, ekHi: f32,
  lifeLo: f32, lifeHi: f32, wwBuoy: f32, wwDrag: f32,
  wwAir: f32, sprayMax: f32, bubbleMin: f32, invSurf: f32,
  stepDt: f32, pad0: f32, pad1: f32, pad2: f32,
  drains: array<vec4f, 16>,   // min.xyz / max.xyz pairs (8 boxes)
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> prv: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> posB: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> velB: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> prvB: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> keys: array<vec2u>;
@group(0) @binding(8) var<storage, read_write> cells: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> blocks: array<u32>;
@group(0) @binding(10) var<storage, read_write> nbr: array<u32>;
@group(0) @binding(11) var<storage, read_write> aux: array<vec4f>;   // ρ, β, k, gate
@group(0) @binding(12) var<storage, read_write> bnd: array<vec4f>;   // per slot: (∇Ψρ0, Ψ), (v_b, collider)
@group(0) @binding(13) var<storage, read> world: array<f32>;
@group(0) @binding(14) var<storage, read_write> atoms: array<atomic<i32>>;
@group(0) @binding(15) var<storage, read_write> dv: array<vec4f>;    // per particle: Δv, ω
@group(0) @binding(16) var<storage, read> spawnBuf: array<vec4f>;
@group(0) @binding(17) var<storage, read_write> diffA: array<vec4f>;  // (pos, life), (vel, type)
@group(0) @binding(18) var<storage, read_write> diffB: array<vec4f>;

const EMPTY: u32 = 0xffffffffu;

fn count() -> u32 { return u32(atomicLoad(&atoms[${A.COUNT}])); }

fn hashCell(c: vec3i) -> u32 {
  return (bitcast<u32>(c.x) * 92837111u) ^ (bitcast<u32>(c.y) * 689287499u) ^ (bitcast<u32>(c.z) * 283923481u);
}
fn cellOf(p: vec3f) -> vec3i { return vec3i(floor(p * P.invH)); }

// cubic spline W and dW/dr
fn kernW(q: f32) -> f32 {
  if (q <= 0.5) { return P.sigma * (6.0 * q * q * q - 6.0 * q * q + 1.0); }
  if (q < 1.0) { let a = 1.0 - q; return 2.0 * P.sigma * a * a * a; }
  return 0.0;
}
fn kernDW(q: f32) -> f32 {
  let s = P.sigma * P.invH;
  if (q <= 0.5) { return s * (18.0 * q * q - 12.0 * q); }
  if (q < 1.0) { let a = 1.0 - q; return -6.0 * s * a * a; }
  return 0.0;
}
// ∇_i W_ij for d = x_i − x_j (coincident pairs: a fixed antisymmetric direction)
fn gradW(d: vec3f, idi: f32, idj: f32) -> vec3f {
  let r = length(d);
  if (r > 1e-6 * P.h) { return d * (kernDW(r * P.invH) / r); }
  let lo = min(idi, idj); let hi = max(idi, idj);
  let hsh = (u32(lo) * 73856093u) ^ (u32(hi) * 19349663u);
  let th = f32(hsh & 1023u) * (6.2831853 / 1024.0);
  let ph = f32((hsh >> 10u) & 1023u) * (3.1415926 / 1024.0);
  let g = 2.0 * P.sigma * P.invH * select(-1.0, 1.0, idi < idj);
  return g * vec3f(sin(ph) * cos(th), cos(ph), sin(ph) * sin(th));
}

// ---------------------------------------------------------------- colliders
fn rot(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}
fn cf(c: u32, k: u32) -> f32 { return world[c * ${STRIDE}u + k]; }
fn cPos(c: u32) -> vec3f { return vec3f(cf(c, 1u), cf(c, 2u), cf(c, 3u)); }
fn cRot(c: u32) -> vec4f { return vec4f(cf(c, 4u), cf(c, 5u), cf(c, 6u), cf(c, 7u)); }
fn cSize(c: u32) -> vec3f { return vec3f(cf(c, 8u), cf(c, 9u), cf(c, 10u)); }
fn cVel(c: u32, x: vec3f) -> vec3f {
  let w = vec3f(cf(c, 14u), cf(c, 15u), cf(c, 16u));
  return vec3f(cf(c, 11u), cf(c, 12u), cf(c, 13u)) + cross(w, x - cPos(c));
}
fn cNear(c: u32, x: vec3f, reach: f32) -> bool {
  let b = cf(c, 23u);
  if (b >= 1e29) { return true; }
  let d = x - cPos(c);
  let r = b + reach;
  return dot(d, d) < r * r;
}
fn cDynamic(c: u32) -> bool { return (u32(cf(c, 18u)) & 1u) != 0u; }
fn cSlot(c: u32) -> u32 { return u32(cf(c, 20u)); }

struct Hit { d: f32, n: vec3f };

fn heightfieldSDF(hfi: u32, x: vec3f) -> Hit {
  var o = P.hfBase + u32(world[P.hfBase + hfi]);
  let minX = world[o]; let minZ = world[o + 1u]; let dx = world[o + 2u]; let dz = world[o + 3u];
  let nx = u32(world[o + 4u]); let nz = u32(world[o + 5u]);
  o = o + 6u;
  let fx = (x.x - minX) / dx; let fz = (x.z - minZ) / dz;
  if (!(fx >= 0.0 && fx < f32(nx - 1u) && fz >= 0.0 && fz < f32(nz - 1u))) { return Hit(1e30, vec3f(0.0, 1.0, 0.0)); }
  let ix = u32(fx); let iz = u32(fz);
  let tx = fx - f32(ix); let tz = fz - f32(iz);
  let b = o + iz * nx + ix;
  let h00 = world[b]; let h10 = world[b + 1u]; let h01 = world[b + nx]; let h11 = world[b + nx + 1u];
  let g = h00 * (1.0 - tx) * (1.0 - tz) + h10 * tx * (1.0 - tz) + h01 * (1.0 - tx) * tz + h11 * tx * tz;
  let gx = ((h10 - h00) * (1.0 - tz) + (h11 - h01) * tz) / dx;
  let gz = ((h01 - h00) * (1.0 - tx) + (h11 - h10) * tx) / dz;
  let inv = 1.0 / sqrt(gx * gx + 1.0 + gz * gz);
  return Hit((x.y - g) * inv, vec3f(-gx * inv, inv, -gz * inv));
}

fn boxSDF(l: vec3f, e: vec3f) -> Hit {
  let a = abs(l) - e;
  let o = max(a, vec3f(0.0));
  let out = length(o);
  if (out > 0.0) { return Hit(out, sign(l) * o / out); }
  if (a.x >= a.y && a.x >= a.z) { return Hit(a.x, vec3f(sign(l.x), 0.0, 0.0)); }
  if (a.y >= a.z) { return Hit(a.y, vec3f(0.0, sign(l.y), 0.0)); }
  return Hit(a.z, vec3f(0.0, 0.0, sign(l.z)));
}

// signed distance (+ fluid side) and fluid-side normal of collider c
fn sdf(c: u32, x: vec3f) -> Hit {
  let t = u32(cf(c, 0u));
  if (t == 5u) { return heightfieldSDF(u32(cf(c, 19u)), x); }
  let q = cRot(c);
  let qi = vec4f(-q.xyz, q.w);
  let l = rot(qi, x - cPos(c));
  let s = cSize(c);
  var h = Hit(1e30, vec3f(0.0, 1.0, 0.0));
  if (t == 1u) { h = Hit(l.y, vec3f(0.0, 1.0, 0.0)); }
  else if (t == 2u) { h = boxSDF(l, s); }
  else if (t == 3u) {
    let r = length(l);
    h = Hit(r - s.x, select(vec3f(0.0, 1.0, 0.0), l / r, r > 1e-9));
  } else if (t == 4u) {
    let sy = clamp(l.y, -s.y, s.y);
    let d = l - vec3f(0.0, sy, 0.0);
    let r = length(d);
    h = Hit(r - s.x, select(vec3f(1.0, 0.0, 0.0), d / r, r > 1e-9));
  } else if (t == 6u) {
    var d = s.x - l.x; var n = vec3f(-1.0, 0.0, 0.0);
    if (s.x + l.x < d) { d = s.x + l.x; n = vec3f(1.0, 0.0, 0.0); }
    if (s.z - l.z < d) { d = s.z - l.z; n = vec3f(0.0, 0.0, -1.0); }
    if (s.z + l.z < d) { d = s.z + l.z; n = vec3f(0.0, 0.0, 1.0); }
    if (s.y + l.y < d) { d = s.y + l.y; n = vec3f(0.0, 1.0, 0.0); }
    h = Hit(d, n);
  } else { return h; }
  return Hit(h.d, rot(q, h.n));
}

// Ψ table (lattice-calibrated boundary volume) and slope at d ∈ [0, h)
fn psiAt(d: f32) -> vec2f {
  let f = max(d, 0.0) * P.bndInv;
  let i = u32(f); let t = f - f32(i);
  let a = P.tabBase + i; let b = P.tabBase + P.tabN + i;
  return vec2f(mix(world[a], world[a + 1u], t), mix(world[b], world[b + 1u], t));
}

// boundary volume and ∇Ψ of collider c at x (containers: union of walls)
struct Vol { psi: f32, g: vec3f };
fn volume(c: u32, x: vec3f) -> Vol {
  if (u32(cf(c, 0u)) == 6u) {
    let q = cRot(c); let l = rot(vec4f(-q.xyz, q.w), x - cPos(c)); let s = cSize(c);
    var wd = array<f32, 5>(s.x - l.x, s.x + l.x, s.z - l.z, s.z + l.z, s.y + l.y);
    var wn = array<vec3f, 5>(vec3f(-1.0, 0.0, 0.0), vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, -1.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0));
    var ps = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
    var sl = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
    var prod = 1.0;
    for (var w = 0; w < 5; w++) {
      if (wd[w] < P.h) { let v = psiAt(wd[w]); ps[w] = v.x; sl[w] = v.y; prod *= 1.0 - v.x; }
    }
    if (prod == 1.0) { return Vol(0.0, vec3f(0.0)); }
    var g = vec3f(0.0);
    for (var w = 0; w < 5; w++) {
      if (sl[w] != 0.0) {
        let rest = select(0.0, prod / (1.0 - ps[w]), 1.0 - ps[w] > 1e-9);
        g += sl[w] * rest * rot(q, wn[w]);
      }
    }
    return Vol(1.0 - prod, g);
  }
  let hit = sdf(c, x);
  if (hit.d >= P.h) { return Vol(0.0, vec3f(0.0)); }
  let v = psiAt(hit.d);
  return Vol(v.x, v.y * hit.n);
}

fn addImpulse(c: u32, x: vec3f, j: vec3f) {
  let base = ${A.IMPULSES}u + cSlot(c) * 6u;
  let m = cross(x - cPos(c), j);
  atomicAdd(&atoms[base], i32(round(j.x * ${IMPULSE_SCALE}.0)));
  atomicAdd(&atoms[base + 1u], i32(round(j.y * ${IMPULSE_SCALE}.0)));
  atomicAdd(&atoms[base + 2u], i32(round(j.z * ${IMPULSE_SCALE}.0)));
  atomicAdd(&atoms[base + 3u], i32(round(m.x * ${IMPULSE_SCALE}.0)));
  atomicAdd(&atoms[base + 4u], i32(round(m.y * ${IMPULSE_SCALE}.0)));
  atomicAdd(&atoms[base + 5u], i32(round(m.z * ${IMPULSE_SCALE}.0)));
}

// ------------------------------------------------------------- bookkeeping

// append spawned particles at the live count
// spawnBuf[0].x = number staged, then (pos, vel) pairs
@compute @workgroup_size(64) fn appendSpawn(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x;
  let n = u32(spawnBuf[0].x);
  if (k >= n) { return; }
  let base = u32(atomicLoad(&atoms[${A.COUNT}]));
  let i = base + k;
  if (i >= arrayLength(&pos)) { return; }
  let p = spawnBuf[1u + k * 2u]; let v = spawnBuf[2u + k * 2u];
  var id = f32(atomicAdd(&atoms[${A.NEXT_ID}], 1));
  // spawns inside a solid are dropped (flagged dead; the next sort removes them)
  for (var c = 0u; c < P.nCol; c++) {
    if (cNear(c, p.xyz, 0.0) && sdf(c, p.xyz).d < 0.0) { id = -1.0; break; }
  }
  pos[i] = vec4f(p.xyz, id);
  vel[i] = vec4f(v.xyz, 0.0);
  prv[i] = vec4f(p.xyz, 0.0);
}
// after appendSpawn: count += staged (single invocation)
@compute @workgroup_size(1) fn commitSpawn() {
  let n = min(u32(atomicLoad(&atoms[${A.COUNT}])) + u32(spawnBuf[0].x), arrayLength(&pos));
  atomicStore(&atoms[${A.COUNT}], i32(n));
}

// interpolation origin for rendering: prev.xyz = pos.xyz
@compute @workgroup_size(64) fn markFrame(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= count()) { return; }
  prv[i] = vec4f(pos[i].xyz, prv[i].w);
}

// drains: kill particles inside the boxes
@compute @workgroup_size(64) fn drain(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= count()) { return; }
  let p = pos[i];
  if (p.w < 0.0) { return; }
  for (var k = 0u; k < P.nDrains; k++) {
    let lo = P.drains[k * 2u].xyz; let hi = P.drains[k * 2u + 1u].xyz;
    if (all(p.xyz >= lo) && all(p.xyz <= hi)) { pos[i].w = -1.0; atomicAdd(&atoms[${A.DRAINED}], 1); return; }
  }
}

// ------------------------------------------------------------------- sort

@compute @workgroup_size(64) fn hash(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= P.upper) { return; }
  if (i >= count()) { keys[i] = vec2u(EMPTY, 0u); return; }
  let p = pos[i];
  var key = P.mask + 1u; // dead → sentinel bucket
  if (p.w >= 0.0) { key = hashCell(cellOf(p.xyz)) & P.mask; }
  keys[i] = vec2u(key, atomicAdd(&cells[key], 1u));
}

// exclusive scan of cells[0 .. nScan) in blocks of 512 (Blelloch, shared memory)
var<workgroup> sh: array<u32, 1024>;
fn scanShared(t: u32, n: u32) {
  // up-sweep
  var off = 1u;
  var d = n >> 1u;
  loop {
    if (d == 0u) { break; }
    workgroupBarrier();
    if (t < d) { let ai = off * (2u * t + 1u) - 1u; let bi = off * (2u * t + 2u) - 1u; sh[bi] += sh[ai]; }
    off *= 2u; d = d >> 1u;
  }
  if (t == 0u) { sh[n - 1u] = 0u; }
  d = 1u;
  loop {
    if (d >= n) { break; }
    off = off >> 1u;
    workgroupBarrier();
    if (t < d) {
      let ai = off * (2u * t + 1u) - 1u; let bi = off * (2u * t + 2u) - 1u;
      let x = sh[ai]; sh[ai] = sh[bi]; sh[bi] += x;
    }
    d *= 2u;
  }
  workgroupBarrier();
}
@compute @workgroup_size(256) fn scanBlocks(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) w: vec3u) {
  let t = l.x; let base = w.x * 512u;
  let a = base + 2u * t; let b = a + 1u;
  let va = select(0u, atomicLoad(&cells[a]), a < P.nScan);
  let vb = select(0u, atomicLoad(&cells[b]), b < P.nScan);
  sh[2u * t] = va; sh[2u * t + 1u] = vb;
  scanShared(t, 512u);
  if (a < P.nScan) { atomicStore(&cells[a], sh[2u * t]); }
  if (b < P.nScan) { atomicStore(&cells[b], sh[2u * t + 1u]); }
  if (t == 255u) { blocks[w.x] = sh[511u] + vb; }
}
@compute @workgroup_size(256) fn scanTop(@builtin(local_invocation_id) l: vec3u) {
  let t = l.x;
  for (var k = 0u; k < 4u; k++) { let i = t * 4u + k; sh[i] = select(0u, blocks[i], i < P.nBlocks); }
  scanShared256x4(t);
  for (var k = 0u; k < 4u; k++) { let i = t * 4u + k; if (i < P.nBlocks) { blocks[i] = sh[i]; } }
}
fn scanShared256x4(t: u32) {
  // exclusive scan of sh[0..1024) with 256 threads: sequential per thread, then Blelloch over 256 partials
  let b0 = t * 4u;
  let s0 = sh[b0]; let s1 = sh[b0 + 1u]; let s2 = sh[b0 + 2u]; let s3 = sh[b0 + 3u];
  workgroupBarrier();
  sh[t] = s0 + s1 + s2 + s3;
  scanShared(t, 256u);
  let base = sh[t];
  workgroupBarrier();
  sh[b0] = base; sh[b0 + 1u] = base + s0; sh[b0 + 2u] = base + s0 + s1; sh[b0 + 3u] = base + s0 + s1 + s2;
  workgroupBarrier();
}
@compute @workgroup_size(256) fn scanAdd(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= P.nScan) { return; }
  let off = blocks[i / 512u];
  let v = atomicAdd(&cells[i], off) + off;
  // the sentinel bucket starts at the live count
  if (i == P.mask + 1u) { atomicStore(&atoms[${A.COUNT}], i32(v)); }
}
@compute @workgroup_size(64) fn scatter(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= P.upper) { return; }
  let k = keys[i];
  if (k.x == EMPTY) { return; }
  let dst = atomicLoad(&cells[k.x]) + k.y;
  posB[dst] = pos[i]; velB[dst] = vel[i]; prvB[dst] = prv[i];
}

// ------------------------------------------------------------- neighbors

@compute @workgroup_size(64) fn neighbors(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= count()) { return; }
  let xi = pos[i].xyz;
  let c = cellOf(xi);
  let h2 = P.h * P.h;
  let base = i * (P.M + 1u);
  var k = 0u;
  // the 27 neighbor cells' buckets; a bucket shared by several cells (hash
  // collision) is walked once
  var bk: array<u32, 27>;
  var nb = 0u;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let b = hashCell(c + vec3i(dx, dy, dz)) & P.mask;
        var dup = false;
        for (var q = 0u; q < nb; q++) { if (bk[q] == b) { dup = true; break; } }
        if (!dup) { bk[nb] = b; nb++; }
      }
    }
  }
  for (var q = 0u; q < nb; q++) {
    let b = bk[q];
    let e = atomicLoad(&cells[b + 1u]);
    for (var j = atomicLoad(&cells[b]); j < e; j++) {
      if (j == i) { continue; }
      let d = xi - pos[j].xyz;
      if (dot(d, d) < h2) {
        if (k < P.M) { nbr[base + 1u + k] = j; k++; } else { atomicAdd(&atoms[${A.OVERFLOW}], 1); }
      }
    }
  }
  nbr[base] = k;
  // contacts with dynamic bodies (count + fluid velocity sum): drag/impulse bounds
  let v = vel[i].xyz;
  for (var cI = 0u; cI < P.nCol; cI++) {
    if (!cDynamic(cI) || !cNear(cI, xi, P.touch)) { continue; }
    if (sdf(cI, xi).d < P.touch) {
      let o = ${A.IMPULSES}u + P.nCol * 6u + cSlot(cI) * 4u;
      atomicAdd(&atoms[o], 1);
      atomicAdd(&atoms[o + 1u], i32(round(v.x * ${CONTACT_SCALE}.0)));
      atomicAdd(&atoms[o + 2u], i32(round(v.y * ${CONTACT_SCALE}.0)));
      atomicAdd(&atoms[o + 3u], i32(round(v.z * ${CONTACT_SCALE}.0)));
    }
  }
}

// ---------------------------------------------------------------- solver

// density, β, boundary cache and the divergence-solve gate (Dρ/Dt > 0)
@compute @workgroup_size(64) fn density(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= count()) { return; }
  let pi = pos[i]; let xi = pi.xyz; let vi = vel[i].xyz;
  let base = i * (P.M + 1u);
  let nn = nbr[base];
  var rho = P.sigma; // W(0)
  var G = vec3f(0.0); var S = 0.0; var drho = 0.0;
  for (var t = 0u; t < nn; t++) {
    let j = nbr[base + 1u + t];
    let pj = pos[j];
    let d = xi - pj.xyz;
    rho += kernW(length(d) * P.invH);
    let gw = gradW(d, pi.w, pj.w);
    G += gw; S += dot(gw, gw);
    drho += dot(vi - vel[j].xyz, gw);
  }
  var slot = 0u;
  for (var c = 0u; c < P.nCol && slot < ${K_BND}u; c++) {
    if (!cNear(c, xi, P.h)) { continue; }
    let vol = volume(c, xi);
    if (vol.psi <= 0.0) { continue; }
    let gb = P.rho0 * vol.g;
    let vb = cVel(c, xi);
    rho += P.rho0 * vol.psi;
    G += gb;
    drho += dot(vi - vb, gb);
    bnd[(i * ${K_BND}u + slot) * 2u] = vec4f(gb, vol.psi);
    bnd[(i * ${K_BND}u + slot) * 2u + 1u] = vec4f(vb, f32(c));
    slot++;
  }
  if (slot < ${K_BND}u) { bnd[(i * ${K_BND}u + slot) * 2u + 1u] = vec4f(0.0, 0.0, 0.0, -1.0); }
  let den = dot(G, G) + S;
  let gate = select(0.0, 1.0, f32(nn) >= P.minN && drho > 0.0);
  aux[i] = vec4f(rho, select(0.0, 1.0 / den, den > 1e-9), gate, gate);
}

// warm start: k = factor × carried pressure where the particle compresses
@compute @workgroup_size(64) fn warmDensity(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= count()) { return; }
  let a = aux[i];
  let k = select(0.0, vel[i].w * P.warmD, a.z > 0.0);
  aux[i].z = k; vel[i].w = k;
}
@compute @workgroup_size(64) fn warmDivergence(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= count()) { return; }
  let a = aux[i];
  let k = select(0.0, prv[i].w * P.warmV, a.w > 0.0);
  aux[i].z = k; prv[i].w = k;
}

// v_i −= dt·[Σ (k_i + k_j)∇W_ij + k_i ρ0∇Ψ]; boundary reactions → bodies
@compute @workgroup_size(64) fn velocity(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= count()) { return; }
  let pi = pos[i]; let xi = pi.xyz;
  let ki = aux[i].z;
  let base = i * (P.M + 1u);
  let nn = nbr[base];
  var acc = vec3f(0.0);
  for (var t = 0u; t < nn; t++) {
    let j = nbr[base + 1u + t];
    let s = ki + aux[j].z;
    if (s == 0.0) { continue; }
    let pj = pos[j];
    acc += s * gradW(xi - pj.xyz, pi.w, pj.w);
  }
  var v = vel[i].xyz - P.dt * acc;
  if (ki != 0.0) {
    for (var s = 0u; s < ${K_BND}u; s++) {
      let b1 = bnd[(i * ${K_BND}u + s) * 2u + 1u];
      if (b1.w < 0.0) { break; }
      let dvb = -P.dt * ki * bnd[(i * ${K_BND}u + s) * 2u].xyz;
      v += dvb;
      let c = u32(b1.w);
      if (cDynamic(c)) { addImpulse(c, xi, -P.pmass * dvb); }
    }
  }
  vel[i] = vec4f(v, vel[i].w);
}

// residual → pressure increment k (density: ρ* − ρ0; divergence: Dρ/Dt)
fn residual(i: u32, density: bool, accumulate: bool) {
  let pi = pos[i]; let xi = pi.xyz; let vi = vel[i].xyz;
  let base = i * (P.M + 1u);
  let nn = nbr[base];
  var drho = 0.0;
  for (var t = 0u; t < nn; t++) {
    let j = nbr[base + 1u + t];
    let pj = pos[j];
    drho += dot(vi - vel[j].xyz, gradW(xi - pj.xyz, pi.w, pj.w));
  }
  for (var s = 0u; s < ${K_BND}u; s++) {
    let b1 = bnd[(i * ${K_BND}u + s) * 2u + 1u];
    if (b1.w < 0.0) { break; }
    drho += dot(vi - b1.xyz, bnd[(i * ${K_BND}u + s) * 2u].xyz);
  }
  let a = aux[i];
  var r = 0.0; var k = 0.0;
  if (density) {
    r = max(min(a.x - P.rho0, P.maxOver * P.rho0) + P.dt * drho, 0.0);
    k = r * a.y / (P.dt * P.dt);
  } else {
    r = select(0.0, max(drho, 0.0), f32(nn) >= P.minN);
    k = r * a.y / P.dt;
    r = r * P.dt;
  }
  aux[i].z = k;
  if (!accumulate) { aux[i].w = select(0.0, 1.0, k > 0.0); return; }
  if (density) {
    vel[i].w += k;
    let e = r / P.rho0;
    atomicAdd(&atoms[${A.ERR_SUM}], i32(round(e * 1e5)));
    atomicMax(&atoms[${A.ERR_MAX}], bitcast<i32>(e));
  } else { prv[i].w += k; }
}
@compute @workgroup_size(64) fn densityGate(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= count()) { return; }
  residual(i, true, false);
  aux[i].z = select(0.0, 1.0, aux[i].w > 0.0); // gate flag for warmDensity
}
@compute @workgroup_size(64) fn densityResidual(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= count()) { return; }
  residual(i, true, true);
}
@compute @workgroup_size(64) fn divergenceResidual(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= count()) { return; }
  residual(i, false, true);
}

// gravity, XSPH, wall drag, vorticity ω → dv
@compute @workgroup_size(64) fn forces(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= count()) { return; }
  let pi = pos[i]; let xi = pi.xyz; let vi = vel[i].xyz;
  let base = i * (P.M + 1u);
  let nn = nbr[base];
  var ax = vec3f(0.0); var w = vec3f(0.0);
  var ta = 0.0; var cg = vec3f(0.0);
  // whitewater potentials only above the kinetic-energy ramp (rate 0 below)
  let ww = P.D > 0u && 0.5 * dot(vi, vi) > P.ekLo;
  for (var t = 0u; t < nn; t++) {
    let j = nbr[base + 1u + t];
    let pj = pos[j];
    let d = xi - pj.xyz;
    let invRho = 1.0 / aux[j].x;
    let u = vel[j].xyz - vi;
    let r = length(d);
    ax += u * (kernW(r * P.invH) * invRho);
    let gq = gradW(d, pi.w, pj.w) * invRho;
    w += cross(u, gq);
    // whitewater: colour-field gradient and trapped air (neighbors converging)
    if (ww) {
      cg += gq;
      let vr = length(u);
      if (r > 1e-9 && vr > 1e-6) { ta += vr * (1.0 + dot(u, d) / (vr * r)) * (1.0 - r * P.invH); }
    }
  }
  var gen = 0.0;
  if (ww) {
    let cl = length(cg);
    var crest = 0.0;
    if (cl > 1e-9) { let o = -dot(vi, cg) / cl; if (o > 0.0) { crest = min(1.0, cl * P.invSurf) * o; } }
    let ek = ramp(0.5 * dot(vi, vi), P.ekLo, P.ekHi);
    gen = ek * (P.wwTa * ramp(ta, P.taLo, P.taHi) + P.wwWc * ramp(crest, P.wcLo, P.wcHi));
  }
  let xsph = 1.0 - pow(1.0 - min(P.visc, 0.99), P.frames);
  var dvi = vec3f(P.gx, P.gy, P.gz) * P.dt + xsph * ax;
  if (P.fric > 0.0) {
    for (var s = 0u; s < ${K_BND}u; s++) {
      let b0 = bnd[(i * ${K_BND}u + s) * 2u];
      let b1 = bnd[(i * ${K_BND}u + s) * 2u + 1u];
      if (b1.w < 0.0) { break; }
      let bl = length(b0.xyz);
      if (b0.w <= 0.0 || bl == 0.0) { continue; }
      let c = u32(b1.w);
      let cfr = select(P.fric, cf(c, 17u), cf(c, 17u) >= 0.0);
      if (cfr <= 0.0) { continue; }
      let n = -b0.xyz / bl;
      let rel = vi - b1.xyz;
      let tv = rel - dot(rel, n) * n;
      let ut = length(tv);
      if (ut == 0.0) { continue; }
      let decay = 1.0 - 1.0 / (1.0 + P.dt * cfr * ut * 2.0 * b0.w / P.spacing);
      let f = -decay * tv;
      dvi += f;
      if (cDynamic(c)) { addImpulse(c, xi, -P.pmass * f); }
    }
  }
  dv[i * 2u] = vec4f(dvi, gen);
  dv[i * 2u + 1u] = vec4f(w, length(w));
}
fn ramp(x: f32, lo: f32, hi: f32) -> f32 { return clamp((x - lo) / (hi - lo), 0.0, 1.0); }

// ------------------------------------------------------------ whitewater
fn rnd(seed: u32) -> f32 {
  var x = seed * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  x = (x >> 22u) ^ x;
  return f32(x) / 4294967296.0;
}
// once per step: emit diffuse particles from each fluid particle's rate
// (the last substep's potentials), stochastic rounding as on the CPU
@compute @workgroup_size(64) fn wwEmit(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= count()) { return; }
  let rate = dv[i * 2u].w;
  if (!(rate > 0.0)) { return; }
  let p = pos[i]; let v = vel[i].xyz;
  let seed = bitcast<u32>(p.w) * 9781u + u32(atomicLoad(&atoms[${A.FRAME}])) * 6271u;
  let n = u32(floor(rate * P.stepDt + rnd(seed)));
  if (n == 0u) { return; }
  let sp = length(v);
  var a = vec3f(0.0, 1.0, 0.0);
  if (sp > 1e-6) { a = v / sp; }
  let e1 = normalize(select(vec3f(-a.y, a.x, 0.0), vec3f(0.0, -a.z, a.y), abs(a.z) > 0.9));
  let e2 = cross(a, e1);
  for (var k = 0u; k < n; k++) {
    let slot = u32(atomicAdd(&atoms[${A.DIFF_COUNT}], 1));
    if (slot >= P.D) { atomicSub(&atoms[${A.DIFF_COUNT}], 1); return; }
    let s = seed + (k + 1u) * 8u;
    let r = 0.5 * P.spacing * sqrt(rnd(s));
    let th = rnd(s + 1u) * 6.2831853;
    let hv = rnd(s + 2u) * sp * P.stepDt;
    let off = cos(th) * r * e1 + sin(th) * r * e2;
    diffA[slot * 2u] = vec4f(p.xyz + off + hv * a, mix(P.lifeLo, P.lifeHi, rnd(s + 3u)));
    diffA[slot * 2u + 1u] = vec4f(v + 2.0 * off, 0.0);
  }
}
// classify by fluid neighbors, advect, age; survivors appended to diffB
@compute @workgroup_size(64) fn wwAdvect(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x;
  let m = min(u32(atomicLoad(&atoms[${A.DIFF_COUNT}])), P.D);
  if (k >= m) { return; }
  let pl = diffA[k * 2u];
  var x = pl.xyz; var v = diffA[k * 2u + 1u].xyz; var life = pl.w;
  if (life <= 0.0) { return; }
  let c = cellOf(x);
  let h2 = P.h * P.h;
  var cnt = 0.0; var ws = 0.0; var fv = vec3f(0.0);
  var bk: array<u32, 27>;
  var nb = 0u;
  for (var dz = -1; dz <= 1; dz++) { for (var dy = -1; dy <= 1; dy++) { for (var dx = -1; dx <= 1; dx++) {
    let b = hashCell(c + vec3i(dx, dy, dz)) & P.mask;
    var dup = false;
    for (var q = 0u; q < nb; q++) { if (bk[q] == b) { dup = true; break; } }
    if (!dup) { bk[nb] = b; nb++; }
  } } }
  for (var q = 0u; q < nb; q++) {
    let b = bk[q];
    let e = atomicLoad(&cells[b + 1u]);
    for (var j = atomicLoad(&cells[b]); j < e; j++) {
      let d = x - pos[j].xyz;
      let r2 = dot(d, d);
      if (r2 >= h2) { continue; }
      cnt += 1.0;
      let wk = kernW(sqrt(r2) * P.invH);
      ws += wk; fv += wk * vel[j].xyz;
    }
  }
  if (ws > 0.0) { fv /= ws; }
  let grav = vec3f(P.gx, P.gy, P.gz);
  let bdt = P.stepDt;
  var kind = 1.0;
  if (cnt < P.sprayMax) {
    kind = 0.0;
    v = (v + grav * bdt) * exp(-P.wwAir * bdt);
    life -= 0.25 * bdt;
  } else if (cnt > P.bubbleMin) {
    kind = 2.0;
    v += -P.wwBuoy * grav * bdt + P.wwDrag * (fv - v);
    life -= 0.25 * bdt;
  } else {
    v = fv;
    life -= bdt;
  }
  x += v * bdt;
  if (P.boundsOn > 0.5 && (x.x < P.bminX || x.y < P.bminY || x.z < P.bminZ || x.x > P.bmaxX || x.y > P.bmaxY || x.z > P.bmaxZ)) { life = 0.0; }
  if (kind < 0.5 && life > 0.0) {   // spray only: foam moves with the fluid
    for (var ci = 0u; ci < P.nCol; ci++) {
      if (cNear(ci, x, 0.0) && sdf(ci, x).d < 0.0) { life = 0.0; break; }
    }
  }
  if (life <= 0.0) { return; }
  let o = u32(atomicAdd(&atoms[${A.DIFF_NEXT}], 1));
  diffB[o * 2u] = vec4f(x, life);
  diffB[o * 2u + 1u] = vec4f(v, kind);
}
@compute @workgroup_size(1) fn wwCommit() {
  atomicStore(&atoms[${A.DIFF_COUNT}], atomicLoad(&atoms[${A.DIFF_NEXT}]));
  atomicStore(&atoms[${A.DIFF_NEXT}], 0);
  atomicAdd(&atoms[${A.FRAME}], 1);
}
@compute @workgroup_size(64) fn confine(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= count()) { return; }
  var v = vel[i].xyz + dv[i * 2u].xyz;
  let eps = P.vort * P.h * P.frames;
  let wi = dv[i * 2u + 1u];
  if (eps > 0.0 && wi.w > 1e-6) {
    let pi = pos[i]; let xi = pi.xyz;
    let base = i * (P.M + 1u);
    let nn = nbr[base];
    var e = vec3f(0.0);
    for (var t = 0u; t < nn; t++) {
      let j = nbr[base + 1u + t];
      let pj = pos[j];
      e += (dv[j * 2u + 1u].w - wi.w) / aux[j].x * gradW(xi - pj.xyz, pi.w, pj.w);
    }
    let el = length(e);
    if (el > 1e-9) { v += eps * cross(e / el, wi.xyz); }
  }
  vel[i] = vec4f(v, vel[i].w);
}

// x += dt·v, crossing-aware projection out of solids, bounds / NaN removal
@compute @workgroup_size(64) fn integrate(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= count()) { return; }
  var p = pos[i];
  if (p.w < 0.0) { return; }
  var v = vel[i].xyz;
  let s = length(v);
  if (s > P.maxSpeed) { v *= P.maxSpeed / s; }
  let x0 = p.xyz;
  var x = x0 + v * P.dt;
  let rmin = P.contactMin * P.spacing;
  for (var c = 0u; c < P.nCol; c++) {
    if (!cNear(c, x, P.h)) { continue; }
    if (u32(cf(c, 0u)) == 6u) {
      let q = cRot(c); let sz = cSize(c);
      for (var w = 0u; w < 5u; w++) {
        let l = rot(vec4f(-q.xyz, q.w), x - cPos(c));
        var wd = array<f32, 5>(sz.x - l.x, sz.x + l.x, sz.z - l.z, sz.z + l.z, sz.y + l.y);
        var wn = array<vec3f, 5>(vec3f(-1.0, 0.0, 0.0), vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, -1.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0));
        let pen = rmin - wd[w];
        if (pen <= 0.0) { continue; }
        let n = rot(q, wn[w]);
        x += pen * n;
        let vn = dot(v - cVel(c, x), n);
        if (vn < 0.0) { v -= vn * n; if (cDynamic(c)) { addImpulse(c, x, P.pmass * vn * n); } }
      }
      continue;
    }
    var hit = sdf(c, x);
    if (hit.d >= rmin) { continue; }
    var sp = x;
    if (hit.d < 0.0) {
      let d0 = sdf(c, x0).d;
      if (d0 > 0.0) { sp = x0 + (x - x0) * (d0 / (d0 - hit.d)); hit = sdf(c, sp); }
    }
    x = sp + (rmin - hit.d) * hit.n;
    let vn = dot(v - cVel(c, x), hit.n);
    if (vn < 0.0) { v -= vn * hit.n; if (cDynamic(c)) { addImpulse(c, x, P.pmass * vn * hit.n); } }
  }
  var id = p.w;
  let bad = !(all(abs(x) < vec3f(1e30)) && all(abs(v) < vec3f(1e30)));
  if (bad) { id = -1.0; atomicAdd(&atoms[${A.QUARANTINED}], 1); }
  else if (P.boundsOn > 0.5 && (x.x < P.bminX || x.y < P.bminY || x.z < P.bminZ || x.x > P.bmaxX || x.y > P.bmaxY || x.z > P.bmaxZ)) {
    id = -1.0; atomicAdd(&atoms[${A.LEAKED}], 1);
  }
  pos[i] = vec4f(x, id);
  vel[i] = vec4f(v, vel[i].w);
  atomicMax(&atoms[${A.VMAX}], bitcast<i32>(length(v)));
}
`;
