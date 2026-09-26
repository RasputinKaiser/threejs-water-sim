// water/core/colliders.js — rigid colliders as signed distance functions.
//
// Colliders live in one flat Float32Array (STRIDE floats each) so worker
// threads can share them. Every shape answers the same query: signed distance
// d from a point to the solid's surface (positive = in the fluid) and the unit
// normal pointing into the fluid. The solver uses that for collision
// projection, friction and the wall-density term.
//
// Record layout (see F.*):
//   type · position xyz · rotation quat xyzw · size abc · linear velocity xyz ·
//   angular velocity xyz · friction (<0: solver default) · flags · heightfield
//   index · force slot (index into the impulse accumulator, <0: none)

export const COLLIDER_STRIDE = 24;

export const SHAPE = {
  plane: 1,       // pos = point on plane, normal = rotation · +Y
  box: 2,         // oriented box, size = half extents
  sphere: 3,      // size.x = radius
  capsule: 4,     // local Y axis, size.x = radius, size.y = half height (segment)
  heightfield: 5, // terrain grid, see addHeightfield()
  container: 6,   // fluid INSIDE an oriented box (open top), size = inner half extents.
                  // One SDF = nearest wall only, so wall density is under-counted
                  // right at corners; use separate planes/boxes where that matters.
};

export const F = {
  type: 0, px: 1, py: 2, pz: 3, qx: 4, qy: 5, qz: 6, qw: 7,
  sa: 8, sb: 9, sc: 10, vx: 11, vy: 12, vz: 13, wx: 14, wy: 15, wz: 16,
  friction: 17, flags: 18, hf: 19, slot: 20,
};

export const FLAG_DYNAMIC = 1; // accumulate fluid impulses for rigid-body coupling

// Heightfield header layout inside its Float32Array: [minX, minZ, dx, dz, nx, nz, ...heights]
export const HF_HEADER = 6;

/** Pack a heightfield into the flat layout the solver reads. */
export function packHeightfield({ minX, minZ, dx, dz, nx, nz, heights }) {
  const out = new Float32Array(HF_HEADER + nx * nz);
  out[0] = minX; out[1] = minZ; out[2] = dx; out[3] = dz; out[4] = nx; out[5] = nz;
  out.set(heights.subarray ? heights.subarray(0, nx * nz) : heights, HF_HEADER);
  return out;
}

/**
 * Write one collider description into records[i*STRIDE ...].
 * desc: { type:'box'|..., position, rotation?, size|halfExtents|radius|halfHeight,
 *         velocity?, angularVelocity?, friction?, dynamic?, heightfield?, slot? }
 */
export function writeCollider(records, i, desc) {
  const o = i * COLLIDER_STRIDE;
  const type = typeof desc.type === 'number' ? desc.type : SHAPE[desc.type];
  if (!type) throw new Error(`unknown collider type '${desc.type}'`);
  const p = desc.position ?? [0, 0, 0];
  const q = desc.rotation ?? [0, 0, 0, 1];
  const v = desc.velocity ?? [0, 0, 0];
  const w = desc.angularVelocity ?? [0, 0, 0];
  let size = desc.size ?? desc.halfExtents;
  if (!size) size = [desc.radius ?? 0, desc.halfHeight ?? 0, 0];
  records[o + F.type] = type;
  records[o + F.px] = p[0]; records[o + F.py] = p[1]; records[o + F.pz] = p[2];
  records[o + F.qx] = q[0]; records[o + F.qy] = q[1]; records[o + F.qz] = q[2]; records[o + F.qw] = q[3];
  records[o + F.sa] = size[0] ?? 0; records[o + F.sb] = size[1] ?? 0; records[o + F.sc] = size[2] ?? 0;
  records[o + F.vx] = v[0]; records[o + F.vy] = v[1]; records[o + F.vz] = v[2];
  records[o + F.wx] = w[0]; records[o + F.wy] = w[1]; records[o + F.wz] = w[2];
  records[o + F.friction] = desc.friction ?? -1;
  records[o + F.flags] = desc.dynamic ? FLAG_DYNAMIC : 0;
  records[o + F.hf] = desc.heightfield ?? -1;
  records[o + F.slot] = desc.slot ?? (desc.dynamic ? i : -1);
}

// q · v (unit quaternion rotation), written into out[0..2]
function rotate(qx, qy, qz, qw, vx, vy, vz, out) {
  const tx = 2 * (qy * vz - qz * vy), ty = 2 * (qz * vx - qx * vz), tz = 2 * (qx * vy - qy * vx);
  out[0] = vx + qw * tx + (qy * tz - qz * ty);
  out[1] = vy + qw * ty + (qz * tx - qx * tz);
  out[2] = vz + qw * tz + (qx * ty - qy * tx);
}

const _l = new Float64Array(3);
const _n = new Float64Array(3);

/**
 * Signed distance from (x,y,z) to collider i. Writes the fluid-side unit
 * normal into out[0..2] and returns d (Infinity when the shape cannot touch
 * the point, e.g. outside a heightfield's footprint).
 */
export function colliderSDF(records, i, heightfields, x, y, z, out) {
  const o = i * COLLIDER_STRIDE;
  const type = records[o];
  const cx = records[o + 1], cy = records[o + 2], cz = records[o + 3];
  const qx = records[o + 4], qy = records[o + 5], qz = records[o + 6], qw = records[o + 7];
  if (type === 5) return heightfieldSDF(heightfields[records[o + F.hf]], x, y, z, out);
  // point in collider-local space (inverse rotation)
  rotate(-qx, -qy, -qz, qw, x - cx, y - cy, z - cz, _l);
  const lx = _l[0], ly = _l[1], lz = _l[2];
  let d;
  switch (type) {
    case 1: // plane: local +Y is the normal
      d = ly; _n[0] = 0; _n[1] = 1; _n[2] = 0;
      break;
    case 2: { // box
      d = boxSDF(lx, ly, lz, records[o + 8], records[o + 9], records[o + 10], _n);
      break;
    }
    case 3: { // sphere
      const r = Math.sqrt(lx * lx + ly * ly + lz * lz);
      d = r - records[o + 8];
      if (r > 1e-9) { _n[0] = lx / r; _n[1] = ly / r; _n[2] = lz / r; } else { _n[0] = 0; _n[1] = 1; _n[2] = 0; }
      break;
    }
    case 4: { // capsule along local Y
      const hh = records[o + 9];
      const sy = ly > hh ? hh : ly < -hh ? -hh : ly;
      const dx = lx, dy = ly - sy, dz = lz;
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      d = r - records[o + 8];
      if (r > 1e-9) { _n[0] = dx / r; _n[1] = dy / r; _n[2] = dz / r; } else { _n[0] = 1; _n[1] = 0; _n[2] = 0; }
      break;
    }
    case 6: { // container: fluid inside, open top — distance to nearest of 5 walls
      const ex = records[o + 8], ey = records[o + 9], ez = records[o + 10];
      const dxp = ex - lx, dxn = ex + lx, dzp = ez - lz, dzn = ez + lz, dyn = ey + ly;
      d = dxp; _n[0] = -1; _n[1] = 0; _n[2] = 0;
      if (dxn < d) { d = dxn; _n[0] = 1; _n[1] = 0; _n[2] = 0; }
      if (dzp < d) { d = dzp; _n[0] = 0; _n[1] = 0; _n[2] = -1; }
      if (dzn < d) { d = dzn; _n[0] = 0; _n[1] = 0; _n[2] = 1; }
      if (dyn < d) { d = dyn; _n[0] = 0; _n[1] = 1; _n[2] = 0; }
      break;
    }
    default:
      return Infinity;
  }
  rotate(qx, qy, qz, qw, _n[0], _n[1], _n[2], out);
  return d;
}

// Exact box SDF with outward gradient (local space).
function boxSDF(lx, ly, lz, ex, ey, ez, n) {
  const ax = Math.abs(lx) - ex, ay = Math.abs(ly) - ey, az = Math.abs(lz) - ez;
  const ox = ax > 0 ? ax : 0, oy = ay > 0 ? ay : 0, oz = az > 0 ? az : 0;
  const outside = Math.sqrt(ox * ox + oy * oy + oz * oz);
  if (outside > 0) {
    n[0] = (lx < 0 ? -ox : ox) / outside;
    n[1] = (ly < 0 ? -oy : oy) / outside;
    n[2] = (lz < 0 ? -oz : oz) / outside;
    return outside;
  }
  // inside: nearest face
  if (ax >= ay && ax >= az) { n[0] = lx < 0 ? -1 : 1; n[1] = 0; n[2] = 0; return ax; }
  if (ay >= az) { n[0] = 0; n[1] = ly < 0 ? -1 : 1; n[2] = 0; return ay; }
  n[0] = 0; n[1] = 0; n[2] = lz < 0 ? -1 : 1; return az;
}

// First-order SDF of a bilinear heightfield: (y − g) / |(−gx, 1, −gz)|.
function heightfieldSDF(hf, x, y, z, out) {
  if (!hf) return Infinity;
  const nx = hf[4], nz = hf[5];
  const fx = (x - hf[0]) / hf[2], fz = (z - hf[1]) / hf[3];
  if (!(fx >= 0 && fx < nx - 1 && fz >= 0 && fz < nz - 1)) return Infinity;
  const ix = fx | 0, iz = fz | 0;
  const tx = fx - ix, tz = fz - iz;
  const b = HF_HEADER + iz * nx + ix;
  const h00 = hf[b], h10 = hf[b + 1], h01 = hf[b + nx], h11 = hf[b + nx + 1];
  const g = h00 * (1 - tx) * (1 - tz) + h10 * tx * (1 - tz) + h01 * (1 - tx) * tz + h11 * tx * tz;
  const gx = ((h10 - h00) * (1 - tz) + (h11 - h01) * tz) / hf[2];
  const gz = ((h01 - h00) * (1 - tx) + (h11 - h10) * tx) / hf[3];
  const inv = 1 / Math.sqrt(gx * gx + 1 + gz * gz);
  out[0] = -gx * inv; out[1] = inv; out[2] = -gz * inv;
  return (y - g) * inv;
}

/** Velocity of collider i's surface at world point (x,y,z): v + ω × (p − c). */
export function colliderVelocity(records, i, x, y, z, out) {
  const o = i * COLLIDER_STRIDE;
  const rx = x - records[o + 1], ry = y - records[o + 2], rz = z - records[o + 3];
  const wx = records[o + 14], wy = records[o + 15], wz = records[o + 16];
  out[0] = records[o + 11] + wy * rz - wz * ry;
  out[1] = records[o + 12] + wz * rx - wx * rz;
  out[2] = records[o + 13] + wx * ry - wy * rx;
}
