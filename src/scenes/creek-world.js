// scenes/creek-world.js — the Creek: terrain, channel geometry, boulders,
// inflow and outflow, shared by the lab page (creek.html.js) and the headless
// benchmark (tools/creek-bench.mjs). Pure math, no three.js.
//
// A 40 × 24 m heightfield with a meandering parabolic channel (3.2 m wide,
// 0.9 m deep) on a 2% grade. Water enters through a submerged inlet at the
// upstream end, runs the meander past boulders and leaves through an outflow
// region at the downstream end.

export const NX = 161, NZ = 97, SIZE_X = 40, SIZE_Z = 24;
export const DX = SIZE_X / (NX - 1), DZ = SIZE_Z / (NZ - 1); // 0.25 m

// meander centerline: two sinusoids → 2-3 graceful bends across 40 m
const A1 = 2.5, F1 = 0.28;
const A2 = 1.2, F2 = 0.71, PH2 = 1.3;
export const channelZ = (x) => A1 * Math.sin(x * F1) + A2 * Math.sin(x * F2 + PH2);
export const channelDz = (x) => A1 * F1 * Math.cos(x * F1) + A2 * F2 * Math.cos(x * F2 + PH2);

export const HALF_W = 1.6;     // channel half-width (m)
export const DEPTH = 0.9;      // centerline depth below the bank base (m)
export const SLOPE = 0.02;     // downstream grade (0.8 m drop over 40 m)
const SHOULDER_W = 1.2;        // bank shoulder band beyond HALF_W

function hills(x, z) {
  return 0.18 * Math.sin(x * 0.21) * Math.cos(z * 0.27) + 0.09 * Math.sin(x * 0.53 + z * 0.41);
}

/* bed bumps: [x, lateral offset from centerline, height, radius] — gaussian */
export const BED_BUMPS = [
  [-12.0, -0.5, 0.18, 0.60],
  [-3.0, -0.9, 0.20, 0.70],
  [6.0, -0.6, 0.22, 0.90],
  [15.0, -0.4, 0.18, 0.65],
];
function bumpHeight(x, z) {
  let s = 0;
  for (const [bx, bz, bh, br] of BED_BUMPS) {
    const d = (x - bx) ** 2 + (z - channelZ(bx) - bz) ** 2;
    s += bh * Math.exp(-d / (2 * br * br));
  }
  return s;
}

/** Bed centerline elevation at x (bottom of the channel). */
export const bedY = (x) => -SLOPE * x - DEPTH;

/** Terrain elevation without the map rim. */
export function terrainH(x, z) {
  const t = Math.abs(z - channelZ(x)) / HALF_W;
  let h;
  if (t < 1) h = -SLOPE * x - DEPTH * (1 - t * t); // parabolic bed
  else {
    const tSh = SHOULDER_W / HALF_W;
    if (t < 1 + tSh) {
      const s = (t - 1) / tSh;
      h = -SLOPE * x + s * hills(x, z) + 0.15 * Math.sin(Math.PI * s) ** 2;
    } else h = hills(x, z) - SLOPE * x;
  }
  return h + bumpHeight(x, z);
}

/** Ground as the collider sees it: terrain + a steep rim so nothing leaves the map. */
export function groundH(x, z) {
  let h = terrainH(x, z);
  const edge = Math.max(Math.abs(x) / (SIZE_X / 2), Math.abs(z) / (SIZE_Z / 2));
  if (edge > 0.82) h += ((edge - 0.82) / 0.18) ** 2 * 6;
  return h;
}

export function buildHeights() {
  const heights = new Float32Array(NX * NZ);
  for (let iz = 0; iz < NZ; iz++) {
    for (let ix = 0; ix < NX; ix++) heights[iz * NX + ix] = groundH(-SIZE_X / 2 + ix * DX, -SIZE_Z / 2 + iz * DZ);
  }
  return heights;
}

export const heightfieldDesc = (heights) => ({
  minX: -SIZE_X / 2, minZ: -SIZE_Z / 2, dx: DX, dz: DZ, nx: NX, nz: NZ, heights,
});

/* boulders standing in the stream: [x, lateral offset, radius, sink] — spheres
 * whose centre sits `sink` of a radius below the bed so they read as embedded */
export const BOULDERS = [
  [-14.5, 0.5, 0.45, 0.35],
  [-8.0, 0.9, 0.55, 0.4],
  [-7.2, -0.7, 0.35, 0.3],
  [1.0, 0.3, 0.45, 0.35],
  [9.0, 0.7, 0.5, 0.4],
  [11.5, -0.5, 0.4, 0.3],
];
export function boulderSpheres() {
  return BOULDERS.map(([x, off, r, sink]) => {
    const z = channelZ(x) + off;
    return { center: [x, terrainH(x, z) + r * (1 - sink) * 0.9, z], radius: r };
  });
}

/* inflow: a submerged inlet low in the upstream channel, aimed down-creek */
export const INLET_X = -17.2;
export function inlet({ radius = 0.45, speed = 1.2 } = {}) {
  const z = channelZ(INLET_X);
  const t = [1, 0, channelDz(INLET_X)];
  const l = Math.hypot(t[0], t[2]);
  return {
    position: [INLET_X, bedY(INLET_X) + radius + 0.12, z],
    direction: [t[0] / l, -0.08, t[2] / l],
    radius, speed,
  };
}

/* outflow: everything past x = 16.5 leaves the world */
export const OUTLET = { min: [16.5, -4, -12], max: [21, 5, 12] };

/* gauge stations along the channel centreline (depth / speed measurement) */
export const STATIONS = [-12, -4, 4, 12];
