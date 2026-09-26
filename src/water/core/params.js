// water/core/params.js — solver parameters and the constants derived from them.
//
// Units are SI (meters, seconds, kg). The solver works with unit particle
// mass internally (density = Σ W, so rest density ≈ 1/spacing³); the real
// particle mass (waterDensity · spacing³) is only used for forces exchanged
// with rigid bodies.

export const DEFAULTS = {
  // --- solver ------------------------------------------------------------
  // 'dfsph': Divergence-Free SPH (Bender & Koschier 2015/2017) — implicit
  //   pressure on velocities with cached kernel gradients; ~0.1–0.5% density
  //   error, CFL-adaptive substeps.
  // 'pbf': Position Based Fluids (Macklin & Müller 2013).
  solver: 'dfsph',

  // --- resolution --------------------------------------------------------
  spacing: 0.1,          // rest distance between particles (m)
  kernelScale: null,     // support radius h = kernelScale · spacing (default: dfsph 2, pbf 1.8)
  maxParticles: 32768,
  maxNeighbors: 64,      // neighbor-list capacity per particle (overflow is counted)

  // --- dynamics ----------------------------------------------------------
  gravity: [0, -9.81, 0],
  // Jacobi PBF over-corrects high-frequency density errors (each particle
  // sums ~20 overlapping constraint corrections), which shows up as
  // step-to-step jitter in deep water. Measured on a 12-layer resting column
  // + dam break (spacing 0.1): 3 it/ω=1 → 0.24 m/s resting rms; 4 it/ω=0.8 →
  // 0.041 m/s, 3.2% compression, 10% peak compression on impact.
  iterations: 4,         // PBF density-constraint iterations per step
  sor: 0.8,              // Jacobi relaxation factor ω applied to each correction
  relaxation: 0.02,      // constraint softness (CFM), relative to the rest-state gradient
  // XSPH velocity smoothing per 1/60 s, 0..~0.1 (default: pbf 0.01, dfsph
  // 0.03 — DFSPH has no inherent damping; below ~0.03 the start-up repacking
  // of emitted lattices takes seconds to settle)
  viscosity: null,
  vorticity: 0.02,       // vorticity-confinement strength (restores swirl lost to damping)
  cohesion: 0.0,         // surface-tension-like attraction at the free surface, 0..~1
  maxSpeed: 25,          // m/s safety clamp

  // --- DFSPH -------------------------------------------------------------
  densityTolerance: 0.002,     // mean compression allowed after the pressure solve
  maxIterations: 16,           // pressure-solve iteration cap
  divergenceTolerance: 0.002,  // mean compression rate · dt allowed after the divergence solve
  maxDivergenceIterations: 4,
  minIterations: 2,
  // max travel per substep, in particle spacings. The implicit pressure solve
  // is stable well past 1 (dam break: CFL 2 kept the 5.7% peak error of CFL
  // 0.5 with no leaks, at 1 substep instead of 2.8); walls are crossing-checked.
  cfl: 1.5,
  minSubsteps: 1,
  maxSubsteps: 4,              // per step; beyond that speeds are clamped
  warmStart: 0.5,              // share of last step's pressure each solve starts from
  maxOverdensity: 0.05,        // existing over-density removed per substep (ρ0 fraction)
  contactMin: 0.3,             // closest approach to a solid, in spacings (safety projection)

  // --- boundaries --------------------------------------------------------
  // wall friction. PBF: tangential slip damping per step on contact, 0..1.
  // DFSPH: the walls' quadratic drag coefficient C_f (τ = ρ C_f |u|u):
  // ~0.003 smooth, ~0.01 gravel bed, ~0.03 boulders. Colliders can override.
  friction: null,        // default: pbf 0.1, dfsph 0.01
  // particle–collider contact distance, as a fraction of spacing. Static
  // walls use half the spacing (the fluid volume ends exactly at the wall);
  // dynamic bodies use less: a particle within the contact distance of a
  // body's EDGE still pushes on it, which makes bodies act ~1 contact radius
  // larger for buoyancy. Measured on a half-submerged 0.4 m box (spacing 0.1):
  // 0.5 → 1.21–1.30× Archimedes, 0.35 → 0.97–1.09×.
  collisionRadius: 0.5,
  dynamicCollisionRadius: 0.35,
  wallDensity: true,     // colliders contribute density (no gap/sticking at walls)
  bounds: null,          // {min:[x,y,z], max:[x,y,z]}: particles leaving it are removed

  // --- coupling ----------------------------------------------------------
  waterDensity: 1000,    // kg/m³ (particle mass = waterDensity · spacing³)
};

export const TWO_PI = Math.PI * 2;

// Poly6 kernel W(r) = K6 (h² − r²)³, spiky gradient ∇W = −KS (h − r)² r̂.
export function kernelConstants(h) {
  return {
    h,
    h2: h * h,
    K6: 315 / (64 * Math.PI * h ** 9),
    KS: 45 / (Math.PI * h ** 6),
  };
}

// ∫ over the half-space beyond distance d of the poly6 kernel (fraction of
// kernel mass inside a planar solid whose surface is d away). F(0) = 0.5.
// Closed form of 315/256 ∫_{d/h}^{1} (1 − u²)⁴ du.
function G(u) {
  const u2 = u * u, u3 = u2 * u, u5 = u3 * u2, u7 = u5 * u2, u9 = u7 * u2;
  return u - (4 / 3) * u3 + (6 / 5) * u5 - (4 / 7) * u7 + u9 / 9;
}
const G1 = G(1);
export function wallFraction(d, h) {
  if (d >= h) return 0;
  if (d <= -h) return 1;
  if (d < 0) return 1 - wallFraction(-d, h);
  return (315 / 256) * (G1 - G(d / h));
}
// dF/dd (negative: less wall density further from the wall).
export function wallFractionSlope(d, h) {
  if (d >= h || d <= -h) return 0;
  const u = d / h, a = 1 - u * u;
  return -(315 / 256) * a * a * a * a / h;
}

// Cubic spline kernel (Monaghan) with support radius h, q = r/h:
//   W = σ(6q³ − 6q² + 1) for q ≤ ½, 2σ(1 − q)³ for q ≤ 1, σ = 8/(πh³)
export function cubicW(r, h) {
  const q = r / h, s = 8 / (Math.PI * h * h * h);
  if (q >= 1) return 0;
  if (q <= 0.5) return s * (6 * q * q * q - 6 * q * q + 1);
  const a = 1 - q;
  return s * 2 * a * a * a;
}
export function cubicDW(r, h) { // dW/dr
  const q = r / h, s = 8 / (Math.PI * h * h * h) / h;
  if (q >= 1) return 0;
  if (q <= 0.5) return s * (18 * q * q - 12 * q);
  const a = 1 - q;
  return -s * 6 * a * a;
}

// Boundary volume function Ψ(d) for the cubic kernel: the density a planar
// solid at distance d supplies, as a fraction of ρ0 — the lattice planes the
// solid replaces, so a particle resting on the emission lattice against a
// wall sees exactly ρ0 (Bender et al. 2019 volume maps, discretised on the
// lattice instead of integrated). Tabulated over d ∈ [0, h] with its slope.
const BND_TABLE = 256;
function boundaryTable(s, h, rho0) {
  const R = Math.ceil(h / s) + 1;
  const plane = (z) => { // Σ W over one lattice plane at normal offset z
    let sum = 0;
    for (let a = -R; a <= R; a++) for (let b = -R; b <= R; b++) sum += cubicW(Math.sqrt(z * z + (a * a + b * b) * s * s), h);
    return sum;
  };
  const F = new Float32Array(BND_TABLE + 2), dF = new Float32Array(BND_TABLE + 2);
  const psi = (d) => {
    let sum = 0;
    for (let k = 0; ; k++) {
      const z = d + (k + 0.5) * s;
      if (z >= h) break;
      sum += plane(z);
    }
    return sum / rho0;
  };
  const step = h / BND_TABLE;
  for (let i = 0; i <= BND_TABLE + 1; i++) F[i] = psi(i * step);
  for (let i = 0; i <= BND_TABLE + 1; i++) {
    const a = F[Math.max(0, i - 1)], b = F[Math.min(BND_TABLE + 1, i + 1)];
    dF[i] = (b - a) / (step * (Math.min(BND_TABLE + 1, i + 1) - Math.max(0, i - 1)));
  }
  return { F, dF, inv: BND_TABLE / h, n: BND_TABLE };
}

/** Resolve user params into the full set of solver constants. */
export function deriveParams(user = {}) {
  const p = { ...DEFAULTS, ...user };
  if (user.gravity) p.gravity = [...user.gravity];
  if (p.kernelScale == null) p.kernelScale = p.solver === 'dfsph' ? 2 : 1.8;
  if (p.friction == null) p.friction = p.solver === 'dfsph' ? 0.01 : 0.1;
  if (p.viscosity == null) p.viscosity = p.solver === 'dfsph' ? 0.03 : 0.01;
  const s = p.spacing;
  const h = p.kernelScale * s;
  if (p.solver === 'dfsph') return deriveDFSPH(p, s, h);
  const k = kernelConstants(h);

  // Rest density and rest-state constraint gradient, measured on the cubic
  // lattice the emitters produce: the constraint is exactly satisfied there.
  let rho0 = 0, grad2 = 0;
  const R = Math.ceil(p.kernelScale) + 1;
  for (let x = -R; x <= R; x++) for (let y = -R; y <= R; y++) for (let z = -R; z <= R; z++) {
    const r = s * Math.hypot(x, y, z);
    if (r >= h) continue;
    const q = k.h2 - r * r;
    rho0 += k.K6 * q * q * q;
    if (r > 0) {
      const g = k.KS * (h - r) * (h - r);
      grad2 += g * g;
    }
  }
  grad2 /= rho0 * rho0;

  return {
    ...p,
    ...k,
    rho0,
    invRho0: 1 / rho0,
    W0: k.K6 * k.h2 * k.h2 * k.h2,                 // self contribution W(0)
    epsilon: p.relaxation * grad2,                 // CFM term in the λ denominator
    particleRadius: p.collisionRadius * s,         // contact distance, static/kinematic colliders
    dynamicRadius: p.dynamicCollisionRadius * s,   // contact distance, dynamic bodies
    particleMass: p.waterDensity * s * s * s,      // kg, for rigid-body coupling
  };
}

function deriveDFSPH(p, s, h) {
  // rest density and neighbor count on the emission lattice
  let rho0 = 0, neighbors = 0;
  const R = Math.ceil(p.kernelScale) + 1;
  for (let x = -R; x <= R; x++) for (let y = -R; y <= R; y++) for (let z = -R; z <= R; z++) {
    const r = s * Math.hypot(x, y, z);
    if (r >= h) continue;
    rho0 += cubicW(r, h);
    if (r > 0) neighbors++;
  }
  const bnd = boundaryTable(s, h, rho0);
  return {
    ...p,
    h, h2: h * h,
    kernelSigma: 8 / (Math.PI * h * h * h),
    W0: cubicW(0, h),
    rho0, invRho0: 1 / rho0,
    latticeNeighbors: neighbors,
    // particles with fewer neighbors are at the free surface: they get no
    // divergence correction (Bender & Koschier: ~20 of a full 3D neighborhood)
    minNeighbors: Math.round(0.6 * neighbors),
    bndF: bnd.F, bndDF: bnd.dF, bndInv: bnd.inv, bndN: bnd.n,
    particleRadius: p.collisionRadius * s,
    dynamicRadius: p.dynamicCollisionRadius * s,
    particleMass: p.waterDensity * s * s * s,
    epsilon: 0,
  };
}
