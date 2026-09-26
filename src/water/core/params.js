// water/core/params.js — solver parameters and the constants derived from them.
//
// Units are SI (meters, seconds, kg). The solver works with unit particle
// mass internally (density = Σ W, so rest density ≈ 1/spacing³); the real
// particle mass (waterDensity · spacing³) is only used for forces exchanged
// with rigid bodies.

export const DEFAULTS = {
  // --- resolution --------------------------------------------------------
  spacing: 0.1,          // rest distance between particles (m)
  kernelScale: 1.8,      // smoothing radius h = kernelScale · spacing (~20 neighbors)
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
  viscosity: 0.01,       // XSPH velocity smoothing, 0..~0.1
  vorticity: 0.02,       // vorticity-confinement strength (restores swirl lost to damping)
  cohesion: 0.0,         // surface-tension-like attraction at the free surface, 0..~1
  maxSpeed: 25,          // m/s safety clamp

  // --- boundaries --------------------------------------------------------
  friction: 0.1,         // tangential slip damping on collider contact, 0..1
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

/** Resolve user params into the full set of solver constants. */
export function deriveParams(user = {}) {
  const p = { ...DEFAULTS, ...user };
  if (user.gravity) p.gravity = [...user.gravity];
  const s = p.spacing;
  const h = p.kernelScale * s;
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
    particleRadius: 0.5 * s,                       // collision radius
    particleMass: p.waterDensity * s * s * s,      // kg, for rigid-body coupling
  };
}
