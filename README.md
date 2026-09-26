# threejs-water-sim

Drop-in, physically based particle water for three.js games, with optional
two-way coupling to [Box3D](https://github.com/erincatto/box3d) (WASM via
box3d.js). The default solver is Divergence-Free SPH (Bender & Koschier) on a
spatial hash grid, with solid boundaries as volume maps and whitewater
(spray, foam, bubbles); Position Based Fluids is available as a cheaper
alternative. It runs on worker threads when the page is cross-origin isolated
and on the main thread otherwise. Rendering is a screen-space fluid pass with
depth-based absorption, refraction, environment reflections and whitewater.

```
npm install
npm run dev        # labs on http://localhost:5184
```

## Quick start

```js
import { createWater } from './src/water/index.js';

const water = await createWater({ renderer, scene, b3, world, quality: 'medium' });
water.addCollider({ type: 'container', position: [0, 1, 0], size: [2, 1, 2] }); // open-top tank
water.fillBox([-2, 0, -2], [2, 0.8, 2]);                                         // fill it
const tap = water.addSource({ position: [0, 3, 0], direction: [0, -1, 0], radius: 0.15, speed: 3 });

function frame(dt) {
  const simDt = water.update(dt);                      // fluid time advanced (fixed steps)
  if (simDt > 0) b3.b3World_Step(world, simDt, 4);     // keep Box3D in lockstep with it
  water.render(camera);                                // replaces renderer.render(scene, camera)
}
```

`b3`/`world` are optional. With them, every Box3D body near the water becomes
a collider each frame: static bodies are walls, kinematic bodies push water,
dynamic bodies also receive the fluid's reaction (buoyancy, drag, splashes).
A body of density ρ floats with about ρ/1000 of its volume submerged.

### API

| | |
|---|---|
| `createWater(opts)` | `renderer, scene, b3?, world?, quality: 'low'\|'medium'\|'high', params, colliders, threads: 'auto'\|n\|0, render: 'screen'\|'points'\|false, look` |
| `water.update(dt)` | advance by real time `dt`; returns the fluid seconds simulated (0 while a worker batch is in flight) |
| `water.render(camera, target?)` | draw the scene with water |
| `water.fillBox(min, max, {velocity})` | fill a box at rest density |
| `water.spawn(particles)` | `Float32Array [x,y,z,vx,vy,vz]*` or `[[x,y,z,vx?,vy?,vz?], …]` |
| `water.removeInBox(min, max)`, `water.reset()` | remove water |
| `water.addCollider(desc)` / `removeCollider(desc)` | `plane`, `box`, `sphere`, `capsule`, `container`; `position`, `rotation` (quat), `size` (half extents) / `radius` / `halfHeight`, `friction` |
| `water.addHeightfield({minX, minZ, dx, dz, nx, nz, heights})` | terrain |
| `water.addSource({position, direction, radius, speed})` | nozzle; flow = π·r²·speed m³/s. Handle has `enabled`, `speed`, `radius`, `position`, `direction`, `remove()` |
| `water.addDrain({min, max})` | removes water entering the box every frame |
| `water.surfaceHeight(x, z)` | water surface height near (x, z), `-Infinity` if dry |
| `water.setParams(patch)` | `viscosity, vorticity, friction, densityTolerance, gravity, maxSpeed, bounds, …` |
| `water.setLook(patch)` | `absorption, scatterColor, scatter, refraction, roughness, f0, envIntensity, foamSize, foamOpacity, …` |
| `water.stats` | `stepMs, threads, avgDensityError, maxDensityError, pressureIterations, substeps, whitewater, kineticEnergy, leaked, drained, colliders` |

Quality presets (`src/water/index.js`): `low` 15 cm spacing, 0.5% density
tolerance; `medium` 10 cm, 0.2%; `high` 8 cm, 0.1%. Override anything through
`params` (`solver: 'dfsph'|'pbf'`, `spacing`, `maxParticles`, `friction`,
`maxDiffuse`, …; every parameter is documented in `src/water/core/params.js`).
`friction` is the walls' quadratic drag coefficient C_f (τ = ρ·C_f·|u|·u):
~0.003 smooth, ~0.01 gravel (default), ~0.03 boulders.

**Threads** need `SharedArrayBuffer`, i.e. the page must be served with
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` (the Vite config here sets both).
Without them `createWater` silently falls back to the main thread.

## Physics

- **Incompressibility — DFSPH.** Each substep solves for pressure twice on
  the velocities: a constant-density solve (mean error ≤ `densityTolerance`,
  0.2% by default) and a divergence-free solve (no jitter, no bounce). Kernel
  gradients are cached once per substep, so every Jacobi iteration only
  streams the cache; both solves are warm-started from the previous step's
  pressure, which travels with the particles through the sort. Substeps adapt
  to the fastest particle (CFL 1.5 — the implicit solve is stable well past 1).
- **Boundaries as volume maps.** A solid contributes ρ0·Ψ(d) of density at
  distance d, with Ψ tabulated from the lattice planes the solid replaces, so
  water resting against a wall reads exactly ρ0; containers are the union of
  their wall half-spaces. The boundary pressure force is the constraint force
  (Jᵀk), so it does no work on density-preserving motion. Particles are kept
  out of solids by a crossing-aware projection; spawns inside solids are dropped.
- **Wall shear**: quadratic bed drag (`friction` = C_f), applied implicitly.
- **Viscosity** (XSPH, per 1/60 s) and **vorticity confinement**.
- **Whitewater** (Ihmsen et al. 2012): every fluid particle has a generation
  potential from trapped air (neighbours converging), wave crests (surface
  particles moving outward) and kinetic energy; diffuse particles are spray
  (ballistic), foam (rides the flow, ages out) or bubbles (buoyant, dragged)
  by their fluid neighbourhood.
- **Hash grid**: particles are counting-sorted into a Teschner-hashed table
  every substep (cache-coherent neighbour walks, unbounded world, memory
  proportional to particle count). Neighbour lists are exact — tested against
  brute force including hash collisions.
- **Box3D coupling**: every velocity change the boundary pressure, the wall
  drag or the projection gives a particle is applied back to the body it
  touched — buoyancy, drag and splashes all come from that one momentum
  exchange — with a per-frame impulse bound so light bodies are not launched.
- **Determinism**: the threaded solver gives bit-identical results to the
  single-threaded one.

Measured on the 4-core VM this was developed on:

| | DFSPH | PBF |
|---|---|---|
| buoyancy on a fixed submerged sphere / box (spacing 0.1) | 1.05× / 1.02× ρgV | 1.26× / 1.10× |
| same at spacing 0.05 | 0.96× / 0.93× | 0.77× / 0.71× |
| mean density error, resting column / creek | 0.2% / 0.07% | ~3% |
| creek, 6.3k particles + 1.3k diffuse, 1 thread | 29 ms/step (4.7 µs/particle) | — |
| dam break, 8k particles, 1 / 3 threads (`npm run bench`) | 65 / 28 ms/step | 41 / 19 ms/step |

Cost scales linearly with particle count; a desktop CPU with more cores is
proportionally faster. PBF (`params.solver: 'pbf'`) is cheaper in violent
scenes because it runs a fixed number of iterations and accepts 3–10%
compression.

## Labs

| page | what it tests |
|---|---|
| `index.html` — Pool Lab | Box3D pool, balls of 250–2000 kg/m³, crates, pour, plughole |
| `bucket.html` | 1.2 m bucket at 5 cm spacing, spout fill, settling |
| `pool.html` — Big Pool | 25 × 12.5 m pool, ~25k particles, waves |
| `terrain.html` | heightfield valley, flow downhill into a basin |
| `creek.html` — Creek | the main proving ground: a 40 m meander on a 2% grade (terrain and boulders in both Box3D and the solver), floating logs carried by the current, submerged inlet + outlet, whitewater, gauges for depth / speed / flow |
| `water-lab.html` | the pack used without the debug harness, as a game would |

URL params: `quality=low|medium|high`, `render=screen|points`, `pour`.
The labs share `src/scenes/water-harness.js` (fill probe, HUD, GUI).

### Debug tooling
- **HUD** (top-left): fps, particles, solver mode/threads, step time, density error, fill level and surface flatness
- **Console overlay** (bottom-left): captured console.error/warn; `window.pushDbg('msg')` logs into it
- **Screenshot**: `P` · **Pause / step**: `Space` / `.`
- **`window.__dbg`** in the devtools console

Agent-driven runs: `node tools/shot-server.mjs`, then open a lab with
`?autoshot=<sec>&label=<name>` (PNGs in `.shots/`) and/or `?metrics=<label>`
(JSON snapshots in `.metrics/`).

## Tests & tools
```
npm test           # node:test suites: DFSPH + PBF cores, threads, Box3D buoyancy and drag, whitewater, createWater
npm run fuzz       # adversarial fuzz (NaN injection, teleports, overlap, param chaos, …)
npm run bench      # ms/step + per-phase split, single-threaded vs threaded (--solver dfsph|pbf)
node tools/creek-bench.mjs   # the Creek headless: inflow/outflow, gauges, density error, cost
```
CI (`.github/workflows/ci.yml`) runs the tests, a shortened fuzz pass and the build.

## Files
- `src/water/` — the pack: `index.js` (createWater), `sim.js` (threaded/inline runner),
  `box3d.js` (coupling), `render/screen-space.js`, and `core/`: `fluid-core.js`
  (buffers, hash grid, neighbours, colliders), `dfsph.js`, `solver.js` (PBF),
  `whitewater.js`, `colliders.js`, `params.js`, `worker.js`/`threads.js`
- `src/scenes/creek-world.js` — the Creek's terrain, channel, boulders, inlet and outlet (shared by the lab and the benchmark)
- `src/main.js`, `src/scenes/` — the labs
- `src/debug-harness.js`, `src/box3d-debug.js` — lab harness and Box3D debug draw
- `tools/` — fuzz, bench, screenshot server · `test/` — `node:test` suites
- `research/` — design notes from earlier iterations (they describe the previous solver)
