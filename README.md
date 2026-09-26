# threejs-water-sim

Drop-in, physically based particle water for three.js games, with optional
two-way coupling to [Box3D](https://github.com/erincatto/box3d) (WASM via
box3d.js). The solver is Position Based Fluids (Macklin & Müller 2013) on a
spatial hash grid; it runs on worker threads when the page is cross-origin
isolated and on the main thread otherwise. Rendering is a screen-space fluid
pass with depth-based absorption, refraction and environment reflections.

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
| `water.setParams(patch)` | `viscosity, vorticity, cohesion, friction, iterations, gravity, maxSpeed, bounds` |
| `water.setLook(patch)` | `absorption, scatterColor, scatter, refraction, roughness, f0, envIntensity, …` |
| `water.stats` | `stepMs, threads, maxDensityError, kineticEnergy, leaked, drained, colliders` |

Quality presets (`src/water/index.js`): `low` 15 cm spacing, 3 iterations;
`medium` 10 cm, 4 iterations; `high` 8 cm, 6 iterations. Override anything
through `params` (`spacing`, `maxParticles`, `iterations`, …; see
`src/water/core/params.js`).

**Threads** need `SharedArrayBuffer`, i.e. the page must be served with
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` (the Vite config here sets both).
Without them `createWater` silently falls back to the main thread.

## Physics

- **Incompressibility**: unilateral PBF density constraint, Jacobi iterations
  with relaxation ω = 0.8, rest density taken from the emission lattice so
  freshly spawned water is at rest. Solid walls contribute an analytic
  density term, so water at a wall is neither sucked in nor pushed away.
- **Hash grid**: particles are counting-sorted into a Teschner-hashed table
  every step (cache-coherent neighbour walks, unbounded world, memory
  proportional to particle count). Neighbour lists are exact — tested against
  brute force including hash collisions.
- **Viscosity** (XSPH), **vorticity confinement**, optional **cohesion**.
- **Colliders** are signed distance fields with continuous exit along the
  crossing path, so fast particles do not tunnel into or get stuck in solids.
- **Box3D coupling**: per-body impulses from the solver's contact reactions,
  mass-weighted contact response, and a per-frame impulse bound (Archimedes
  + inelastic exchange) so light bodies are not launched.
- **Determinism**: the threaded solver gives bit-identical results to the
  single-threaded one.

Measured on the 4-core CI-class VM this was developed on (`npm run bench`):
8,000 particles ≈ 38 ms/step single-threaded, ≈ 18 ms/step on 3 threads.
Cost scales linearly with particle count; a desktop CPU with more cores is
proportionally faster.

## Labs

| page | what it tests |
|---|---|
| `index.html` — Pool Lab | Box3D pool, balls of 250–2000 kg/m³, crates, pour, plughole |
| `bucket.html` | 1.2 m bucket at 5 cm spacing, spout fill, settling |
| `pool.html` — Big Pool | 25 × 12.5 m pool, ~25k particles, waves |
| `terrain.html` | heightfield valley, flow downhill into a basin |
| `creek.html` | 40 m meandering channel, inflow nozzle + outflow drain |
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
npm test           # node:test suites: solver core, threads, Box3D buoyancy, createWater
npm run fuzz       # adversarial fuzz (NaN injection, teleports, overlap, param chaos, …)
npm run bench      # ms/step + per-phase split, single-threaded vs threaded
```
CI (`.github/workflows/ci.yml`) runs the tests, a shortened fuzz pass and the build.

## Files
- `src/water/` — the pack: `index.js` (createWater), `sim.js` (threaded/inline runner),
  `box3d.js` (coupling), `core/` (solver, colliders, params, worker), `render/screen-space.js`
- `src/main.js`, `src/scenes/` — the labs
- `src/debug-harness.js`, `src/box3d-debug.js` — lab harness and Box3D debug draw
- `tools/` — fuzz, bench, screenshot server · `test/` — `node:test` suites
- `research/` — design notes from earlier iterations (they describe the previous solver)
