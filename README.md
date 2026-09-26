# ThreeJS-Mods — Debug World

Box3D (Erin Catto's 3D physics, WASM via box3d.js) + three.js debug playground.
Dev server: **http://localhost:5184**

## Run
```
npm run dev        # vite on port 5184
```

## Debug tooling built in
- **HUD** (top-left): fps, sim ms, particle count, awake bodies, pool fill level + flatness σy (< 0.06 = settled flat)
- **Console overlay** (bottom-left): captured console.error/warn; click header to expand. `window.pushDbg('msg')` logs into it.
- **Screenshot**: press `P` → saves PNG to ~/Downloads
- **Pause / step**: `Space` pauses the fixed-step loop, `.` advances one 60Hz tick
- **lil-gui panels**: View toggles · Debug Draw (AABBs, contact points, velocity vectors, body axes) · 🚰 Water Sim (drop/splash/pour controls + fluid params) · 💧 Water Render (metaballs vs particles)
- **`window.__dbg`** in devtools console: `{ b3, world, sim, harness, probe }`

## Agent-driven runs (no GUI needed)
Append URL params — the page auto-captures canvas PNGs and POSTs them to the shot server:
```
node tools/shot-server.mjs                        # terminal 2 (saves to .shots/)
open http://localhost:5184/?autoshot=3&label=test # shot every 3s → .shots/test-*.png
```
Params: `autoshot=<sec>` · `label=<name>` · `mode=particles` (speed-colored points) ·
`pour=<particles/sec>` continuous pour · `autodrain=1` plughole draining.
Then inspect the PNGs (they're real frames — preserveDrawingBuffer is on).

## Tests & benchmarks
```
npm test           # solver regression tests + worker protocol test + effects smoke
npm run fuzz       # adversarial solver fuzz suite (NaN injection, teleports, param chaos, ...)
npm run bench      # per-phase solver profile at 8k–30k particles (sim.phaseMs)
```
CI (`.github/workflows/ci.yml`) runs tests, a shortened fuzz pass and the build.

## Realistic dynamic water
Clavet 2005 double-density relaxation solver in `src/water-pack/solver.js`.
Each step sorts particles by grid cell into an internal working copy, so the
neighbor walk reads contiguous memory; one pass builds the pair list and
densities and applies the viscosity impulses; relaxation runs off the stored
pairs. `pos`/`vel`/`nCount` keep their particle order for renderers.
~40 ms/step at 26.5k particles on one core. Large scenes (creek) run it in a
Web Worker (`workerSim: true`, needs the COOP/COEP headers the dev server sets).
Per-step phase timings are on `sim.phaseMs`.

Rendering: screen-space fluid (`src/water-pack/screen-fluid.js`: nearest-
surface sphere splats → bilateral-smoothed depth → view-space normals →
Fresnel/absorption composite, occluded by scene depth) or three.js
MarchingCubes metaballs. Fill probe measures level + flatness σy; plughole
drain with auto-drain toggle. Colliders mirror the Box3D static pool.

## Files
- `src/main.js` — Pool Lab: scene assembly, Box3D pool, spawn controls, HUD
- `src/scenes/*.html.js` — Bucket, Terrain, Big Pool and Creek labs
- `src/debug-harness.js` — camera/lights/HUD/console-capture/screenshot/fixed-timestep loop
- `src/box3d-debug.js` — Box3D→three sync + AABB/contact/velocity debug draws
- `src/water-pack/` — solver, worker sim (`sim-worker.mjs` + `async-sim.js`),
  surfaces (metaballs, screen-space), foam effects, probes, fuzz + benches
- `src/water-render.js` — Pool Lab metaball surface + particle rendering + fill probe
- `test/` — `node:test` suites
