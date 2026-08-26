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

## First project: realistic dynamic water
PBF solver (Clavet 2005 double-density relaxation) in `src/pbf-water.js` — flat
typed-array grid, ~9ms/step at 6k particles. Water spawns above the pool, pours
in, splashes (white foam on low-density/high-speed particles), and settles flat
(fill probe measures level + flatness σy). Plughole drain with auto-drain toggle.
Colliders mirror the Box3D static pool. Surface: three.js MarchingCubes metaballs
with RoomEnvironment reflections (transmission off — env-map is what sells it).

## Files
- `src/main.js` — scene assembly, pool build, spawn controls, HUD
- `src/debug-harness.js` — camera/lights/HUD/console-capture/screenshot/fixed-timestep loop
- `src/box3d-debug.js` — Box3D→three sync + AABB/contact/velocity debug draws
- `src/pbf-water.js` — PBF fluid solver
- `src/water-render.js` — metaball surface + particle rendering + fill probe
