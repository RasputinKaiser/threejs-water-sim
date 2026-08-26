// water-pack/box3d-adapter.js — bridges the pack's WaterSim into a Box3D world.
//
//   const water = createBox3DWater({ b3, world });
//   water.spawnBlock(x, y, z, nx, ny, nz);
//   // each frame:
//   water.step(dt);
//   water.sync();      // update three.js visuals
//
// Static bodies are auto-mirrored into pack colliders (axis-aligned boxes +
// planes). Dynamic Box3D bodies are NOT collided (documented limitation — the
// solver is one-way: water pushes nothing yet). Pass explicit `colliders` to
// override auto-mirroring.

import { WaterSim } from './solver.js';
import { createWaterSurface } from './surface.js';
import { createFillProbe } from './probe.js';
import * as THREE from 'three';

export function createBox3DWater({ b3, world, scene, bounds, params = {}, colliders = null, waterline = null, gui = null }) {
  const sim = new WaterSim(params);
  sim.bounds = bounds ? { min: bounds.min, max: [bounds.min[0] + bounds.size[0], bounds.min[1] + bounds.size[1], bounds.min[2] + bounds.size[2]] } : null;

  let cols = colliders;
  let mirrored = 0;
  if (!cols) {
    cols = [];
    // mirror static bodies: overlap the world AABB, read box hulls back
    const filter = b3.b3DefaultQueryFilter();
    const HUGE = [-1e9, -1e9, -1e9, 1e9, 1e9, 1e9];
    const _p = [0, 0, 0], _q = [0, 0, 0, 1];
    b3.b3World_OverlapAABB(world, HUGE, filter, (shapeId) => {
      const body = b3.b3Shape_GetBody(shapeId);
      if (b3.b3Body_GetType(body).value !== b3.b3BodyType.b3_staticBody.value) return true;
      const type = b3.b3Shape_GetType(shapeId).value;
      if (type !== b3.b3ShapeType.b3_hullShape.value) return true; // boxes are hulls in Box3D
      b3.b3Body_GetPosition(_p, body);
      b3.b3Body_GetRotation(_q, body);
      // only axis-aligned static bodies (|qw|≈1) mirror exactly; skip rotated ones
      if (Math.abs(Math.abs(_q[3]) - 1) > 1e-3) return true;
      const flat = b3.b3Shape_GetHullVertices(shapeId);
      let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (let i = 0; i < flat.length; i += 3) {
        minX = Math.min(minX, flat[i]); maxX = Math.max(maxX, flat[i]);
        minY = Math.min(minY, flat[i + 1]); maxY = Math.max(maxY, flat[i + 1]);
        minZ = Math.min(minZ, flat[i + 2]); maxZ = Math.max(maxZ, flat[i + 2]);
      }
      cols.push({
        type: 'box',
        c: [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2],
        e: [(maxX - minX) / 2, (maxY - minY) / 2, (maxZ - minZ) / 2],
      });
      mirrored++;
      return true;
    });
  }

  const surface = createWaterSurface(sim, scene, bounds, { waterline });
  if (gui) surface.addGui(gui);

  const api = {
    sim, surface, colliders: cols,
    mirrored,
    spawnBlock: (...a) => sim.spawnBlock(...a),
    spawn: (...a) => sim.spawn(...a),
    drain: (region) => sim.drain(region),
    reset: () => sim.reset(),
    step(dt) {
      sim.step(dt, cols);
      surface.update();
    },
    // attach a probe: water.probe(region) → metrics object updated each step
    probe(region) {
      const pr = createFillProbe(sim, region);
      api.lastProbe = pr.measure();
      sim.waterLevel = api.lastProbe.meanY;
      return pr;
    },
  };
  return api;
}
