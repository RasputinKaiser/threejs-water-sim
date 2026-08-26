// water-pack/probe.js — fill measurement + flatness metrics for a region.
// The pack's truth-teller: level, flatness σy, fill %, and per-region counts.
// Used by the HUD, the waterline marker, and the agent metrics pipeline.

export function createFillProbe(sim, region) {
  // region: {min:[x,y,z], max:[x,y,z]} — container interior in world space
  return {
    region,
    measure() {
      let sumY = 0, sumY2 = 0, nIn = 0, top = -Infinity;
      const p = sim.pos;
      for (let i = 0; i < sim.count; i++) {
        const x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
        if (x > region.min[0] && x < region.max[0] && z > region.min[2] && z < region.max[2] && y < region.max[1]) {
          sumY += y; sumY2 += y * y; nIn++;
          if (y > top) top = y;
        }
      }
      if (!nIn) return { count: 0, meanY: NaN, stdY: NaN, topY: NaN, fillPct: 0 };
      const mean = sumY / nIn;
      const varr = sumY2 / nIn - mean * mean;
      const depth = region.max[1] - region.min[1];
      const fillPct = Math.max(0, Math.min(100, ((mean - region.min[1]) / depth) * 100));
      return { count: nIn, meanY: mean, stdY: Math.sqrt(Math.max(varr, 0)), topY: top, fillPct };
    },
  };
}

// Multi-region census: how many particles sit in each named region (crevice A/B,
// bucket vs ground, etc.) — shows where water went, not just where it should be.
export function createRegionCensus(sim, regions) {
  return {
    measure() {
      const out = {};
      for (const name in regions) out[name] = 0;
      out._other = 0;
      const p = sim.pos;
      for (let i = 0; i < sim.count; i++) {
        const x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
        let hit = false;
        for (const name in regions) {
          const r = regions[name];
          if (x >= r.min[0] && x <= r.max[0] && y >= r.min[1] && y <= r.max[1] && z >= r.min[2] && z <= r.max[2]) {
            out[name]++; hit = true; break;
          }
        }
        if (!hit) out._other++;
      }
      return out;
    },
  };
}
