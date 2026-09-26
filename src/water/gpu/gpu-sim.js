// water/gpu/gpu-sim.js — the DFSPH solver on WebGPU compute, behind the same
// simulation interface as sim.js (inline / threaded).
//
// Every update() posts one batch of fixed steps as a single command buffer;
// the batch ends by copying positions (+ previous-frame positions for
// interpolation, velocities, counters, body impulses) into a mappable
// staging buffer. Like the threaded runner, the results are latched when the
// map resolves and update() returns 0 while a batch is in flight.
//
// The CPU never needs the exact particle count: spawns are staged and
// appended by the GPU at its live count, removals only flag particles, and
// the sort compacts. Kernels are dispatched over the capacity and exit past
// the live count. Pressure solves run fixed iteration counts (no CPU round
// trip per iteration); substeps follow the CFL condition on the last
// read-back maximum speed.

import { writeCollider, packHeightfield, COLLIDER_STRIDE } from '../core/colliders.js';
import { makeBase, latticeBox, toSpawnArray, interpolateInto } from '../sim.js';
import { WGSL, A, IMPULSE_SCALE, CONTACT_SCALE, K_BND } from './wgsl.js';

const BINDINGS = {
  appendSpawn: [0, 1, 2, 3, 13, 14, 16],
  commitSpawn: [1, 14, 16],
  markFrame: [1, 3, 14],
  drain: [0, 1, 14],
  hash: [0, 1, 7, 8, 14],
  scanBlocks: [0, 8, 9],
  scanTop: [0, 9],
  scanAdd: [0, 8, 9, 14],
  scatter: [0, 1, 2, 3, 4, 5, 6, 7, 8],
  neighbors: [0, 1, 2, 8, 10, 13, 14],
  density: [0, 1, 2, 10, 11, 12, 13, 14],
  warmDensity: [0, 2, 11, 14],
  warmDivergence: [0, 3, 11, 14],
  velocity: [0, 1, 2, 10, 11, 12, 13, 14],
  densityGate: [0, 1, 2, 3, 10, 11, 12, 14],
  densityResidual: [0, 1, 2, 3, 10, 11, 12, 14],
  divergenceResidual: [0, 1, 2, 3, 10, 11, 12, 14],
  forces: [0, 1, 2, 10, 11, 12, 13, 14, 15],
  confine: [0, 1, 2, 10, 11, 14, 15],
  integrate: [0, 1, 2, 13, 14],
};

const PARAM_FLOATS = 40 + 16 * 4;
const MAX_DRAINS = 8;
const HF_CAPACITY = 1 << 20;   // floats reserved for heightfields
const MAX_HEIGHTFIELDS = 16;

function nextPow2(n) { let p = 1; while (p < n) p <<= 1; return p; }

/**
 * @param {object} dp    derived params (solver 'dfsph')
 * @param {object} opts  sim.js options (+ gpuIterations, gpuDivergenceIterations, readVelocities)
 * @returns {Promise<object|null>} the simulation, or null without a usable adapter
 */
export async function createGPUSimulation(dp, opts = {}) {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  const lim = adapter.limits;
  if (lim.maxStorageBuffersPerShaderStage < 8) return null;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBuffersPerShaderStage: Math.min(lim.maxStorageBuffersPerShaderStage, 10),
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      maxBufferSize: lim.maxBufferSize,
    },
  });
  const base = makeBase(dp, opts);
  const N = dp.maxParticles, M = Math.min(dp.maxNeighbors, 48), C = base.maxColliders;
  const T = nextPow2(Math.max(64, 2 * N));
  const nScan = T + 2;                           // buckets 0..T (T: dead) + total
  const nBlocks = Math.ceil(nScan / 512);
  if (nBlocks > 1024) throw new Error('water gpu: maxParticles too large for the scan');
  // pressure iterations adapt to the density tolerance one batch late (no
  // CPU round trip inside a batch): above tolerance → more, well below → fewer
  let pressureIterations = opts.gpuIterations ?? 3;
  const adaptive = opts.gpuIterations == null;
  const divergenceIterations = opts.gpuDivergenceIterations ?? 1;
  const readVelocities = opts.readVelocities ?? true;

  /* ------------------------------ buffers ------------------------------ */
  const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
  const mk = (size, usage) => device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage });
  const tabN = dp.bndN + 2;
  const hfBase = C * COLLIDER_STRIDE;
  const tabBase = hfBase + MAX_HEIGHTFIELDS + HF_CAPACITY;
  const worldFloats = tabBase + 2 * tabN;
  const nAtoms = A.IMPULSES + C * 10;
  const buf = {
    params: mk(PARAM_FLOATS * 4, GPUBufferUsage.UNIFORM | CD),
    pos: mk(N * 16, S | CD | CS), vel: mk(N * 16, S | CD | CS), prv: mk(N * 16, S | CD | CS),
    posB: mk(N * 16, S | CS), velB: mk(N * 16, S | CS), prvB: mk(N * 16, S | CS),
    keys: mk(N * 8, S), cells: mk(nScan * 4, S | CD), blocks: mk(1024 * 4, S),
    nbr: mk(N * (M + 1) * 4, S), aux: mk(N * 16, S), bnd: mk(N * K_BND * 32, S),
    world: mk(worldFloats * 4, S | CD), atoms: mk(nAtoms * 4, S | CD | CS),
    dv: mk(N * 32, S), spawn: mk((1 + 2 * N) * 16, S | CD),
  };
  const byBinding = [buf.params, buf.pos, buf.vel, buf.prv, buf.posB, buf.velB, buf.prvB, buf.keys,
    buf.cells, buf.blocks, buf.nbr, buf.aux, buf.bnd, buf.world, buf.atoms, buf.dv, buf.spawn];

  // world: colliders | heightfield offsets | heightfields | Ψ table | Ψ' table
  const world = new Float32Array(worldFloats);
  world.set(dp.bndF.subarray(0, tabN), tabBase);
  world.set(dp.bndDF.subarray(0, tabN), tabBase + tabN);
  let hfUsed = 0, hfCount = 0;
  device.queue.writeBuffer(buf.world, 0, world);

  /* ----------------------------- pipelines ----------------------------- */
  const module = device.createShaderModule({ code: WGSL });
  const info = await module.getCompilationInfo?.();
  const errors = info?.messages?.filter((m) => m.type === 'error') ?? [];
  if (errors.length) throw new Error(`water gpu: WGSL errors:\n${errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n')}`);
  const pipes = {};
  for (const [entry, bindings] of Object.entries(BINDINGS)) {
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: entry } });
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map((b) => ({ binding: b, resource: { buffer: byBinding[b] } })),
    });
    pipes[entry] = { pipeline, group };
  }

  /* ------------------------------ params ------------------------------- */
  const pf = new Float32Array(PARAM_FLOATS), pu = new Uint32Array(pf.buffer);
  let colliderCount = 0;
  const drains = [];      // persistent {min,max} boxes set via removeInBox (one batch)
  function writeParams(dt, frames) {
    const g = dp.gravity, b = dp.bounds;
    pf.set([dt, g[0], g[1], g[2], dp.h, 1 / dp.h, dp.kernelSigma, dp.rho0,
      dp.spacing, dp.viscosity, dp.vorticity, dp.friction,
      dp.maxSpeed, dp.bndInv, dp.contactMin, dp.maxOverdensity,
      dp.warmStart, dp.warmStart, dp.particleMass, dp.minNeighbors,
      b ? b.min[0] : 0, b ? b.min[1] : 0, b ? b.min[2] : 0, b ? 1 : 0,
      b ? b.max[0] : 0, b ? b.max[1] : 0, b ? b.max[2] : 0, dp.particleRadius + 0.25 * dp.spacing], 0);
    pu.set([colliderCount, T - 1, M, N, hfBase, tabBase, tabN, Math.min(drains.length, MAX_DRAINS)], 28);
    pf[36] = frames; pu[37] = nScan; pu[38] = nBlocks; pu[39] = 0;
    for (let k = 0; k < MAX_DRAINS; k++) {
      const d = drains[k];
      pf.set(d ? [...d.min, 0, ...d.max, 0] : [0, 0, 0, 0, 0, 0, 0, 0], 40 + k * 8);
    }
    device.queue.writeBuffer(buf.params, 0, pf);
  }

  /* ------------------------------ readback ----------------------------- */
  const readBytes = N * 16;
  const staging = device.createBuffer({ size: readBytes * 3 + nAtoms * 4, usage: GPUBufferUsage.MAP_READ | CD });
  const out = {
    pos: new Float32Array(N * 3), prev: new Float32Array(N * 3), vel: new Float32Array(N * 3),
    ids: new Int32Array(N), nbr: new Int32Array(N),
  };
  let count = 0, lastStats = {}, lastMs = 0, vmax = 0;
  const impulseAcc = new Float64Array(C * 6);
  let impulseTime = 0, contactStats = new Float64Array(C * 4);
  let drained = 0, leaked = 0, quarantined = 0;

  /* ------------------------------ stepping ----------------------------- */
  let pendingSpawn = [];
  let busy = false, onFrame = null;
  const wg = (n) => Math.ceil(n / 64);

  function dispatch(pass, entry, n) {
    const p = pipes[entry];
    pass.setPipeline(p.pipeline);
    pass.setBindGroup(0, p.group);
    pass.dispatchWorkgroups(n);
  }

  function encodeSubstep(enc, { last, final }) {
    if (last) { const p = enc.beginComputePass(); dispatch(p, 'markFrame', wg(N)); p.end(); }
    if (drains.length) { const p = enc.beginComputePass(); dispatch(p, 'drain', wg(N)); p.end(); }
    enc.clearBuffer(buf.cells);
    let p = enc.beginComputePass();
    dispatch(p, 'hash', wg(N));
    dispatch(p, 'scanBlocks', nBlocks);
    dispatch(p, 'scanTop', 1);
    dispatch(p, 'scanAdd', Math.ceil(nScan / 256));
    dispatch(p, 'scatter', wg(N));
    p.end();
    enc.copyBufferToBuffer(buf.posB, 0, buf.pos, 0, N * 16);
    enc.copyBufferToBuffer(buf.velB, 0, buf.vel, 0, N * 16);
    enc.copyBufferToBuffer(buf.prvB, 0, buf.prv, 0, N * 16);
    p = enc.beginComputePass();
    dispatch(p, 'neighbors', wg(N));
    dispatch(p, 'density', wg(N));
    for (let it = 0; it < divergenceIterations; it++) {
      if (it === 0) dispatch(p, 'warmDivergence', wg(N));
      dispatch(p, 'velocity', wg(N));
      dispatch(p, 'divergenceResidual', wg(N));
    }
    if (divergenceIterations > 0) dispatch(p, 'velocity', wg(N));
    dispatch(p, 'forces', wg(N));
    dispatch(p, 'confine', wg(N));
    dispatch(p, 'densityGate', wg(N));
    dispatch(p, 'warmDensity', wg(N));
    dispatch(p, 'velocity', wg(N));
    for (let it = 0; it < pressureIterations; it++) {
      if (final && it === pressureIterations - 1) {
        // the batch's error stats come from its last residual pass only
        p.end();
        enc.clearBuffer(buf.atoms, A.ERR_SUM * 4, 2 * 4);
        p = enc.beginComputePass();
      }
      dispatch(p, 'densityResidual', wg(N));
      dispatch(p, 'velocity', wg(N));
    }
    dispatch(p, 'integrate', wg(N));
    p.end();
  }

  function postSteps(k) {
    busy = true;
    const t0 = performance.now();
    // substeps from the CFL condition on the last known max speed
    const g = Math.hypot(...dp.gravity);
    const sub = Math.min(Math.max(1, dp.maxSubsteps | 0), Math.max(dp.minSubsteps ?? 1,
      Math.ceil(base.fixedDt * (vmax + g * base.fixedDt) / (dp.cfl * dp.spacing))));
    const dt = base.fixedDt / sub;
    writeParams(dt, dt * 60);
    const enc = device.createCommandEncoder();
    // per-batch accumulators
    enc.clearBuffer(buf.atoms, A.ERR_SUM * 4, (A.VMAX - A.ERR_SUM + 1) * 4);
    enc.clearBuffer(buf.atoms, A.IMPULSES * 4, C * 10 * 4);
    if (pendingSpawn.length) {
      const total = pendingSpawn.reduce((s, a) => s + a.length / 6, 0);
      const n = Math.min(total, N);
      const data = new Float32Array(4 + n * 8);
      data[0] = n;
      let q = 0;
      for (const a of pendingSpawn) {
        for (let r = 0; r + 5 < a.length && q < n; r += 6, q++) {
          data.set([a[r], a[r + 1], a[r + 2], 0, a[r + 3], a[r + 4], a[r + 5], 0], 4 + q * 8);
        }
      }
      pendingSpawn = [];
      device.queue.writeBuffer(buf.spawn, 0, data);
      const p = enc.beginComputePass();
      dispatch(p, 'appendSpawn', wg(n));
      dispatch(p, 'commitSpawn', 1);
      p.end();
    }
    for (let s = 0; s < k; s++) {
      for (let u = 0; u < sub; u++) encodeSubstep(enc, { last: s === k - 1 && u === 0, final: s === k - 1 && u === sub - 1 });
    }
    enc.copyBufferToBuffer(buf.pos, 0, staging, 0, readBytes);
    enc.copyBufferToBuffer(buf.prv, 0, staging, readBytes, readBytes);
    if (readVelocities) enc.copyBufferToBuffer(buf.vel, 0, staging, 2 * readBytes, readBytes);
    enc.copyBufferToBuffer(buf.atoms, 0, staging, 3 * readBytes, nAtoms * 4);
    device.queue.submit([enc.finish()]);
    drains.length = 0;
    staging.mapAsync(GPUMapMode.READ).then(() => {
      const raw = staging.getMappedRange();
      // a copy: the mapped views detach at unmap()
      const at = new Int32Array(raw, 3 * readBytes, nAtoms).slice();
      count = Math.max(0, Math.min(N, at[A.COUNT]));
      const P4 = new Float32Array(raw, 0, count * 4), R4 = new Float32Array(raw, readBytes, count * 4);
      for (let i = 0; i < count; i++) {
        out.pos[i * 3] = P4[i * 4]; out.pos[i * 3 + 1] = P4[i * 4 + 1]; out.pos[i * 3 + 2] = P4[i * 4 + 2];
        out.ids[i] = P4[i * 4 + 3];
        out.prev[i * 3] = R4[i * 4]; out.prev[i * 3 + 1] = R4[i * 4 + 1]; out.prev[i * 3 + 2] = R4[i * 4 + 2];
      }
      if (readVelocities) {
        const V4 = new Float32Array(raw, 2 * readBytes, count * 4);
        for (let i = 0; i < count; i++) {
          out.vel[i * 3] = V4[i * 4]; out.vel[i * 3 + 1] = V4[i * 4 + 1]; out.vel[i * 3 + 2] = V4[i * 4 + 2];
        }
      }
      const f = new Float32Array(1), fi = new Int32Array(f.buffer);
      fi[0] = at[A.ERR_MAX]; const errMax = f[0];
      fi[0] = at[A.VMAX]; vmax = f[0];
      drained = at[A.DRAINED]; leaked = at[A.LEAKED]; quarantined = at[A.QUARANTINED];
      const steps = k * sub;
      for (let q = 0; q < C * 6; q++) impulseAcc[q] += at[A.IMPULSES + q] / IMPULSE_SCALE;
      impulseTime += k * base.fixedDt;
      contactStats = new Float64Array(C * 4);
      for (let s = 0; s < C; s++) {
        const o = A.IMPULSES + C * 6 + s * 4, nC = at[o];
        if (nC > 0) {
          contactStats[s * 4] = nC / steps;
          for (let d = 1; d < 4; d++) contactStats[s * 4 + d] = at[o + d] / CONTACT_SCALE / nC;
        }
      }
      staging.unmap();
      lastMs = performance.now() - t0;
      lastStats = {
        kineticEnergy: NaN, maxDensityError: errMax,
        avgDensityError: count ? at[A.ERR_SUM] / 1e5 / count : 0,
        pressureIterations, divergenceIterations, substeps: sub, maxSpeed: vmax,
        overflow: at[A.OVERFLOW], leaked, quarantined, drained,
      };
      if (adaptive && count > 0) {
        const tol = dp.densityTolerance, e = lastStats.avgDensityError;
        if (e > tol && pressureIterations < dp.maxIterations) pressureIterations++;
        else if (e < 0.4 * tol && pressureIterations > (dp.minIterations ?? 2)) pressureIterations--;
      }
      busy = false;
      onFrame?.();
    });
  }

  function flushColliders(list) {
    const n = Math.min(list.length, C);
    for (let i = 0; i < n; i++) writeCollider(world, i, list[i]);
    colliderCount = n;
    device.queue.writeBuffer(buf.world, 0, world, 0, C * COLLIDER_STRIDE);
  }

  const atomsInit = new Int32Array(nAtoms);
  device.queue.writeBuffer(buf.atoms, 0, atomsInit);

  let owed = 0;
  const sim = {
    mode: 'gpu',
    backend: 'webgpu',
    maxColliders: C,
    fixedDt: base.fixedDt,
    threads: 0,
    ready: Promise.resolve(),
    device,
    get params() { return dp; },
    get count() { return count; },
    get positions() { return out.pos; },
    get prevPositions() { return out.prev; },
    get velocities() { return out.vel; },
    get neighborCounts() { return out.nbr; },
    get ids() { return out.ids; },
    get alpha() { return base.alpha; },
    get stepMs() { return lastMs; },
    get stats() { return lastStats; },
    get diffuse() { return { count: 0, data: new Float32Array(0) }; },
    get busy() { return busy; },
    takeImpulses() {
      const t = impulseTime; impulseTime = 0;
      const o = impulseAcc.slice();
      impulseAcc.fill(0);
      return { impulses: o, time: t, contacts: contactStats };
    },
    setParams(patch) {
      for (const [k, v] of Object.entries(patch)) dp[k] = k === 'gravity' ? [...v] : v;
    },
    spawn(particles) { const d = toSpawnArray(particles); pendingSpawn.push(d); return d.length / 6; },
    fillBox(min, max, o) { return sim.spawn(latticeBox(dp, min, max, o)); },
    removeInBox(min, max) { if (drains.length < MAX_DRAINS) drains.push({ min: [...min], max: [...max] }); },
    reset() {
      pendingSpawn = [];
      device.queue.writeBuffer(buf.atoms, A.COUNT * 4, new Int32Array([0]));
      count = 0;
    },
    setColliders(list) { flushColliders(list); },
    addHeightfield(desc) {
      const packed = desc instanceof Float32Array ? desc : packHeightfield(desc);
      if (hfCount >= MAX_HEIGHTFIELDS || hfUsed + packed.length > HF_CAPACITY) throw new Error('water gpu: heightfield capacity exceeded');
      world[hfBase + hfCount] = MAX_HEIGHTFIELDS + hfUsed;          // offset relative to hfBase
      world.set(packed, hfBase + MAX_HEIGHTFIELDS + hfUsed);
      hfUsed += packed.length;
      device.queue.writeBuffer(buf.world, hfBase * 4, world, hfBase, MAX_HEIGHTFIELDS + hfUsed);
      return hfCount++;
    },
    update(frameDt) {
      owed += base.takeSteps(frameDt);
      if (busy || owed === 0) return 0;
      const k = Math.min(owed, opts.maxStepsPerFrame ?? 3);
      owed = 0;
      postSteps(k);
      return k;
    },
    stepNow(steps = 1) {
      return new Promise((resolve) => {
        const run = () => { onFrame = () => { onFrame = null; resolve(); }; postSteps(steps); };
        if (!busy) run();
        else { const prev = onFrame; onFrame = () => { prev?.(); run(); }; }
      });
    },
    interpolate(o, alpha = base.alpha) { return interpolateInto(o, out.prev, out.pos, count, alpha); },
    dispose() { for (const b of Object.values(buf)) b.destroy(); staging.destroy(); device.destroy(); },
  };
  return sim;
}
