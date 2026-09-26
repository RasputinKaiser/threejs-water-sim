// water/core/create-solver.js — pick the solver class for a parameter set.
import { PBFSolver } from './solver.js';
import { DFSPHSolver } from './dfsph.js';

export function createSolver(dp, buffers, opts) {
  return dp.solver === 'dfsph' ? new DFSPHSolver(dp, buffers, opts) : new PBFSolver(dp, buffers, opts);
}
