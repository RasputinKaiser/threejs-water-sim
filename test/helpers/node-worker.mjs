// test/helpers/node-worker.mjs — a browser-Worker-like wrapper around
// node:worker_threads for the water pack's workerFactory option.
import { Worker } from 'node:worker_threads';

const SHIM = new URL('./worker-shim.mjs', import.meta.url);
const ENTRY = new URL('../../src/water/core/worker.js', import.meta.url).href;

export function nodeWorkerFactory() {
  const w = new Worker(SHIM, { workerData: { entry: ENTRY } });
  return {
    postMessage: (m, t) => w.postMessage(m, t),
    addEventListener: (type, fn) => {
      if (type === 'message') w.on('message', (data) => fn({ data }));
      else if (type === 'error') w.on('error', (error) => fn({ error }));
    },
    terminate: () => w.terminate(),
  };
}
