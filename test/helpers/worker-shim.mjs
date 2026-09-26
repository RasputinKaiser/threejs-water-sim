// test/helpers/worker-shim.mjs — runs src/water-pack/sim-worker.mjs inside a
// node worker_threads Worker by mapping the browser worker globals it uses
// (self / postMessage / addEventListener('message')) onto parentPort.
import { parentPort } from 'node:worker_threads';

globalThis.self = globalThis;
globalThis.postMessage = (msg) => parentPort.postMessage(msg);
globalThis.addEventListener = (type, fn) => {
  if (type === 'message') parentPort.on('message', (data) => fn({ data }));
};
globalThis.close = () => process.exit(0);

await import('../../src/water-pack/sim-worker.mjs');
