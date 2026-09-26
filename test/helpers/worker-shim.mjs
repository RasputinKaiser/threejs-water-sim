// test/helpers/worker-shim.mjs — runs a browser worker module inside a node
// worker_threads Worker by mapping the worker globals it uses
// (self / postMessage / addEventListener('message') / close) onto parentPort.
// workerData.entry: module URL to load (default: the legacy water-pack worker).
import { parentPort, workerData } from 'node:worker_threads';

globalThis.self = globalThis;
globalThis.postMessage = (msg, transfer) => parentPort.postMessage(msg, transfer);
globalThis.addEventListener = (type, fn) => {
  if (type === 'message') parentPort.on('message', (data) => fn({ data }));
};
globalThis.close = () => process.exit(0);

await import(workerData?.entry ?? '../../src/water-pack/sim-worker.mjs');
