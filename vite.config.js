import { defineConfig } from 'vite';
import { readdirSync } from 'node:fs';
import { resolve } from 'node:path';

// Multi-page build: every *.html at the repo root becomes an entry,
// so new lab pages are picked up automatically.
const htmlInputs = Object.fromEntries(
  readdirSync(import.meta.dirname)
    .filter((f) => f.endsWith('.html'))
    .map((f) => [f.replace(/\.html$/, ''), resolve(import.meta.dirname, f)]),
);

export default defineConfig({
  server: {
    port: 5184,
    strictPort: true,
    host: true,
    // Cross-origin isolation — required for SharedArrayBuffer / Web Workers
    // with atomics (the threaded water solver, see src/water/sim.js).
    // NOTE: any third-party script/image loaded without CORP will break under
    // require-corp; CDN assets need crossorigin/CORS.
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  build: {
    rollupOptions: {
      input: htmlInputs,
    },
  },
});
