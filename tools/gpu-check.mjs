#!/usr/bin/env node
// tools/gpu-check.mjs — run test/gpu/check.html in headless Chromium with
// WebGPU (SwiftShader) against the dev server: GPU backend vs CPU solver.
// Usage: npm run dev (other terminal), then node tools/gpu-check.mjs [column,dam,buoy]
const { chromium } = await import(process.env.PLAYWRIGHT ?? 'playwright'); // not a dependency: point PLAYWRIGHT at an install
const scen = process.argv[2] ?? 'column,dam,buoy';
const port = process.env.PORT ?? 5184;
const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage();
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log(`[${m.type()}] ${m.text()}`); });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`http://localhost:${port}/test/gpu/check.html?s=${scen}`);
await page.waitForFunction(() => window.__result, null, { timeout: 30 * 60 * 1000 });
console.log(await page.evaluate(() => document.getElementById('log').textContent));
const r = await page.evaluate(() => window.__result);
await browser.close();
process.exit(r.error ? 1 : 0);
