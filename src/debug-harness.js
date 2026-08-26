// debug-harness.js — shared debug scaffolding for the Box3D world.
// Scene, camera + orbit controls, lights/shadows, HUD readout, console capture,
// screenshot hotkey (P), pause/step (Space/.), and a fixed-timestep frame loop.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import GUI from 'lil-gui';
import Stats from 'stats.js';

export function createDebugHarness({ container, cameraPos = [14, 11, 16], target = [0, 2.5, 0] } = {}) {
  if (!container) {
    container = document.getElementById('app');
    if (!container) {
      container = document.createElement('div');
      container.id = 'app';
      container.style.cssText = 'width:100%;height:100%';
      document.body.appendChild(container);
    }
  }
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x101014);

  const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 500);
  camera.position.set(...cameraPos);

  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(...target);

  // lights
  scene.add(new THREE.HemisphereLight(0x9db4dd, 0x3a4152, 1.1));
  const sun = new THREE.DirectionalLight(0xfff4e0, 2.6);
  sun.position.set(12, 20, 8);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = -25; sun.shadow.camera.right = 25;
  sun.shadow.camera.top = 25; sun.shadow.camera.bottom = -25;
  sun.shadow.camera.far = 80;
  sun.shadow.bias = -0.0005;
  scene.add(sun);

  // image-based lighting so reflective materials (water!) read correctly
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.5;

  // grid + axes
  const gridHelper = new THREE.GridHelper(40, 40, 0x4a5261, 0x333845);
  gridHelper.position.y = -0.001;
  const axesHelper = new THREE.AxesHelper(3);

  // ---- shared overlay chrome (nav / HUD / console / hint) ----
  // Injected here so every scene page gets identical styling; index.html keeps
  // matching copies in its <head>. ?hud=0 hides HUD readouts + stats entirely
  // for clean screenshots.
  const hudHidden = new URLSearchParams(location.search).get('hud') === '0';
  {
    const style = document.createElement('style');
    style.textContent = `
      #scene-nav{position:fixed;top:8px;left:50%;transform:translateX(-50%);z-index:100;display:flex;gap:6px;
        background:rgba(10,12,18,.72);padding:5px 8px;border-radius:9px;font:12px system-ui,sans-serif;backdrop-filter:blur(4px)}
      #scene-nav a{color:#9fb0cc;text-decoration:none;padding:3px 9px;border-radius:6px}
      #scene-nav a:hover{color:#fff;background:#2a3040}
      #scene-nav a.active{color:#fff;background:#2f6fed}
      #scene-nav a.active .fps{margin-left:7px;font:600 10px ui-monospace,Menlo,monospace;color:#cfe0ff;
        background:rgba(255,255,255,.14);padding:1px 5px;border-radius:5px}
      #hud{position:fixed;left:10px;top:10px;z-index:999;font:12px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;
        color:#9aa4b5;background:rgba(10,10,14,0.75);border:1px solid #26262f;
        border-radius:6px;padding:8px 12px;min-width:230px;white-space:pre}
      #hud b { color: #e8eaf0; font-weight: 600; }
      #hud .ok { color: #56d364; } #hud .bad { color: #ff7b72; } #hud .warn { color: #e3b341; }
      #console-overlay{
        position:fixed;left:8px;bottom:8px;z-index:1000;width:460px;max-height:180px;
        background:rgba(10,10,14,0.88);border:1px solid #2a2a35;border-radius:6px;
        font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;
        color:#c8c8d4;overflow:hidden;display:flex;flex-direction:column;pointer-events:auto}
      #console-overlay .hdr{
        padding:3px 8px;cursor:pointer;user-select:none;color:#7f8ea3;font-size:10px;letter-spacing:.08em;text-transform:uppercase;
        border-bottom:1px solid #22222c;display:flex;justify-content:space-between;align-items:center;gap:12px}
      #console-overlay .log{padding:4px 8px;overflow-y:auto;flex:1;white-space:pre-wrap;word-break:break-word}
      #console-overlay .err{color:#ff7b72}
      #console-overlay .warn{color:#e3b341}
      #console-overlay .info{color:#8b949e}
      #console-overlay.min .log{display:none}
      #console-overlay .legend{display:inline-flex;gap:8px;align-items:center;text-transform:none;letter-spacing:0;color:#9aa4b5}
      #console-overlay .dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:3px}
      #console-overlay .dot.e{background:#ff7b72}#console-overlay .dot.w{background:#e3b341}#console-overlay .dot.i{background:#58a6ff}`;
    document.head.appendChild(style);
  }

  // ---- scene switcher (top-center nav between lab scenes) ----
  let navFpsEl = null;
  {
    const nav = document.createElement('div');
    nav.id = 'scene-nav';
    const here = location.pathname.split('/').pop() || 'index.html';
    const scenes = [
      ['index.html', '🏊 Pool Lab'],
      ['bucket.html', '🪣 Bucket'],
      ['terrain.html', '⛰ Terrain'],
      ['pool.html', '🤽 Big Pool'],
      ['creek.html', '🌊 Creek'],
    ];
    nav.innerHTML = scenes.map(([file, label]) => {
      const badge = file === here && !hudHidden ? ' <span class="fps" id="nav-fps">—</span>' : '';
      return `<a href="/${file}" class="${file === here ? 'active' : ''}">${label}${badge}</a>`;
    }).join('');
    document.body.appendChild(nav);
    navFpsEl = nav.querySelector('#nav-fps');
  }

  // gui
  const gui = new GUI({ title: '🧪 Debug World' });
  const viewFolder = gui.addFolder('View');
  const viewState = { grid: true, axes: true };
  viewFolder.add(viewState, 'grid').onChange(v => gridHelper.visible = v);
  viewFolder.add(viewState, 'axes').onChange(v => axesHelper.visible = v);
  viewFolder.close();

  const stats = new Stats();
  stats.dom.style.cssText += ';left:auto;right:10px;top:10px;';
  document.body.appendChild(stats.dom);
  if (hudHidden) stats.dom.style.display = 'none';

  // ---- HUD ----
  const hudEl = document.createElement('div');
  hudEl.id = 'hud';
  if (hudHidden) hudEl.style.display = 'none';
  document.body.appendChild(hudEl);
  let hudLinesFn = () => [];
  function setHudProvider(fn) { hudLinesFn = fn; }
  function updateHud() {
    if (hudHidden) return; // ?hud=0 — stay hidden for clean screenshots
    const lines = hudLinesFn();
    if (!lines.length) { hudEl.style.display = 'none'; return; }
    hudEl.style.display = '';
    hudEl.innerHTML = lines.join('\n');
  }

  // ---- console capture overlay ----
  const MAX_LOG = 100; // stored entries
  const overlay = document.createElement('div');
  overlay.id = 'console-overlay';
  overlay.className = 'min'; // collapsed by default
  overlay.innerHTML =
    `<div class="hdr"><span class="legend"><i class="dot e"></i>err<i class="dot w"></i>warn<i class="dot i"></i>info</span>` +
    `<span><span id="con-count"></span>console</span></div><div class="log" id="con-log"></div>`;
  document.body.appendChild(overlay);
  const conLog = overlay.querySelector('#con-log');
  const conCount = overlay.querySelector('#con-count');
  let errCount = 0;
  let lastErrorAt = -Infinity;
  let errorExpanded = false; // true only while the error-triggered expansion is active
  overlay.querySelector('.hdr').addEventListener('click', () => {
    errorExpanded = false; // manual toggle wins — no auto re-collapse after this
    overlay.classList.toggle('min');
  });
  function pushLog(kind, args) {
    if (kind === 'error') {
      errCount++;
      lastErrorAt = performance.now();
      if (!errorExpanded) { overlay.classList.remove('min'); errorExpanded = true; }
    }
    conCount.textContent = errCount ? `${errCount} ✗ ` : '';
    const line = document.createElement('div');
    line.className = kind;
    line.textContent = `[${kind}] ${args.map(a => {
      try { return typeof a === 'string' ? a : JSON.stringify(a); } catch { return String(a); }
    }).join(' ')}`;
    conLog.appendChild(line);
    while (conLog.children.length > MAX_LOG) conLog.removeChild(conLog.firstChild);
    conLog.scrollTop = conLog.scrollHeight;
  }
  // auto-collapse again once ≥5s have passed with no new errors (unless the user toggled)
  setInterval(() => {
    if (errorExpanded && performance.now() - lastErrorAt > 5000) {
      overlay.classList.add('min');
      errorExpanded = false;
    }
  }, 1000);
  const origError = console.error.bind(console);
  const origWarn = console.warn.bind(console);
  console.error = (...a) => { pushLog('error', a); origError(...a); };
  console.warn = (...a) => { pushLog('warn', a); origWarn(...a); };
  window.addEventListener('error', e => pushLog('error', [e.message]));
  window.pushDbg = (msg) => pushLog('info', [msg]);

  // ---- screenshot / pause / step keys ----
  window.addEventListener('keydown', (e) => {
    if (e.key === 'p' || e.key === 'P') takeScreenshot();
    else if (e.code === 'Space') { state.paused = !state.paused; window.pushDbg(`paused=${state.paused}`); }
    else if (e.key === '.') { stepOnce(); }
  });

  function takeScreenshot() {
    renderFrame();
    const url = renderer.domElement.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = `debug-${Date.now()}.png`;
    a.click();
    window.pushDbg('screenshot saved to ~/Downloads');
  }

  // ---- fixed timestep loop with pause/step ----
  const state = { paused: false };
  let fpsEma = 60;
  // nav fps badge — subtle live readout next to the active scene name (2x/s)
  setInterval(() => { if (navFpsEl) navFpsEl.textContent = `${Math.round(fpsEma)}`; }, 500);
  let pendingStep = false;
  function stepOnce() { pendingStep = true; }
  const callbacks = [];
  let last = performance.now();
  let acc = 0;
  const FIXED_DT = 1 / 60;

  // ---- optional compositor hook: when set, replaces the plain renderFrame
  // (scene → screen) with fn(renderer, scene, camera), which typically renders
  // the scene offscreen and composites water. Unset (null) = default render.
  let compositor = null;
  function setCompositor(fn) { compositor = typeof fn === 'function' ? fn : null; }

  function renderFrame() {
    if (compositor) { compositor(renderer, scene, camera); return; }
    renderer.render(scene, camera);
  }

  function animate() {
    requestAnimationFrame(animate);
    stats.begin();
    const now = performance.now();
    let delta = Math.min((now - last) / 1000, 1 / 20);
    last = now;
    // real fps (EMA) — stats.js object doesn't expose .fps
    fpsEma = fpsEma * 0.95 + (1 / Math.max(delta, 1e-4)) * 0.05;
    if (!state.paused || pendingStep) {
      acc += delta;
      let steps = 0;
      while (acc >= FIXED_DT && steps < 4) {
        for (const cb of callbacks) cb(FIXED_DT);
        acc -= FIXED_DT;
        steps++;
        pendingStep = false;
      }
    }
    controls.update();
    updateHud();
    renderFrame();
    stats.end();
  }

  return {
    scene, camera, renderer, controls, gui, stats,
    onFixed(cb) { callbacks.push(cb); },
    setCompositor,
    start() { last = performance.now(); animate(); },
    setHudProvider, takeScreenshot, stepOnce,
    /** Collapse every GUI folder (recursively) so param subfolders default to
     * closed. Safe to call again after late addGui() calls (e.g. async foam). */
    tidyGui(root = gui) { closeFoldersDeep(root); },
    get paused() { return state.paused; },
    get fps() { return fpsEma; },
  };

  function closeFoldersDeep(g) {
    for (const f of g.folders ?? []) { f.close(); closeFoldersDeep(f); }
  }
}
