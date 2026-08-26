// debug-shot.js — headless-friendly screenshot pipeline.
// If the URL has ?autoshot=<seconds> (or &autoshot), the page captures its canvas
// every N seconds and POSTs it to the shot server (default :5185). Files land in
// .shots/ in the repo so an agent can inspect them without touching the GUI.

export function setupAutoShots(renderer, defaultInterval = 4) {
  const params = new URLSearchParams(location.search);
  if (!params.has('autoshot')) return;
  const interval = Math.max(0.5, parseFloat(params.get('autoshot')) || defaultInterval);
  const port = params.get('shotport') || '5185';
  const label = params.get('label') || 'run';

  setInterval(async () => {
    try {
      const dataUrl = renderer.domElement.toDataURL('image/png');
      await fetch(`http://localhost:${port}/shot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label, dataUrl }),
      });
    } catch (e) {
      console.warn('autoshot failed:', e.message);
    }
  }, interval * 1000);
}
