// water-pack/metrics.js — agent-facing telemetry.
// When the URL has ?metrics=<label>, POSTs a JSON snapshot every N seconds to
// the shot server (:5185/metrics) → saved as .metrics/<label>.json (latest
// wins). Lets an agent read sim health (counts, leaks, energy, fill, census)
// without touching the GUI. Pairs with ?autoshot for visuals.

export function setupMetrics(getSnapshot, { interval = 2, port = '5185' } = {}) {
  const params = new URLSearchParams(location.search);
  if (!params.has('metrics')) return;
  const label = params.get('metrics') || 'run';
  setInterval(async () => {
    try {
      const snap = typeof getSnapshot === 'function' ? getSnapshot() : getSnapshot;
      await fetch(`http://localhost:${port}/metrics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label, snapshot: snap }),
      });
    } catch { /* shot server not running — silent */ }
  }, interval * 1000);
}
