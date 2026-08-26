// tools/shot-server.mjs — tiny receiver for auto-shots from the debug world.
// POST /shot {label, dataUrl} → saves .shots/<label>-<timestamp>.png
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const outDir = path.join(root, '.shots');
const metricsDir = path.join(root, '.metrics');
fs.mkdirSync(outDir, { recursive: true });
fs.mkdirSync(metricsDir, { recursive: true });

let n = 0;
http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') { res.end(); return; }
  if (req.method === 'GET' && req.url === '/health') { res.end('ok'); return; }
  if (req.method === 'POST' && req.url === '/metrics') {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', () => {
      try {
        const { label, snapshot } = JSON.parse(body);
        // latest-wins per label (agent reads current state), plus append log
        fs.writeFileSync(path.join(metricsDir, `${label}.json`), JSON.stringify(snapshot, null, 1));
        fs.appendFileSync(path.join(metricsDir, `${label}.log`), JSON.stringify(snapshot) + '\n');
        res.end('ok');
      } catch (e) { res.statusCode = 400; res.end(e.message); }
    });
    return;
  }
  if (req.method === 'POST' && req.url === '/shot') {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', () => {
      try {
        const { label, dataUrl } = JSON.parse(body);
        if (!dataUrl || !dataUrl.startsWith('data:image/png;base64,')) {
          console.error(`bad payload: len=${body.length} head=${body.slice(0, 80)}`);
          res.statusCode = 400; res.end('bad dataUrl'); return;
        }
        const b64 = dataUrl.replace(/^data:image\/png;base64,/, '');
        const file = path.join(outDir, `${label}-${Date.now()}.png`);
        fs.writeFileSync(file, Buffer.from(b64, 'base64'));
        if (++n % 10 === 1) console.log(`saved ${file}`);
        res.end('ok');
      } catch (e) { res.statusCode = 400; res.end(e.message); }
    });
    return;
  }
  res.statusCode = 404; res.end();
}).listen(5185, () => console.log('shot server on :5185 → .shots/'));
