#!/usr/bin/env node
/* Static server + LLM scenario generation endpoint.
 *   node server.js            (default port 8811)
 *   ANTHROPIC_API_KEY=sk-... node server.js
 * POST /api/scenario {prompt} -> {cfg, tags, source:'llm'|'unavailable'}
 * GET  /api/status            -> {llm:boolean, model}
 * The API key stays on the server; the browser never sees it.
 */
const http = require('http'), fs = require('fs'), path = require('path');
const PORT = +process.env.PORT || 8811, ROOT = __dirname;
const KEY = process.env.ANTHROPIC_API_KEY, MODEL = process.env.SCENARIO_MODEL || 'claude-sonnet-5-5';
const BASE = process.env.ANTHROPIC_BASE_URL || 'https://api.anthropic.com';
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.onnx': 'application/octet-stream', '.jpg': 'image/jpeg', '.md': 'text/plain' };

const SYSTEM = `You convert a natural-language driving-scenario description into a JSON config for a highway driving simulator.
Reply with ONLY a JSON object, no prose, with these keys (omit any you are unsure of):
  "weather": one of "clear","rain","fog","night"
  "density": traffic vehicles per km, number 2..40 (default 10; rush hour ~34)
  "curve": road curvature multiplier 0.2..2 (1 = normal, 1.8 = winding mountain road)
  "events": object with boolean keys among "cutin","hardbrake","stopped","ped"
      cutin = neighbouring car merges in front; hardbrake = lead car brakes hard; stopped = stationary broken-down car ahead; ped = pedestrian crosses the road
  "setSpeedKmh": cruise set speed 30..140
  "tags": array of up to 6 short human-readable labels describing the generated scenario
Choose events that best stress-test what the user describes.`;

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
function sanitize(o) {
  const cfg = { events: {} }, tags = Array.isArray(o.tags) ? o.tags.slice(0, 6).map(t => String(t).slice(0, 40)) : [];
  if (['clear', 'rain', 'fog', 'night'].includes(o.weather)) cfg.weather = o.weather;
  if (isFinite(o.density)) cfg.density = clamp(+o.density, 2, 40);
  if (isFinite(o.curve)) cfg.curve = clamp(+o.curve, 0.2, 2);
  if (isFinite(o.setSpeedKmh)) cfg.setSpeed = clamp(+o.setSpeedKmh, 30, 140) / 3.6;
  for (const k of ['cutin', 'hardbrake', 'stopped', 'ped']) if (o.events && o.events[k] === true) cfg.events[k] = true;
  return { cfg, tags: tags.length ? tags : ['LLM scenario'] };
}

async function llmScenario(prompt) {
  const r = await fetch(BASE + '/v1/messages', {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-api-key': KEY, 'anthropic-version': '2023-06-01' },
    body: JSON.stringify({ model: MODEL, max_tokens: 500, system: SYSTEM, messages: [{ role: 'user', content: String(prompt).slice(0, 600) }] }),
  });
  if (!r.ok) throw new Error('Anthropic API ' + r.status + ': ' + (await r.text()).slice(0, 200));
  const j = await r.json(), text = (j.content || []).map(b => b.text || '').join('');
  const m = text.match(/\{[\s\S]*\}/); if (!m) throw new Error('no JSON in model reply');
  return sanitize(JSON.parse(m[0]));
}

function body(req) { return new Promise((res, rej) => { let b = ''; req.on('data', c => { b += c; if (b.length > 1e5) req.destroy(); }); req.on('end', () => res(b)); req.on('error', rej); }); }
const send = (res, code, obj) => { res.writeHead(code, { 'content-type': 'application/json' }); res.end(JSON.stringify(obj)); };

http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://x');
    if (url.pathname === '/api/status') return send(res, 200, { llm: !!KEY, model: MODEL });
    if (url.pathname === '/api/scenario' && req.method === 'POST') {
      if (!KEY) return send(res, 503, { error: 'ANTHROPIC_API_KEY not set on the server', source: 'unavailable' });
      const { prompt } = JSON.parse(await body(req) || '{}');
      if (!prompt) return send(res, 400, { error: 'missing prompt' });
      try { return send(res, 200, Object.assign(await llmScenario(prompt), { source: 'llm' })); }
      catch (e) { return send(res, 502, { error: e.message, source: 'unavailable' }); }
    }
    let f = path.normalize(path.join(ROOT, url.pathname === '/' ? 'index.html' : decodeURIComponent(url.pathname)));
    if (!f.startsWith(ROOT) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); return res.end('not found'); }
    res.writeHead(200, { 'content-type': MIME[path.extname(f)] || 'application/octet-stream' }); fs.createReadStream(f).pipe(res);
  } catch (e) { send(res, 500, { error: String(e) }); }
}).listen(PORT, () => console.log(`AutoPilot Sim on http://localhost:${PORT}  (LLM scenarios: ${KEY ? 'enabled, model ' + MODEL : 'disabled - set ANTHROPIC_API_KEY'})`));
