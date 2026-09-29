(function () {
'use strict';
const { World, makeAI, parsePrompt, genEpisode, CAM, LW, ROAD_HALF } = Sim;
const $ = id => document.getElementById(id);
const camC = $('cam'), ovC = $('ov'), cam = camC.getContext('2d', { willReadFrequently: true }), ov = ovC.getContext('2d');
const topC = $('top'), top = topC.getContext('2d'), edgeC = $('edge'), edge = edgeC.getContext('2d');
const ai = makeAI(7);
let W = null, paused = false, speed = 1, scenarioCfg = {}, cvRes = null, cvMs = 0, frame = 0, lastLog = 0;
const pref = { policy: 'expert', perception: 'gt', detector: 'sim', autoPass: false, setSpeed: 100 / 3.6 };
let yoloBusy = false, yoloDets = [], yoloMs = 0, llmOn = false;
const proj = CV.proj;

/* ------------------------------------------------ bootstrap (pre-train NN headlessly) */
async function bootstrap() {
  $('boot').style.display = 'flex';
  const eps = 10, bar = $('bar').firstChild, msg = $('bootmsg');
  ai.net = new Sim.MLP([Sim.NFEAT, 32, 32, 2], ai.rnd); ai.n = 0; ai.head = 0; ai.lossHist = []; ai.steps = 0; ai.lastLoss = NaN;
  for (let i = 0; i < eps; i++) {
    genEpisode(ai, i + 1, 90); bar.style.width = (i + 1) / eps * 50 + '%';
    msg.textContent = `Expert demonstrations: episode ${i + 1}/${eps} (${ai.n} samples)`;
    await new Promise(r => setTimeout(r, 0));
  }
  const chunks = 40;
  for (let i = 0; i < chunks; i++) {
    ai.trainSteps(50, 64, i < 25 ? 3e-3 : 1e-3); bar.style.width = 50 + (i + 1) / chunks * 50 + '%';
    msg.textContent = `Training network: loss ${ai.lastLoss.toFixed(4)}`;
    await new Promise(r => setTimeout(r, 0));
  }
  $('boot').style.display = 'none';
  newWorld({});
}
function newWorld(cfg) {
  scenarioCfg = cfg;
  W = new World(ai, Object.assign({ collect: true }, cfg, pref, { seed: Math.floor(Math.random() * 1e6) }));
  W.ego.setSpeed = pref.setSpeed; cvRes = null; W.alert('Scenario loaded', 'info');
}

/* ------------------------------------------------ controls */
const setOn = (a, b, first) => { a.classList.toggle('on', first); b.classList.toggle('on', !first); };
function syncEng() { const b = $('bEng'); b.classList.toggle('on', W && W.ego.engaged); b.textContent = W && W.ego.engaged ? 'Engaged' : 'Disengaged'; }
function toggleEngage() { if (!W || W.crashed) return; W.ego.engaged ? W.disengage('Disengaged by driver', false) : W.engage(); }
$('bEng').onclick = toggleEngage;
$('bLL').onclick = () => W.requestLaneChange(1); $('bLR').onclick = () => W.requestLaneChange(-1);
$('cPass').onchange = e => { pref.autoPass = e.target.checked; W.cfg.autoPass = pref.autoPass; };
$('rSet').oninput = e => { $('vSet').textContent = e.target.value; pref.setSpeed = e.target.value / 3.6; W.ego.setSpeed = pref.setSpeed; };
$('bPause').onclick = e => { paused = !paused; e.target.textContent = paused ? 'Resume' : 'Pause'; };
$('sSpd').onchange = e => speed = +e.target.value;
$('pExp').onclick = () => { pref.policy = W.cfg.policy = 'expert'; setOn($('pExp'), $('pNN'), true); };
$('pNN').onclick = () => { pref.policy = W.cfg.policy = 'nn'; setOn($('pExp'), $('pNN'), false); };
$('cGT').onclick = () => { pref.perception = W.cfg.perception = 'gt'; setOn($('cGT'), $('cCV'), true); };
$('cCV').onclick = () => { pref.perception = W.cfg.perception = 'cv'; setOn($('cGT'), $('cCV'), false); W.cvLane = null; W.laneLost = 0; };
$('dSim').onclick = () => { pref.detector = W.cfg.detector = 'sim'; W.tracks.clear(); setOn($('dSim'), $('dYolo'), true); $('yoloInfo').textContent = ''; };
$('dYolo').onclick = () => {
  pref.detector = W.cfg.detector = 'yolo'; W.tracks.clear(); W.yolo = null; setOn($('dSim'), $('dYolo'), false);
  Yolo.init(m => $('yoloInfo').textContent = m).catch(e => { $('dSim').click(); $('yoloInfo').textContent = 'YOLO failed: ' + e.message; });
};
$('bTrain').onclick = e => { ai.trainOn = !ai.trainOn; e.target.classList.toggle('on', ai.trainOn); };
$('bBoot').onclick = () => bootstrap();
function applyPrompt(text) {
  const r = parsePrompt(text);
  $('tags').innerHTML = r.tags.map(t => `<span>${t}</span>`).join('');
  const cfg = r.cfg; if (cfg.setSpeed) { pref.setSpeed = cfg.setSpeed; $('rSet').value = Math.round(cfg.setSpeed * 3.6); $('vSet').textContent = $('rSet').value; delete cfg.setSpeed; }
  newWorld(cfg);
}
async function generate(text) {
  if (llmOn) {
    try {
      const r = await fetch('/api/scenario', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ prompt: text }) });
      const j = await r.json();
      if (r.ok && j.cfg) {
        $('tags').innerHTML = j.tags.map(t => `<span>${t}</span>`).join('') + '<span style="border-color:#b48cff">LLM</span>';
        const cfg = j.cfg; if (cfg.setSpeed) { pref.setSpeed = cfg.setSpeed; $('rSet').value = Math.round(cfg.setSpeed * 3.6); $('vSet').textContent = $('rSet').value; delete cfg.setSpeed; }
        return newWorld(cfg);
      }
      logMsg('LLM scenario failed (' + (j.error || r.status) + ') - using keyword grammar', 'warn');
    } catch (e) { logMsg('LLM server unreachable - using keyword grammar', 'warn'); }
  }
  applyPrompt(text);
}
$('bGen').onclick = () => generate($('prompt').value);
fetch('/api/status').then(r => r.json()).then(j => { llmOn = !!j.llm; $('llmInfo').innerHTML = j.llm ? `LLM scenarios <b style="color:#b48cff">enabled</b> (${j.model}) - Generate uses Claude; preset buttons use the keyword grammar.` : 'LLM off: run <kbd>node server.js</kbd> with <kbd>ANTHROPIC_API_KEY</kbd> set to enable Claude-generated scenarios (keyword grammar in use).'; })
  .catch(() => { $('llmInfo').innerHTML = 'LLM off (static server): run <kbd>node server.js</kbd> with <kbd>ANTHROPIC_API_KEY</kbd> for Claude-generated scenarios. Keyword grammar in use.'; });
document.querySelectorAll('[data-sc]').forEach(b => b.onclick = () => { $('prompt').value = b.dataset.sc; applyPrompt(b.dataset.sc); });
const keys = {};
addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' && e.target.type === 'text') return;
  keys[e.key] = true;
  if (e.key === ' ') { toggleEngage(); e.preventDefault(); }
  else if (e.key === 'q' || e.key === 'Q') W.requestLaneChange(1);
  else if (e.key === 'e' || e.key === 'E') W.requestLaneChange(-1);
  else if (e.key === 'r' || e.key === 'R') newWorld(scenarioCfg);
  if (e.key.startsWith('Arrow')) e.preventDefault();
});
addEventListener('keyup', e => keys[e.key] = false);

/* ------------------------------------------------ overlay on camera */
function drawOverlay() {
  ov.clearRect(0, 0, CAM.W, CAM.H);
  const P = W.P, e = W.ego; if (!P) return;
  // openpilot-style planned path ribbon
  const pts = [];
  for (let x = 5; x <= 70; x += 2.5) { const y = P.path.c0 + P.path.c1 * x + P.path.c2 * x * x; pts.push([x, y]); }
  const hw = 0.9, a = [], b = [];
  for (const [x, y] of pts) { a.push(proj(x, y + hw, 0)); b.push(proj(x, y - hw, 0)); }
  const g = ov.createLinearGradient(0, CAM.H, 0, CAM.vh);
  g.addColorStop(0, e.engaged ? 'rgba(57,229,140,0.55)' : 'rgba(150,150,150,0.45)'); g.addColorStop(1, 'rgba(57,229,140,0.0)');
  ov.beginPath(); ov.moveTo(a[0][0], a[0][1]); for (const p of a) ov.lineTo(p[0], p[1]); for (let i = b.length - 1; i >= 0; i--) ov.lineTo(b[i][0], b[i][1]); ov.closePath(); ov.fillStyle = g; ov.fill();
  // CV lane fits
  if (cvRes && pref.perception === 'cv') {
    ov.lineWidth = 2;
    for (const [fit, col] of [[cvRes.fitL, '#ff6b6b'], [cvRes.fitR, '#4cc9f0']]) {
      if (!fit) continue; ov.strokeStyle = col; ov.beginPath();
      for (let x = 6; x <= 46; x += 2) { const [u, v] = proj(x, fit[0] + fit[1] * x + fit[2] * x * x, 0); x === 6 ? ov.moveTo(u, v) : ov.lineTo(u, v); }
      ov.stroke();
    }
  }
  // detections
  ov.font = '11px system-ui'; ov.lineWidth = 1.5;
  for (const t of P.tracks) {
    if (t.miss > 3 || t.x < 3) continue;
    const [u, v0] = proj(t.x, t.y, 0), [, v1] = proj(t.x, t.y, t.hgt), w = CAM.f * t.wid / t.x;
    const lead = t === P.lead, col = lead ? '#ff7a45' : (t.pMerge > 0.4 ? '#ffb020' : '#4cc9f0');
    ov.strokeStyle = col; ov.strokeRect(u - w / 2, v1, w, v0 - v1);
    const lbl = `${t.cls} ${t.conf.toFixed(2)} ${t.x.toFixed(0)}m${t.pMerge > 0.15 && !t.inPath ? ' cut-in ' + Math.round(t.pMerge * 100) + '%' : ''}`;
    const tw = ov.measureText(lbl).width + 6, ly = Math.max(12, v1 - 14);
    ov.fillStyle = col; ov.fillRect(u - w / 2, ly, tw, 13); ov.fillStyle = '#000'; ov.fillText(lbl, u - w / 2 + 3, ly + 10);
  }
  if (pref.detector === 'yolo') {
    ov.setLineDash([4, 3]); ov.strokeStyle = '#e879f9'; ov.lineWidth = 1.2; ov.fillStyle = '#e879f9'; ov.font = '10px system-ui';
    for (const d of yoloDets) { ov.strokeRect(d.x0, d.y0, d.w, d.h); ov.fillText(`YOLO ${d.cls} ${d.conf.toFixed(2)}`, d.x0, d.y0 - 2); }
    ov.setLineDash([]);
  }
  // HUD
  ov.fillStyle = 'rgba(0,0,0,0.45)'; ov.fillRect(8, 8, 150, 62);
  ov.fillStyle = '#fff'; ov.font = 'bold 26px system-ui'; ov.fillText((e.v * 3.6).toFixed(0), 16, 40);
  ov.font = '11px system-ui'; ov.fillStyle = '#9fb2c4'; ov.fillText('km/h', 62, 40);
  ov.fillStyle = e.engaged ? '#39e58c' : '#ff9f43'; ov.fillText((e.engaged ? 'ENGAGED · ' : 'MANUAL · ') + (W.cfg.policy === 'nn' ? 'NN policy' : 'planner') + ' · ' + (W.cfg.perception === 'cv' ? 'CV' : 'GT'), 16, 58);
  ov.fillStyle = '#ffb020'; ov.fillText('set ' + Math.round(Math.min(e.setSpeed, W.I ? W.I.vset : e.setSpeed) * 3.6), 112, 40);
  const al = W.curAlert;
  if (al && al.until > W.t) {
    ov.fillStyle = al.level === 'crit' ? 'rgba(200,30,30,0.85)' : (al.level === 'warn' ? 'rgba(190,120,0,0.85)' : 'rgba(20,80,60,0.8)');
    ov.fillRect(120, CAM.H - 44, CAM.W - 240, 30); ov.fillStyle = '#fff'; ov.font = 'bold 15px system-ui'; ov.textAlign = 'center';
    ov.fillText(al.msg, CAM.W / 2, CAM.H - 24); ov.textAlign = 'left';
  }
  if (W.crashed) {
    ov.fillStyle = 'rgba(120,0,0,0.6)'; ov.fillRect(0, 0, CAM.W, CAM.H); ov.fillStyle = '#fff'; ov.font = 'bold 28px system-ui'; ov.textAlign = 'center';
    ov.fillText(W.crashed.kind.toUpperCase(), CAM.W / 2, CAM.H / 2 - 6); ov.font = '14px system-ui'; ov.fillText('press R to reset', CAM.W / 2, CAM.H / 2 + 20); ov.textAlign = 'left';
  }
}

/* ------------------------------------------------ bird's eye */
function drawTop() {
  const w = topC.width, h = topC.height, sc = 2.6, ox = w / 2, oy = h * 0.82, R = W.road, pose = W.pose, e = W.ego;
  top.fillStyle = '#0a1016'; top.fillRect(0, 0, w, h);
  const T = (x, y) => [ox - y * sc, oy - x * sc], p = { x: 0, y: 0 }, q = { x: 0, y: 0 };
  const strip = (d0, d1, style, dash) => {
    top.beginPath(); let first = true;
    for (let ds = -30; ds <= 140; ds += 4) { R.toEgo(pose, e.s + ds, d0, p); const [u, v] = T(p.x, p.y); first ? top.moveTo(u, v) : top.lineTo(u, v); first = false; }
    if (d1 !== null) for (let ds = 140; ds >= -30; ds -= 4) { R.toEgo(pose, e.s + ds, d1, p); const [u, v] = T(p.x, p.y); top.lineTo(u, v); }
    if (d1 !== null) { top.fillStyle = style; top.fill(); } else { top.strokeStyle = style; top.setLineDash(dash || []); top.lineWidth = 1; top.stroke(); top.setLineDash([]); }
  };
  strip(ROAD_HALF, -ROAD_HALF, '#1b232d');
  strip(ROAD_HALF, null, '#e6b800'); strip(-ROAD_HALF, null, '#8a97a5'); strip(LW / 2, null, '#5a6672', [6, 8]); strip(-LW / 2, null, '#5a6672', [6, 8]);
  // generative futures
  for (const f of W.fut) for (const s of f.samples) {
    top.strokeStyle = s.hit ? 'rgba(255,77,77,0.55)' : 'rgba(180,140,255,0.28)'; top.lineWidth = 1.2; top.beginPath();
    let first = true; for (const [x, y] of s.pts) { const [u, v] = T(x, y); first ? top.moveTo(u, v) : top.lineTo(u, v); first = false; } top.stroke();
    const [u, v] = T(s.pts[s.pts.length - 1][0], s.pts[s.pts.length - 1][1]); top.fillStyle = s.hit ? 'rgba(255,77,77,0.8)' : 'rgba(180,140,255,0.6)'; top.fillRect(u - 1.5, v - 1.5, 3, 3);
  }
  const box = (o, col) => {
    R.toEgo(pose, o.s, o.d, p); R.toEgo(pose, o.s + 1, o.d, q);
    const ang = Math.atan2(q.y - p.y, q.x - p.x), [u, v] = T(p.x, p.y);
    top.save(); top.translate(u, v); top.rotate(-(Math.PI / 2 - ang) * 1 - 0); top.rotate(0);
    top.restore();
    const c = Math.cos(ang), s = Math.sin(ang), L = o.len / 2, Wd = o.wid / 2;
    top.beginPath();
    [[-L, -Wd], [L, -Wd], [L, Wd], [-L, Wd]].forEach(([a, b], i) => { const [uu, vv] = T(p.x + a * c - b * s, p.y + a * s + b * c); i ? top.lineTo(uu, vv) : top.moveTo(uu, vv); });
    top.closePath(); top.fillStyle = col; top.fill();
  };
  for (const o of W.objs) box(o, o.type === 'ped' ? '#ff9f43' : (o.type === 'truck' ? '#8391a3' : '#5d7288'));
  if (W.P && W.P.lead) { const o = W.objs.find(o => o.id === W.P.lead.id); if (o) box(o, '#ff7a45'); }
  box({ s: e.s, d: e.d, len: e.len, wid: e.wid }, e.engaged ? '#39e58c' : '#ffb020');
  top.fillStyle = '#7d8ea1'; top.font = '10px system-ui'; top.fillText('lead ' + (W.P && W.P.lead ? (W.P.lead.x - 2.3).toFixed(0) + ' m' : '—') + '   risk ' + (W.P ? Math.round(W.P.risk * 100) : 0) + '%', 8, 14);
}

/* ------------------------------------------------ CV debug panel */
function drawEdge() {
  if (!cvRes) return;
  CV.edgeMap(cvRes.img, edge);
  edge.fillStyle = '#ff6b6b'; for (const p of cvRes.Lp) edge.fillRect(p[2] / 2 - 1.5, p[3] / 2 - 1.5, 3, 3);
  edge.fillStyle = '#4cc9f0'; for (const p of cvRes.Rp) edge.fillRect(p[2] / 2 - 1.5, p[3] / 2 - 1.5, 3, 3);
  edge.strokeStyle = '#39e58c'; edge.lineWidth = 1.5;
  if (cvRes.valid) {
    edge.beginPath();
    for (let x = 6; x <= 46; x += 2) { const y = cvRes.c0 + cvRes.c1 * x + cvRes.c2 * x * x, [u, v] = proj(x, y, 0); x === 6 ? edge.moveTo(u / 2, v / 2) : edge.lineTo(u / 2, v / 2); }
    edge.stroke();
  }
  $('cvInfo').textContent = `${cvMs.toFixed(1)} ms · quality ${Math.round(cvRes.q * 100)}%`;
  $('sC0').textContent = cvRes.valid ? cvRes.c0.toFixed(2) : '–'; $('sC1').textContent = cvRes.valid ? cvRes.c1.toFixed(3) : '–'; $('sC2').textContent = cvRes.valid ? (2 * cvRes.c2).toFixed(4) : '–';
}

/* ------------------------------------------------ neural net + charts */
function drawNN() {
  const c = $('nn'), g = c.getContext('2d'), net = ai.net, w = c.width, h = c.height; g.clearRect(0, 0, w, h);
  const names = ['off', 'hdg', 'curv', 'y_t', 'v', 'Δv set', 'lead', 'gap', 'closing', 'risk'];
  const xs = [50, 150, 250, 350], pos = net.sizes.map((n, l) => Array.from({ length: n }, (_, i) => [xs[l], 14 + (h - 28) * (i + 0.5) / n]));
  for (let l = 0; l < net.L; l++) {
    const W_ = net.W[l], ni = net.sizes[l], no = net.sizes[l + 1];
    for (let j = 0; j < no; j++) for (let i = 0; i < ni; i++) {
      const wv = W_[j * ni + i], a = Math.min(1, Math.abs(wv) * 1.2 * Math.abs(net.act[l][i]) + 0.02); if (a < 0.08) continue;
      g.strokeStyle = wv > 0 ? `rgba(57,229,140,${a * 0.5})` : `rgba(255,120,120,${a * 0.5})`; g.lineWidth = 0.6;
      g.beginPath(); g.moveTo(...pos[l][i]); g.lineTo(...pos[l + 1][j]); g.stroke();
    }
  }
  for (let l = 0; l <= net.L; l++) for (let i = 0; i < net.sizes[l]; i++) {
    const a = net.act[l][i], t = Math.min(1, Math.abs(a));
    g.fillStyle = a >= 0 ? `rgba(76,201,240,${0.15 + t * 0.85})` : `rgba(255,176,32,${0.15 + t * 0.85})`;
    g.beginPath(); g.arc(pos[l][i][0], pos[l][i][1], l === 0 || l === net.L ? 5 : 3.2, 0, 7); g.fill();
  }
  g.fillStyle = '#7d8ea1'; g.font = '9px system-ui'; g.textAlign = 'right'; names.forEach((n, i) => g.fillText(n, pos[0][i][0] - 8, pos[0][i][1] + 3));
  g.textAlign = 'left'; g.fillText('κ', pos[3][0][0] + 9, pos[3][0][1] + 3); g.fillText('accel', pos[3][1][0] + 9, pos[3][1][1] + 3);
  $('nnInfo').textContent = `${ai.steps} steps · ${ai.n} samples`;
  $('pmInfo').textContent = `cut-in BCE ${ai.pm.ema.toFixed(3)} · ${ai.pm.n} labels`;
}
function chart(id, series, opts) {
  const c = $(id), g = c.getContext('2d'), w = c.width, h = c.height; g.clearRect(0, 0, w, h);
  let lo = opts.lo, hi = opts.hi;
  if (lo === undefined) { lo = Infinity; hi = -Infinity; for (const s of series) for (const v of s.d) { if (v < lo) lo = v; if (v > hi) hi = v; } if (!isFinite(lo)) { lo = 0; hi = 1; } if (hi - lo < 1e-9) hi = lo + 1; }
  g.strokeStyle = '#1e2a36'; g.lineWidth = 1; for (let i = 0; i <= 4; i++) { const y = 4 + (h - 8) * i / 4; g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); }
  if (lo < 0 && hi > 0) { const y = 4 + (h - 8) * (1 - (0 - lo) / (hi - lo)); g.strokeStyle = '#36485a'; g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); }
  for (const s of series) {
    g.strokeStyle = s.c; g.lineWidth = 1.5; g.beginPath();
    s.d.forEach((v, i) => { const x = i / Math.max(1, s.d.length - 1) * w, y = 4 + (h - 8) * (1 - (v - lo) / (hi - lo)); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
  }
  g.fillStyle = '#7d8ea1'; g.font = '10px system-ui'; g.fillText(hi.toFixed(opts.dp || 1), 3, 12); g.fillText(lo.toFixed(opts.dp || 1), 3, h - 4);
  if (opts.title) { g.textAlign = 'right'; g.fillText(opts.title, w - 4, 12); g.textAlign = 'left'; }
}
function drawCharts() {
  const H = W.hist.slice(-400);
  chart('chSpd', [{ d: H.map(x => x.vs), c: '#ffb020' }, { d: H.map(x => x.v), c: '#4cc9f0' }], { title: 'speed km/h', dp: 0 });
  chart('chOff', [{ d: H.map(x => x.d), c: '#39e58c' }], { lo: -2, hi: 2, title: 'lateral offset m', dp: 1 });
  chart('chSteer', [{ d: H.map(x => x.exK), c: '#39e58c' }, { d: H.map(x => x.nnK), c: '#b48cff' }], { title: 'curvature 1/m', dp: 4 });
  chart('loss', [{ d: ai.lossHist.map(v => Math.log10(v + 1e-6)), c: '#ff7a45' }], { title: 'log10 MSE', dp: 1 });
}
function drawStats() {
  const s = W.stats, e = W.ego;
  const items = [['Distance', (s.dist / 1000).toFixed(2) + ' km'], ['Engaged', (s.engT / 60).toFixed(1) + ' min'], ['Min TTC', s.minTTC > 90 ? '–' : s.minTTC.toFixed(1) + ' s'], ['AEB events', s.aeb],
    ['FCW alerts', s.fcw], ['Lane changes', s.lc], ['Disengagements', s.dis], ['Collisions', s.crashes], ['Lane', ['right', 'middle', 'left'][e.lane]], ['Accel', e.a.toFixed(1) + ' m/s²'],
    ['Steer', (e.delta * 57.3).toFixed(1) + '°'], ['Jerk', s.jerkN ? (s.jerkSum / s.jerkN).toFixed(1) : '–']];
  $('stats').innerHTML = items.map(([k, v]) => `<div class="stat"><b>${v}</b><i>${k}</i></div>`).join('');
}
const NODES = [['camerad', 20], ['modeld', 20], ['radard', 20], ['plannerd', 20], ['controlsd', 50], ['card', 50]];
const nodeEls = {};
NODES.forEach(([n, hz], i) => { const d = document.createElement('div'); d.className = 'node'; d.innerHTML = `<b>${n}</b><i>${hz} Hz</i>`; $('pipe').appendChild(d); nodeEls[n] = d; if (i < NODES.length - 1) { const a = document.createElement('span'); a.className = 'arrow'; a.textContent = '→'; $('pipe').appendChild(a); } });
function flash(n, txt) { const d = nodeEls[n]; d.classList.add('flash'); if (txt) d.lastChild.textContent = txt; setTimeout(() => d.classList.remove('flash'), 90); }
function logMsg(t, lvl) { const l = $('log'), d = document.createElement('div'); d.className = lvl; d.textContent = t; l.appendChild(d); while (l.childNodes.length > 80) l.removeChild(l.firstChild); l.scrollTop = l.scrollHeight; }
let alertIdx = 0;

/* ------------------------------------------------ main loop */
let last = performance.now(), acc = 0, camT = 0;
function loop(now) {
  requestAnimationFrame(loop);
  const dtReal = Math.min(0.05, (now - last) / 1000); last = now;
  if (!W) return;
  W.input.steer = (keys.ArrowLeft ? 1 : 0) - (keys.ArrowRight ? 1 : 0); W.input.accel = keys.ArrowDown ? -1 : (keys.ArrowUp ? 1 : 0);
  if (!paused) {
    acc += dtReal * speed; let n = 0;
    while (acc >= 0.02 && n < 6) {
      W.step(0.02); acc -= 0.02; n++;
      if (W.perFlag) { W.perFlag = false; flash('camerad'); flash('modeld'); flash('radard'); flash('plannerd'); if (ai.trainOn) ai.trainSteps(1, 32, 8e-4); }
    }
    if (n === 6) acc = 0;
    flash('controlsd'); flash('card');
  }
  // render camera; run CV whenever the sim advanced
  CV.render(cam, W, Math.random);
  if (!paused && (pref.perception === 'cv' || frame % 3 === 0)) {
    const t0 = performance.now(); cvRes = CV.detectLanes(cam, W.P && W.P.tracks); cvMs = performance.now() - t0;
    if (pref.perception === 'cv') W.setCvLane(cvRes);
    nodeEls.modeld.lastChild.textContent = cvMs.toFixed(1) + ' ms CV';
  }
  if (pref.detector === 'yolo' && !yoloBusy && !paused) {
    yoloBusy = true; const t0 = performance.now(), tW = W.t;
    Yolo.detect(camC, 0.25).then(d => { yoloDets = d; yoloMs = performance.now() - t0; if (W.t - tW < 1.5) W.yolo = { dets: d, fresh: true, t: W.t }; })
      .catch(e => { $('yoloInfo').textContent = 'YOLO error: ' + e.message; }).finally(() => { yoloBusy = false; });
  }
  frame++;
  drawOverlay(); drawTop();
  if (frame % 3 === 0) { drawEdge(); drawNN(); drawCharts(); drawStats(); syncEng(); $('pipeInfo').textContent = `${W.tracks.size} tracks · ${W.cfg.weather} · t=${W.t.toFixed(0)}s`;
    if (pref.detector === 'yolo' && W.yoloStats) $('yoloInfo').textContent = `${yoloMs.toFixed(0)} ms/frame · ${W.yoloStats.det} boxes · ${W.yoloStats.matched} matched · ${W.yoloStats.fp} false pos · ${W.yoloStats.miss} missed`; }
  while (alertIdx < W.alerts.length) { const a = W.alerts[alertIdx++]; logMsg(`[${a.t.toFixed(1)}s] alert: ${a.msg}`, a.level); }
  if (W.alerts.length > 55 && alertIdx > 0) alertIdx = W.alerts.length;
  if (W.t - lastLog > 2 && !paused) { lastLog = W.t; if (W.P) logMsg(`[${W.t.toFixed(1)}s] longitudinalPlan a=${W.ego.aCmd.toFixed(2)} lead=${W.P.lead ? (W.P.lead.x - 2.3).toFixed(0) + 'm' : 'none'} risk=${W.P.risk.toFixed(2)} | lateral κ=${(W.ex ? W.ex.kap : 0).toFixed(4)}`, ''); }
}
/* ------------------------------------------------ real openpilot live view */
let opTimer = null, opState = null;
function drawOp() {
  const c = $('opc'), g = c.getContext('2d'), w = c.width, h = c.height, sc = 3.2, ox = w / 2, oy = h - 14;
  g.fillStyle = '#0a1016'; g.fillRect(0, 0, w, h);
  const T = (x, y) => [ox - y * sc, oy - x * sc];
  if (!opState || !opState.model) { g.fillStyle = '#7d8ea1'; g.font = '12px system-ui'; g.fillText('no data', 10, 20); return; }
  const line = (pts, col, wd) => { g.strokeStyle = col; g.lineWidth = wd; g.beginPath(); pts.x.forEach((x, i) => { const [u, v] = T(x, pts.y[i]); i ? g.lineTo(u, v) : g.moveTo(u, v); }); g.stroke(); };
  for (const e of opState.model.roadEdges) line(e, '#5a4630', 1.5);
  for (const l of opState.model.laneLines) line(l, '#c9d4df', 1.5);
  line(opState.model.path, '#39e58c', 4);
  for (const l of opState.model.leads) { const [u, v] = T(l.x, 0); g.fillStyle = '#ff7a45'; g.fillRect(u - 5, v - 5, 10, 10); g.fillText(`${l.x.toFixed(0)}m ${(l.prob * 100) | 0}%`, u + 8, v); }
  g.fillStyle = '#39e58c'; g.fillRect(ox - 4, oy - 8, 8, 14);
  const car = opState.car || {}, sd = opState.selfdrive || {};
  g.fillStyle = '#d6e0ea'; g.font = '12px system-ui';
  g.fillText(`${((car.vEgo || 0) * 3.6).toFixed(0)} km/h · steer ${(car.steeringAngleDeg || 0).toFixed(1)}° · ${sd.enabled ? 'ENGAGED' : 'disengaged'}`, 8, 16);
}
async function pollOp() {
  try { const r = await fetch($('opUrl').value); opState = await r.json(); $('opInfo').textContent = opState.alive ? 'connected · live' : 'connected · waiting for openpilot'; }
  catch (e) { opState = null; $('opInfo').textContent = 'unreachable (is bridge/run_sim.sh running in WSL?)'; }
  drawOp();
}
$('bOp').onclick = () => { if (opTimer) { clearInterval(opTimer); opTimer = null; $('bOp').textContent = 'Connect'; $('opInfo').textContent = 'not connected'; opState = null; drawOp(); } else { $('bOp').textContent = 'Disconnect'; pollOp(); opTimer = setInterval(pollOp, 100); } };
drawOp();
requestAnimationFrame(loop);
bootstrap().then(() => {
  $('tags').innerHTML = '<span>nominal highway</span>';
  const q = new URLSearchParams(location.search);   // e.g. ?cv=1&nn=1&scenario=night+cut-in
  if (q.get('cv')) $('cCV').click();
  if (q.get('nn')) $('pNN').click();
  if (q.get('yolo')) $('dYolo').click();
  if (q.get('scenario')) { $('prompt').value = q.get('scenario'); applyPrompt(q.get('scenario')); }
});
window.__ap = { get W() { return W; }, ai, pref };
})();
