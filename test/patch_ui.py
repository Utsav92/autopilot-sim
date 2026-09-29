import re
p = 'index.html'
s = open(p, encoding='utf-8').read()
s = s.replace('''    <div class="legend" style="margin-top:6px"><kbd>Space</kbd>''', '''    <div class="row" style="margin-top:8px">
      <label>Object detection</label><button id="dSim" class="on">Simulated</button><button id="dYolo">YOLOv8n (real, in-browser)</button><span class="sub" id="yoloInfo"></span>
    </div>
    <div class="legend" style="margin-top:6px"><kbd>Space</kbd>''')
s = s.replace('''<button id="bGen">Generate</button></div>''', '''<button id="bGen">Generate</button></div>
    <div class="legend" style="margin-top:4px" id="llmInfo">Checking for LLM server…</div>''')
s = s.replace('''  <div class="panel" style="margin-top:12px"><h2>Event log''', '''  <div class="panel" style="margin-top:12px"><h2>Real openpilot (WSL + MetaDrive) <span id="opInfo">not connected</span></h2>
    <div class="row" style="margin-bottom:6px"><input type="text" id="opUrl" value="http://localhost:8765/state" style="min-width:220px"><button id="bOp">Connect</button></div>
    <canvas class="f" id="opc" width="420" height="200"></canvas>
    <div class="legend" style="margin-top:6px">Live modelV2 path and lane lines, lead and carState streamed from openpilot itself. Setup steps are in the README (bridge section).</div></div>
  <div class="panel" style="margin-top:12px"><h2>Event log''')
s = s.replace('<script src="js/cv.js"></script>', '<script src="js/cv.js"></script>\n<script src="js/yolo.js"></script>')
open(p, 'w', encoding='utf-8').write(s)

p = 'js/app.js'
s = open(p, encoding='utf-8').read()
s = s.replace("const pref = { policy: 'expert', perception: 'gt', autoPass: false, setSpeed: 100 / 3.6 };",
  "const pref = { policy: 'expert', perception: 'gt', detector: 'sim', autoPass: false, setSpeed: 100 / 3.6 };\nlet yoloBusy = false, yoloDets = [], yoloMs = 0, llmOn = false;")
s = s.replace("$('bTrain').onclick", """$('dSim').onclick = () => { pref.detector = W.cfg.detector = 'sim'; W.tracks.clear(); setOn($('dSim'), $('dYolo'), true); $('yoloInfo').textContent = ''; };
$('dYolo').onclick = () => {
  pref.detector = W.cfg.detector = 'yolo'; W.tracks.clear(); W.yolo = null; setOn($('dSim'), $('dYolo'), false);
  Yolo.init(m => $('yoloInfo').textContent = m).catch(e => { $('dSim').click(); $('yoloInfo').textContent = 'YOLO failed: ' + e.message; });
};
$('bTrain').onclick""", 1)
s = s.replace("$('bGen').onclick = () => applyPrompt($('prompt').value);", """async function generate(text) {
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
  .catch(() => { $('llmInfo').innerHTML = 'LLM off (static server): run <kbd>node server.js</kbd> with <kbd>ANTHROPIC_API_KEY</kbd> for Claude-generated scenarios. Keyword grammar in use.'; });""")
s = s.replace("  // HUD\n", """  if (pref.detector === 'yolo') {
    ov.setLineDash([4, 3]); ov.strokeStyle = '#e879f9'; ov.lineWidth = 1.2; ov.fillStyle = '#e879f9'; ov.font = '10px system-ui';
    for (const d of yoloDets) { ov.strokeRect(d.x0, d.y0, d.w, d.h); ov.fillText(`YOLO ${d.cls} ${d.conf.toFixed(2)}`, d.x0, d.y0 - 2); }
    ov.setLineDash([]);
  }
  // HUD
""", 1)
s = s.replace("  frame++;\n  drawOverlay();", """  if (pref.detector === 'yolo' && !yoloBusy && !paused) {
    yoloBusy = true; const t0 = performance.now(), tW = W.t;
    Yolo.detect(camC, 0.25).then(d => { yoloDets = d; yoloMs = performance.now() - t0; if (W.t - tW < 1.5) W.yolo = { dets: d, fresh: true, t: W.t }; })
      .catch(e => { $('yoloInfo').textContent = 'YOLO error: ' + e.message; }).finally(() => { yoloBusy = false; });
  }
  frame++;
  drawOverlay();""", 1)
s = s.replace("$('pipeInfo').textContent = `${W.tracks.size} tracks · ${W.cfg.weather} · t=${W.t.toFixed(0)}s`;", """$('pipeInfo').textContent = `${W.tracks.size} tracks · ${W.cfg.weather} · t=${W.t.toFixed(0)}s`;
    if (pref.detector === 'yolo' && W.yoloStats) $('yoloInfo').textContent = `${yoloMs.toFixed(0)} ms/frame · ${W.yoloStats.det} boxes · ${W.yoloStats.matched} matched · ${W.yoloStats.fp} false pos · ${W.yoloStats.miss} missed`;""", 1)
s = s.replace("requestAnimationFrame(loop);\nbootstrap()", """/* ------------------------------------------------ real openpilot live view */
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
bootstrap()""", 1)
s = s.replace("if (q.get('nn')) $('pNN').click();", "if (q.get('nn')) $('pNN').click();\n  if (q.get('yolo')) $('dYolo').click();", 1)
open(p, 'w', encoding='utf-8').write(s)
print('patched')
