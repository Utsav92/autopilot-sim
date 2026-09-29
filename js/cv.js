/* Camera renderer (pseudo-3D dashcam view) + real computer-vision lane detection on the rendered pixels. */
(function (root) {
'use strict';
const { CAM, LW, ROAD_HALF } = root.Sim;
const WX = {
  clear: { sky: ['#4f9de6', '#cfe6fb'], grass: '#3d7a3c', road: '#4b4e53' },
  rain:  { sky: ['#55606a', '#9aa4ad'], grass: '#2f5236', road: '#30343a' },
  fog:   { sky: ['#c3c8cd', '#dfe2e5'], grass: '#7d8c7b', road: '#6b6e72' },
  night: { sky: ['#01020a', '#0a1228'], grass: '#07110a', road: '#25272c' },
};
const CARCOL = ['#b3202a', '#2a4a8c', '#2b2b30', '#7a7d82', '#1f6b4a', '#c9c9c6'];
const _p = { x: 0, y: 0 }, _q = { x: 0, y: 0 };
const proj = (x, y, z) => [CAM.cx - CAM.f * y / x, CAM.vh + CAM.f * (CAM.h - z) / x];

function poly(ctx, pts, fill) {
  ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
}

function render(ctx, W, rndFrame) {
  const R = W.road, pose = W.pose, e = W.ego, wx = WX[W.cfg.weather] || WX.clear, weather = W.cfg.weather;
  const { W: CW, H: CH, vh } = CAM;
  let g = ctx.createLinearGradient(0, 0, 0, vh); g.addColorStop(0, wx.sky[0]); g.addColorStop(1, wx.sky[1]);
  ctx.fillStyle = g; ctx.fillRect(0, 0, CW, vh + 1);
  // distant hills parallax on global heading
  ctx.fillStyle = weather === 'night' ? '#050b12' : (weather === 'fog' ? '#a9b2ab' : '#4d7a5a');
  ctx.beginPath(); ctx.moveTo(0, vh);
  for (let u = 0; u <= CW; u += 8) {
    const ang = pose.T + Math.atan((CAM.cx - u) / CAM.f);
    ctx.lineTo(u, vh - (14 + 9 * Math.sin(ang * 7) + 6 * Math.sin(ang * 19 + 1) + 4 * Math.sin(ang * 43)));
  }
  ctx.lineTo(CW, vh); ctx.fill();
  ctx.fillStyle = wx.grass; ctx.fillRect(0, vh, CW, CH - vh);
  // road surface
  const step = x => x < 60 ? 1 : 2.5, s0 = e.s;
  const verts = [];
  for (let ds = 0, k = 0; ds < 230; ds += step(ds), k++) {
    R.toEgo(pose, s0 + ds, ROAD_HALF + 0.5, _p); R.toEgo(pose, s0 + ds, -ROAD_HALF - 0.5, _q);
    verts.push({ ds, lx: _p.x, ly: _p.y, rx: _q.x, ry: _q.y });
  }
  for (let i = 0; i + 1 < verts.length; i++) {
    const a = verts[i], b = verts[i + 1];
    if (a.lx < 1 || a.rx < 1 || b.lx < 1 || b.rx < 1) continue;
    const A = proj(a.lx, a.ly, 0), B = proj(a.rx, a.ry, 0), C = proj(b.rx, b.ry, 0), D = proj(b.lx, b.ly, 0);
    poly(ctx, [A, B, C, D], wx.road);
  }
  // lane markings
  const lines = [{ d: ROAD_HALF, c: '#e6b800', dash: false }, { d: LW / 2, c: '#ececE6', dash: true, ph: 0 }, { d: -LW / 2, c: '#ececE6', dash: true, ph: 6 }, { d: -ROAD_HALF, c: '#ececE6', dash: false }];
  for (const L of lines) {
    for (let ds = 0; ds < 200; ds += step(ds)) {
      const sa = s0 + ds, sb = s0 + ds + step(ds);
      if (L.dash && (((sa + L.ph) % 12) + 12) % 12 > 8) continue;
      R.toEgo(pose, sa, L.d + 0.08, _p); const a1x = _p.x, a1y = _p.y;
      R.toEgo(pose, sa, L.d - 0.08, _p); const a2x = _p.x, a2y = _p.y;
      R.toEgo(pose, sb, L.d - 0.08, _p); const b2x = _p.x, b2y = _p.y;
      R.toEgo(pose, sb, L.d + 0.08, _p); const b1x = _p.x, b1y = _p.y;
      if (Math.min(a1x, a2x, b1x, b2x) < 1) continue;
      poly(ctx, [proj(a1x, a1y, 0), proj(a2x, a2y, 0), proj(b2x, b2y, 0), proj(b1x, b1y, 0)], L.c);
    }
  }
  // objects far -> near
  const items = [];
  for (const o of W.objs) {
    R.toEgo(pose, o.s - o.len / 2, o.d, _p); const rx = _p.x, ry = _p.y;
    R.toEgo(pose, o.s + o.len / 2, o.d, _p);
    if (rx < 3 || rx > 260) continue;
    items.push({ o, rx, ry, fx: _p.x, fy: _p.y });
  }
  items.sort((a, b) => b.rx - a.rx);
  for (const it of items) {
    const o = it.o, { rx, ry, fx, fy } = it;
    if (o.type === 'ped') { drawPed(ctx, o, rx, ry); continue; }
    const dx = fx - rx, dy = fy - ry, L = Math.hypot(dx, dy) || 1, nx = -dy / L, ny = dx / L, hw = o.wid / 2;
    const RL = [rx + nx * hw, ry + ny * hw], RR = [rx - nx * hw, ry - ny * hw];
    const FL = [fx + nx * hw, fy + ny * hw], FR = [fx - nx * hw, fy - ny * hw];
    drawVehicle(ctx, o, RL, RR, FL, FR, nx, ny, ry, weather);
  }
  weatherFx(ctx, weather, rndFrame);
}

function grad(ctx, x0, y0, x1, y1, c0, c1) { const g = ctx.createLinearGradient(x0, y0, x1, y1); g.addColorStop(0, c0); g.addColorStop(1, c1); return g; }
function drawPed(ctx, o, rx, ry) {
  const [u, v0] = proj(rx, ry, 0), [, v1] = proj(rx, ry, o.hgt), H = v0 - v1, w = Math.max(3, H * 0.28);
  const sk = '#e2b98f', shirt = ['#c0392b', '#2c6fbb', '#3a3a3a', '#d9a520'][o.color % 4], pants = '#252a3a';
  ctx.fillStyle = 'rgba(0,0,0,0.3)'; ctx.beginPath(); ctx.ellipse(u, v0, w * 0.7, Math.max(1, H * 0.03), 0, 0, 7); ctx.fill();
  const leg = (dx) => { ctx.fillStyle = pants; ctx.fillRect(u + dx - w * 0.16, v1 + H * 0.5, w * 0.32, H * 0.5); };
  leg(-w * 0.18); leg(w * 0.18);
  ctx.fillStyle = shirt; ctx.fillRect(u - w * 0.42, v1 + H * 0.18, w * 0.84, H * 0.36);
  ctx.fillRect(u - w * 0.62, v1 + H * 0.2, w * 0.2, H * 0.32); ctx.fillRect(u + w * 0.42, v1 + H * 0.2, w * 0.2, H * 0.32);
  ctx.fillStyle = sk; ctx.beginPath(); ctx.ellipse(u, v1 + H * 0.09, w * 0.24, H * 0.085, 0, 0, 7); ctx.fill();
  ctx.fillStyle = '#3b2a20'; ctx.beginPath(); ctx.ellipse(u, v1 + H * 0.06, w * 0.25, H * 0.06, 0, Math.PI, 0); ctx.fill();
}
function drawVehicle(ctx, o, RL, RR, FL, FR, nx, ny, ry, weather) {
  const truck = o.type === 'truck', hw = o.wid / 2, cx = (RL[0] + RR[0]) / 2, cy = (RL[1] + RR[1]) / 2, x = cx;
  const P = (t, z) => proj(cx + nx * hw * t, cy + ny * hw * t, z);   // point on rear face: t in [-1,1] across, z height
  const body = truck ? '#dfe1e4' : CARCOL[o.color % 6], dark = shade(body, 0.55);
  // ground shadow
  const [su, sv] = proj(x, cy, 0), sw = CAM.f * o.wid / x;
  ctx.fillStyle = 'rgba(0,0,0,0.45)'; ctx.beginPath(); ctx.ellipse(su, sv, sw * 0.6, Math.max(1.5, sw * 0.05), 0, 0, 7); ctx.fill();
  // visible side face
  if (Math.min(FL[0], FR[0]) > 1.5) {
    const S = ry > 0 ? [RR, FR] : [RL, FL], z1 = o.hgt * (truck ? 1 : 0.62);
    const a = proj(S[0][0], S[0][1], 0.3), b = proj(S[1][0], S[1][1], 0.3), c = proj(S[1][0], S[1][1], z1), d = proj(S[0][0], S[0][1], z1);
    poly(ctx, [a, b, c, d], shade(body, 0.72));
    if (!truck) {   // side windows
      const w0 = proj(S[0][0] * 0.85 + S[1][0] * 0.15, S[0][1] * 0.85 + S[1][1] * 0.15, 0.95), w1 = proj(S[0][0] * 0.35 + S[1][0] * 0.65, S[0][1] * 0.35 + S[1][1] * 0.65, 0.95);
      const w2 = proj(S[0][0] * 0.35 + S[1][0] * 0.65, S[0][1] * 0.35 + S[1][1] * 0.65, 1.38), w3 = proj(S[0][0] * 0.85 + S[1][0] * 0.15, S[0][1] * 0.85 + S[1][1] * 0.15, 1.38);
      poly(ctx, [w0, w1, w2, w3], 'rgba(20,28,38,0.85)');
    }
  }
  if (truck) {
    const top = o.hgt;
    let [x0, y0] = P(-1, 0.55), [x1, y1] = P(1, top);
    ctx.fillStyle = grad(ctx, 0, y1, 0, y0, '#eceef0', '#c7cacd'); ctx.fillRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y0 - y1));
    ctx.strokeStyle = 'rgba(0,0,0,0.25)'; ctx.lineWidth = 1; ctx.beginPath(); const m = P(0, 0.55), m2 = P(0, top); ctx.moveTo(m[0], m[1]); ctx.lineTo(m2[0], m2[1]); ctx.stroke();
    const u0 = P(-1, 0.3), u1 = P(1, 0.6); ctx.fillStyle = '#25262a'; ctx.fillRect(Math.min(u0[0], u1[0]), Math.min(u0[1], u1[1]), Math.abs(u1[0] - u0[0]), Math.abs(u0[1] - u1[1]));
    ctx.fillStyle = '#1a1a1c'; for (const t of [-0.8, 0.8]) { const a = P(t - 0.18, 0), b = P(t + 0.18, 0.5); ctx.fillRect(Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]), Math.abs(a[1] - b[1])); }
  } else {
    // wheels
    ctx.fillStyle = '#111'; for (const t of [-0.78, 0.78]) { const a = P(t - 0.2, 0), b = P(t + 0.2, 0.42); ctx.fillRect(Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]), Math.abs(a[1] - b[1])); }
    // lower body + trunk
    let a = P(-1, 0.28), b = P(1, 1.0);
    ctx.fillStyle = grad(ctx, 0, b[1], 0, a[1], shade(body, 1.1), shade(body, 0.8));
    poly(ctx, [P(-1, 0.28), P(1, 0.28), P(1, 0.98), P(-1, 0.98)], ctx.fillStyle);
    // cabin: rear window (trapezoid) + roof
    poly(ctx, [P(-0.86, 0.98), P(0.86, 0.98), P(0.66, 1.42), P(-0.66, 1.42)], grad(ctx, 0, P(0, 1.42)[1], 0, P(0, 0.98)[1], 'rgba(40,52,66,0.95)', 'rgba(14,18,26,0.95)'));
    poly(ctx, [P(-0.7, 1.42), P(0.7, 1.42), P(0.66, 1.5), P(-0.66, 1.5)], shade(body, 0.95));
    // bumper + plate
    poly(ctx, [P(-1, 0.28), P(1, 0.28), P(1, 0.42), P(-1, 0.42)], '#1c1d20');
    const pa = P(-0.24, 0.5), pb = P(0.24, 0.62); ctx.fillStyle = '#e8e8e0'; ctx.fillRect(Math.min(pa[0], pb[0]), Math.min(pa[1], pb[1]), Math.abs(pb[0] - pa[0]), Math.abs(pa[1] - pb[1]));
  }
  // tail lights (brake lights brighter)
  const braking = o.a < -1 || o.static, lz0 = truck ? 0.7 : 0.78, lz1 = truck ? 0.95 : 0.92;
  for (const [t0, t1] of [[-1, -0.62], [0.62, 1]]) {
    const a = P(t0, lz0), b = P(t1, lz1);
    ctx.fillStyle = braking ? '#ff2a2a' : '#8f1418'; ctx.fillRect(Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]), Math.max(1.5, Math.abs(a[1] - b[1])));
    if (weather === 'night' || braking) { ctx.fillStyle = 'rgba(255,40,40,0.22)'; ctx.fillRect(Math.min(a[0], b[0]) - 3, Math.min(a[1], b[1]) - 3, Math.abs(b[0] - a[0]) + 6, Math.abs(a[1] - b[1]) + 6); }
  }
}

function shade(hex, f) {
  const n = parseInt(hex.slice(1), 16);
  return `rgb(${(n >> 16 & 255) * f | 0},${(n >> 8 & 255) * f | 0},${(n & 255) * f | 0})`;
}

function weatherFx(ctx, weather, rnd) {
  const { W: CW, H: CH, vh } = CAM;
  if (weather === 'night') {
    const g = ctx.createLinearGradient(0, vh, 0, CH);
    g.addColorStop(0, 'rgba(0,0,10,0.78)'); g.addColorStop(0.3, 'rgba(0,0,10,0.6)'); g.addColorStop(1, 'rgba(0,0,10,0.32)');
    ctx.fillStyle = g; ctx.fillRect(0, vh, CW, CH - vh);
  } else if (weather === 'fog') {
    const g = ctx.createLinearGradient(0, vh - 20, 0, CH);
    g.addColorStop(0, 'rgba(208,213,218,0.97)'); g.addColorStop(0.12, 'rgba(208,213,218,0.85)'); g.addColorStop(0.3, 'rgba(208,213,218,0.55)');
    g.addColorStop(0.6, 'rgba(208,213,218,0.2)'); g.addColorStop(1, 'rgba(208,213,218,0.05)');
    ctx.fillStyle = g; ctx.fillRect(0, vh - 20, CW, CH - vh + 20);
  } else if (weather === 'rain') {
    ctx.fillStyle = 'rgba(30,40,55,0.18)'; ctx.fillRect(0, 0, CW, CH);
    ctx.strokeStyle = 'rgba(205,215,228,0.4)'; ctx.lineWidth = 1; ctx.beginPath();
    for (let i = 0; i < 160; i++) { const x = rnd() * CW, y = rnd() * CH; ctx.moveTo(x, y); ctx.lineTo(x - 3, y + 13); }
    ctx.stroke();
  }
}

/* ---------------- computer vision: lane detection from pixels ---------------- */
function lsq(pts) {   // least squares y = a + b x + c x^2
  const s = new Float64Array(5), t = new Float64Array(3);
  for (const [x, y] of pts) { let xp = 1; for (let k = 0; k < 5; k++) { s[k] += xp; if (k < 3) t[k] += y * xp; xp *= x; } }
  const M = [[s[0], s[1], s[2], t[0]], [s[1], s[2], s[3], t[1]], [s[2], s[3], s[4], t[2]]];
  for (let i = 0; i < 3; i++) {
    let p = i; for (let r = i + 1; r < 3; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
    [M[i], M[p]] = [M[p], M[i]]; if (Math.abs(M[i][i]) < 1e-12) return null;
    for (let r = 0; r < 3; r++) if (r !== i) { const f = M[r][i] / M[i][i]; for (let c = i; c < 4; c++) M[r][c] -= f * M[i][c]; }
  }
  return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
}
function fitQuad(pts) {   // RANSAC (random 3-point hypotheses) then least-squares refit on the inliers
  if (pts.length < 6) return null;
  let best = null, bestN = 0; const TH = 0.22;
  for (let it = 0; it < 60; it++) {
    const a = pts[Math.floor(Math.random() * pts.length)], b = pts[Math.floor(Math.random() * pts.length)], c = pts[Math.floor(Math.random() * pts.length)];
    if (a[0] === b[0] || b[0] === c[0] || a[0] === c[0]) continue;
    const m = lsq([a, b, c]); if (!m || Math.abs(m[2]) > 0.02) continue;
    let n = 0; for (const [x, y] of pts) if (Math.abs(y - (m[0] + m[1] * x + m[2] * x * x)) < TH) n++;
    if (n > bestN) { bestN = n; best = m; }
  }
  if (!best || bestN < 6) return null;
  const inl = pts.filter(([x, y]) => Math.abs(y - (best[0] + best[1] * x + best[2] * x * x)) < TH);
  const c = lsq(inl); return c && { c, n: inl.length };
}

let _masks = [], _prior = [0, 0, 0];
function detectLanes(ctx, tracks) {
  _masks = (tracks || []).filter(t => t.x > 3 && t.miss < 4).map(t => {
    const [u, v0] = proj(t.x, t.y, 0), [, v1] = proj(t.x, t.y, t.hgt), hw = CAM.f * t.wid / t.x / 2 + 12;
    return [u - hw, u + hw, v1 - 8, v0 + 8];
  });
  const img = ctx.getImageData(0, 0, CAM.W, CAM.H);
  let r = detectPass(img);   // iterate: each pass re-centres the search windows on the latest fit
  for (let i = 0; i < 2 && r.c0 !== undefined && r.fitL; i++) { _prior = [r.c0, r.c1, r.c2]; r = detectPass(img); }
  return r;
}
function detectPass(img) {
  const d = img.data, Wd = CAM.W;
  const Lp = [], Rp = [], cand = []; let rows = 0;
  const hist = new Uint16Array(256);
  for (let x = 6; x <= 46; x += 2) {
    const v = Math.round(CAM.vh + CAM.f * CAM.h / x); if (v >= CAM.H) continue; rows++;
    const base = v * Wd * 4; hist.fill(0);
    for (let u = 60; u < 580; u += 2) { const i = base + u * 4; hist[(d[i] * 0.3 + d[i + 1] * 0.59 + d[i + 2] * 0.11) | 0]++; }
    let acc = 0, med = 0; for (; med < 256; med++) { acc += hist[med]; if (acc >= 130) break; }
    const thr = med + 42, expW = CAM.f * 0.16 / x;
    let runS = -1, bestL = null, bestR = null;
    for (let u = 0; u <= Wd; u++) {
      let on = false;
      if (u < Wd) { const i = base + u * 4, r = d[i], g = d[i + 1], b = d[i + 2]; on = (r * 0.3 + g * 0.59 + b * 0.11) > thr && r < 1.35 * g + 20; }
      if (on && runS < 0) runS = u;
      else if (!on && runS >= 0) {
        const w = u - runS, c = (runS + u - 1) / 2; runS = -1;
        if (w < 0.4 * expW || w > 3 * expW + 3) continue;
        if (_masks.some(m => c > m[0] && c < m[1] && v > m[2] && v < m[3])) continue;
        cand.push([c, v]);
        // lateral position relative to the tracked lane centre (prior from the previous frame handles curvature)
        const rel = (CAM.cx - c) * x / CAM.f - (_prior[1] * x + _prior[2] * x * x);   // remove predicted heading/curvature, keep ego-anchored offset
        if (rel > -0.3 && rel < LW + 0.2) { if (bestL === null || rel < bestL.rel) bestL = { c, rel }; }
        else if (rel < 0.3 && rel > -LW - 0.2) { if (bestR === null || rel > bestR.rel) bestR = { c, rel }; }
      }
    }
    if (bestL !== null) Lp.push([x, (CAM.cx - bestL.c) * x / CAM.f, bestL.c, v]);
    if (bestR !== null) Rp.push([x, (CAM.cx - bestR.c) * x / CAM.f, bestR.c, v]);
  }
  // pool both markings into one lane-centre polynomial (lane width from map prior)
  const pool = Lp.map(p => [p[0], p[1] - LW / 2]).concat(Rp.map(p => [p[0], p[1] + LW / 2]));
  const f = fitQuad(pool), c = f && f.c;
  const q = pool.length / (2 * Math.max(rows, 1));
  const valid = !!c && q > 0.25;
  if (valid) _prior = c;
  const off = s => c && [c[0] + s, c[1], c[2]];
  return {
    valid, c0: c ? c[0] : 0, c1: c ? c[1] : 0, c2: c ? c[2] : 0, q,
    fitL: off(LW / 2), fitR: off(-LW / 2), Lp, Rp, cand, img,
  };
}

/* Sobel edge map at half resolution for the "what the CV stack sees" panel */
function edgeMap(img, out) {
  const w = CAM.W / 2, h = CAM.H / 2, d = img.data, gray = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const i = ((y * 2) * CAM.W + x * 2) * 4; gray[y * w + x] = d[i] * 0.3 + d[i + 1] * 0.59 + d[i + 2] * 0.11; }
  const o = out.createImageData(w, h), od = o.data;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    const gx = -gray[i - w - 1] - 2 * gray[i - 1] - gray[i + w - 1] + gray[i - w + 1] + 2 * gray[i + 1] + gray[i + w + 1];
    const gy = -gray[i - w - 1] - 2 * gray[i - w] - gray[i - w + 1] + gray[i + w - 1] + 2 * gray[i + w] + gray[i + w + 1];
    const m = Math.min(255, Math.sqrt(gx * gx + gy * gy) * 0.5), j = i * 4;
    od[j] = m * 0.35; od[j + 1] = m * 0.9; od[j + 2] = m; od[j + 3] = 255;
  }
  out.putImageData(o, 0, 0);
}

root.CV = { render, detectLanes, edgeMap, proj };
})(window);
