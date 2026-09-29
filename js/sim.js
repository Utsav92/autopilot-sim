/* AutoPilot-Sim core: road, traffic, ego vehicle, perception tracker, planner/controller,
 * neural policy (behaviour cloning), online cut-in classifier and generative future sampler.
 * Pure logic (no DOM) so it can run headless in Node for tests and for fast NN pre-training.
 * Architecture mirrors openpilot: camerad -> modeld -> plannerd -> controlsd -> card.
 */
(function (root) {
'use strict';

const LW = 3.7, NLANES = 3, ROAD_HALF = LW * NLANES / 2, WB = 2.8;
const CAM = { W: 640, H: 360, f: 700, vh: 170, h: 1.3, cx: 320 };
const DIM = {
  car:   { len: 4.6, wid: 1.9, hgt: 1.5 },
  truck: { len: 12,  wid: 2.5, hgt: 3.6 },
  ped:   { len: 0.5, wid: 0.5, hgt: 1.75 },
};
const NFEAT = 10;
const clamp = (x, a, b) => x < a ? a : (x > b ? b : x);
const sgn = x => x < 0 ? -1 : 1;

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

/* ---------------------------------------------------------------- Road */
const _g = { X: 0, Y: 0 };
class Road {
  constructor(scale) { this.N = 60000; this.build(scale || 1); }
  build(scale) {
    const N = this.N;
    this.X = new Float32Array(N + 2); this.Y = new Float32Array(N + 2);
    this.TH = new Float32Array(N + 2); this.K = new Float32Array(N + 2);
    let x = 0, y = 0, th = 0;
    for (let i = 0; i <= N + 1; i++) {
      const env = 0.3 + 0.7 * (0.5 + 0.5 * Math.sin(i / 1100 + 2));
      const k = scale * env * (0.0030 * Math.sin(i / 420) + 0.0022 * Math.sin(i / 190 + 1.3) + 0.0008 * Math.sin(i / 75));
      this.K[i] = k; this.X[i] = x; this.Y[i] = y; this.TH[i] = th;
      th += k; x += Math.cos(th); y += Math.sin(th);
    }
    this._f = 0;
  }
  _i(s) { s = s < 0 ? 0 : (s > this.N ? this.N : s); const i = Math.floor(s); this._f = s - i; return i; }
  th(s) { const i = this._i(s); return this.TH[i] + (this.TH[i + 1] - this.TH[i]) * this._f; }
  k(s) { const i = this._i(s); return this.K[i] + (this.K[i + 1] - this.K[i]) * this._f; }
  glob(s, d, out) {
    const i = this._i(s), f = this._f;
    const x = this.X[i] + (this.X[i + 1] - this.X[i]) * f;
    const y = this.Y[i] + (this.Y[i + 1] - this.Y[i]) * f;
    const th = this.TH[i] + (this.TH[i + 1] - this.TH[i]) * f;
    out.X = x - d * Math.sin(th); out.Y = y + d * Math.cos(th);
    return out;
  }
  toEgo(pose, s, d, out) {
    this.glob(s, d, _g);
    const dx = _g.X - pose.X, dy = _g.Y - pose.Y;
    out.x = dx * pose.c + dy * pose.sn; out.y = -dx * pose.sn + dy * pose.c;
    return out;
  }
}

/* ------------------------------------------------- Neural policy (MLP) */
class MLP {
  constructor(sizes, rnd) {
    this.sizes = sizes; this.L = sizes.length - 1; this.t = 0;
    this.W = []; this.b = []; this.mW = []; this.vW = []; this.mb = []; this.vb = [];
    this.gW = []; this.gb = [];
    for (let l = 0; l < this.L; l++) {
      const ni = sizes[l], no = sizes[l + 1], lim = Math.sqrt(6 / (ni + no));
      const w = new Float32Array(ni * no);
      for (let i = 0; i < w.length; i++) w[i] = (rnd() * 2 - 1) * lim;
      this.W.push(w); this.b.push(new Float32Array(no));
      this.mW.push(new Float32Array(ni * no)); this.vW.push(new Float32Array(ni * no));
      this.mb.push(new Float32Array(no)); this.vb.push(new Float32Array(no));
      this.gW.push(new Float32Array(ni * no)); this.gb.push(new Float32Array(no));
    }
    this.act = sizes.map(n => new Float32Array(n));   // live activations (for visualisation)
    this.tact = sizes.map(n => new Float32Array(n));  // scratch for training
    this.delta = sizes.map(n => new Float32Array(n));
  }
  forward(x, act) {
    act = act || this.act; act[0].set(x);
    for (let l = 0; l < this.L; l++) {
      const ai = act[l], ao = act[l + 1], W = this.W[l], b = this.b[l];
      const ni = this.sizes[l], no = this.sizes[l + 1], last = l === this.L - 1;
      for (let j = 0; j < no; j++) {
        let s = b[j]; const off = j * ni;
        for (let i = 0; i < ni; i++) s += W[off + i] * ai[i];
        ao[j] = last ? s : Math.tanh(s);
      }
    }
    return act[this.L];
  }
  train(X, Y, idx, lr) {
    const B = idx.length, L = this.L, sz = this.sizes, nout = sz[L];
    for (let l = 0; l < L; l++) { this.gW[l].fill(0); this.gb[l].fill(0); }
    let loss = 0;
    for (let n = 0; n < B; n++) {
      const k = idx[n];
      const out = this.forward(X.subarray(k * NFEAT, (k + 1) * NFEAT), this.tact);
      const dl = this.delta[L];
      for (let j = 0; j < nout; j++) { const e = out[j] - Y[k * nout + j]; loss += e * e; dl[j] = 2 * e / (B * nout); }
      for (let l = L - 1; l >= 0; l--) {
        const ni = sz[l], no = sz[l + 1], din = this.delta[l], dout = this.delta[l + 1], ain = this.tact[l];
        if (l > 0) din.fill(0);
        for (let j = 0; j < no; j++) {
          const g = dout[j]; if (g === 0) continue;
          this.gb[l][j] += g; const off = j * ni;
          for (let i = 0; i < ni; i++) {
            this.gW[l][off + i] += g * ain[i];
            if (l > 0) din[i] += g * this.W[l][off + i];
          }
        }
        if (l > 0) for (let i = 0; i < ni; i++) din[i] *= (1 - ain[i] * ain[i]);
      }
    }
    this.t++;
    const b1 = 0.9, b2 = 0.999, c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    for (let l = 0; l < L; l++) {
      const upd = (p, g, m, v) => {
        for (let i = 0; i < p.length; i++) {
          m[i] = b1 * m[i] + (1 - b1) * g[i]; v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
          p[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8);
        }
      };
      upd(this.W[l], this.gW[l], this.mW[l], this.vW[l]);
      upd(this.b[l], this.gb[l], this.mb[l], this.vb[l]);
    }
    return loss / (B * nout);
  }
}

/* Shared learning state (survives scenario resets): policy net, replay buffer, cut-in classifier */
function makeAI(seed) {
  const rnd = mulberry32(seed || 7), CAP = 30000;
  return {
    rnd, net: new MLP([NFEAT, 32, 32, 2], rnd),
    X: new Float32Array(CAP * NFEAT), Y: new Float32Array(CAP * 2), n: 0, head: 0, CAP,
    lossHist: [], steps: 0, lastLoss: NaN, trainOn: true, lr: 2e-3,
    pm: { w: [-2.5, 0, 0, 0, 0], n: 0, ema: 0.5, hist: [] },
    push(f, y) {
      const k = this.head; this.X.set(f, k * NFEAT); this.Y[k * 2] = y[0]; this.Y[k * 2 + 1] = y[1];
      this.head = (k + 1) % this.CAP; this.n = Math.min(this.n + 1, this.CAP);
    },
    trainSteps(n, batch, lr) {
      if (this.n < 64) return;
      const idx = new Int32Array(batch || 64); let loss = 0;
      for (let s = 0; s < n; s++) {
        for (let i = 0; i < idx.length; i++) idx[i] = Math.floor(this.rnd() * this.n);
        loss = this.net.train(this.X, this.Y, idx, lr || this.lr); this.steps++;
      }
      this.lastLoss = this.lastLoss !== this.lastLoss ? loss : this.lastLoss * 0.9 + loss * 0.1;
      this.lossHist.push(this.lastLoss); if (this.lossHist.length > 300) this.lossHist.shift();
    },
    pmProb(f) { let z = 0; for (let i = 0; i < 5; i++) z += this.pm.w[i] * f[i]; return 1 / (1 + Math.exp(-z)); },
    pmTrain(f, y) {
      const p = this.pmProb(f), g = p - y, w = this.pm.w;
      for (let i = 0; i < 5; i++) w[i] -= 0.08 * (g * f[i] + 0.002 * w[i]);
      this.pm.n++; this.pm.ema = this.pm.ema * 0.98 + 0.02 * (-(y * Math.log(p + 1e-6) + (1 - y) * Math.log(1 - p + 1e-6)));
    },
  };
}

/* --------------------------------------------------------------- World */
class World {
  constructor(ai, cfg) { this.ai = ai || makeAI(); this.cfg = {}; this.reset(cfg || {}); }

  reset(cfg) {
    const d = {
      seed: 1, weather: 'clear', density: 10, curve: 1, events: {}, setSpeed: 100 / 3.6,
      perception: 'gt', detector: 'sim', policy: 'expert', autoPass: false, collect: true, noise: false,
    };
    this.cfg = Object.assign({}, d, cfg);
    this.rnd = mulberry32(this.cfg.seed * 7919 + 1);
    this.road = new Road(this.cfg.curve);
    this.t = 0; this.crashed = null; this.nextId = 1; this.objs = [];
    this.ego = {
      s: 300, d: 0, psi: 0, v: Math.min(this.cfg.setSpeed, 27), delta: 0, a: 0, aCmd: 0, lane: 1, tgtLane: 1,
      shiftEff: 0, engaged: true, len: 4.6, wid: 1.9, setSpeed: this.cfg.setSpeed, aPlan: 0, ttc: 99,
      lcCool: 0, override: 0,
    };
    this.input = { steer: 0, accel: 0 };
    this.pose = { X: 0, Y: 0, T: 0, c: 1, sn: 0 }; this.updatePose();
    this.P = null; this.cvLane = null; this.lastLane = null; this.laneLost = 0; this.perTimer = 1; this.lblT = 0;
    this.trafT = 0; this.autoT = 0; this.tracks = new Map(); this.fut = []; this.alerts = []; this.curAlert = null;
    this.evt = {};
    const ev = this.cfg.events, base = { cutin: 5, hardbrake: 9, stopped: 16, ped: 12 };
    for (const k in ev) if (ev[k]) this.evt[k] = { next: base[k] };
    this.stats = { dist: 0, minTTC: 99, aeb: 0, lc: 0, dis: 0, crashes: 0, jerkSum: 0, jerkN: 0, engT: 0, fcw: 0 };
    this.noiseState = { ou: 0, nextDist: 6 };
    this.hist = []; this.histT = 0;
    this.initTraffic();
    this.percTick(0.05);
  }

  gauss() { const u = 1 - this.rnd(), v = this.rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  updatePose() {
    const e = this.ego; this.road.glob(e.s, e.d, _g);
    const T = this.road.th(e.s) + e.psi;
    this.pose.X = _g.X; this.pose.Y = _g.Y; this.pose.T = T; this.pose.c = Math.cos(T); this.pose.sn = Math.sin(T);
  }
  alert(msg, level) {
    this.curAlert = { msg, level: level || 'info', until: this.t + 3 };
    this.alerts.push({ t: this.t, msg, level: level || 'info' }); if (this.alerts.length > 60) this.alerts.shift();
  }

  /* ---------------- traffic */
  spawn(type, s, d, v, extra) {
    const dm = DIM[type];
    const o = Object.assign({
      id: this.nextId++, type, s, d, v, vd: 0, a: 0, len: dm.len, wid: dm.wid, hgt: dm.hgt,
      vdes: v, lane: clamp(Math.round(d / LW) + 1, 0, 2), color: Math.floor(this.rnd() * 6),
    }, extra || {});
    this.objs.push(o); return o;
  }
  laneOcc(lane, s0, s1) { const c = (lane - 1) * LW; return this.objs.filter(o => o.s > s0 && o.s < s1 && Math.abs(o.d - c) < 2.2); }
  initTraffic() {
    const e = this.ego, n = Math.round(this.cfg.density * 0.45);
    for (let i = 0; i < n * 3; i++) {
      if (this.objs.length >= n) break;
      this.trySpawn(e.s - 120 + this.rnd() * 420, 30);
    }
  }
  trySpawn(s, minGap) {
    const lane = Math.floor(this.rnd() * 3);
    if (this.laneOcc(lane, s - minGap, s + minGap).length) return null;
    const e = this.ego; if (Math.abs(lane - e.lane) < 1 && Math.abs(s - e.s) < 45) return null;
    const truck = this.rnd() < 0.15;
    const vd = truck ? 19 + this.rnd() * 3 : 20 + lane * 2.2 + this.rnd() * 3.5;
    return this.spawn(truck ? 'truck' : 'car', s, (lane - 1) * LW, vd, { vdes: vd });
  }
  manageTraffic() {
    const e = this.ego, ntgt = Math.round(this.cfg.density * 0.28), nb = Math.round(this.cfg.density * 0.13);
    this.objs = this.objs.filter(o => o.s > e.s - 220 && o.s < e.s + 420);
    const cars = this.objs.filter(o => o.type !== 'ped');
    const ahead = cars.filter(o => o.s > e.s && o.s < e.s + 300).length;
    const behind = cars.filter(o => o.s < e.s && o.s > e.s - 150).length;
    if (ahead < ntgt) this.trySpawn(e.s + 270 + this.rnd() * 40, 35);
    if (behind < nb) {
      const o = this.trySpawn(e.s - 130 - this.rnd() * 30, 40);
      if (o) { o.v = Math.max(o.v, e.v - 1); }
    }
  }

  events() {
    const e = this.ego, t = this.t;
    for (const k in this.evt) {
      const ev = this.evt[k]; if (t < ev.next) continue;
      const jit = 0.8 + this.rnd() * 0.4;
      if (k === 'cutin') {
        const opts = [e.lane - 1, e.lane + 1].filter(l => l >= 0 && l < 3);
        const ln = opts[Math.floor(this.rnd() * opts.length)], c = (ln - 1) * LW;
        this.objs = this.objs.filter(o => !(Math.abs(o.d - c) < 2.2 && o.s > e.s - 6 && o.s < e.s + 70));
        this.spawn('car', e.s + 22 + this.rnd() * 10, c, Math.max(8, e.v - 1),
          { vdes: Math.max(13, e.v - 8), sc: { kind: 'cut', t: 0.6, target: e.lane, slowT: 9 } });
        ev.next = t + 15 * jit;
      } else if (k === 'hardbrake') {
        const c = (e.lane - 1) * LW;
        let lead = null;
        for (const o of this.objs) if (o.type !== 'ped' && Math.abs(o.d - c) < 1.8 && o.s > e.s + 15 && o.s < e.s + 110 && (!lead || o.s < lead.s)) lead = o;
        if (!lead) {
          this.objs = this.objs.filter(o => !(Math.abs(o.d - c) < 2.2 && o.s > e.s - 6 && o.s < e.s + 120));
          lead = this.spawn('car', e.s + 48 + this.rnd() * 12, c, e.v * 0.98, { vdes: e.v });
        }
        lead.sc = { kind: 'brake', t: 1.5, phase: 'wait' };
        ev.next = t + 32 * jit;
      } else if (k === 'stopped') {
        const c = (e.lane - 1) * LW, s = e.s + Math.max(230, e.v * 9);
        this.objs = this.objs.filter(o => !(Math.abs(o.d - c) < 2.2 && o.s > s - 60 && o.s < s + 40));
        this.spawn('car', s, c, 0, { static: true, vdes: 0, hazard: true });
        ev.next = t + 50 * jit;
      } else if (k === 'ped') {
        const c = (e.lane - 1) * LW, dir = this.rnd() < 0.5 ? 1 : -1, T = 5.0;
        this.spawn('ped', e.s + e.v * (T + 0.8) + 8, c - dir * 1.4 * T, 0, {
          vd: dir * 1.4, vdes: 0, pauseD: c + dir * 0.4, pauseT: this.rnd() < 0.3 ? 1.6 : 0,
        });
        ev.next = t + 28 * jit;
      }
    }
  }

  trafficStep(dt) {
    const e = this.ego, all = this.objs;
    for (const o of all) {
      if (o.type === 'ped') {
        if (o.paused > 0) { o.paused -= dt; o.vdReal = 0; if (o.paused <= 0) o.vd = o.vdSave; }
        else o.d += o.vd * dt;
        if (o.pauseT > 0 && !o.paused && ((o.vd > 0 && o.d >= o.pauseD) || (o.vd < 0 && o.d <= o.pauseD))) {
          o.vdSave = o.vd; o.vd = 0; o.paused = o.pauseT; o.pauseT = 0;
        }
        if (Math.abs(o.d) > 10 && Math.abs(o.d - (e.lane - 1) * LW) > 9) o.dead = true;
        continue;
      }
      if (o.static) continue;
      const sc = o.sc;
      if (sc && sc.kind === 'cut') {
        if (sc.t > 0) { sc.t -= dt; if (sc.t <= 0) o.tgtLane = sc.target; }
        else { sc.slowT -= dt; if (sc.slowT <= 0) { o.vdes = 26; o.sc = null; } }
      }
      if (o.tgtLane !== undefined) {
        const dc = (o.tgtLane - 1) * LW, err = dc - o.d;
        o.vd = clamp(err * 1.2, -1.4, 1.4); o.d += o.vd * dt;
        if (Math.abs(err) < 0.05) { o.d = dc; o.vd = 0; o.tgtLane = undefined; }
      }
      // IDM car following (ego is a leader as well)
      let lead = null, lgap = 1e9;
      const cand = (x, sx, dx, lx, wx) => {
        if (sx <= o.s || Math.abs(dx - o.d) > (o.wid + wx) / 2 + 0.7) return;
        const g = sx - o.s - (o.len + lx) / 2; if (g < lgap) { lgap = g; lead = x; }
      };
      for (const x of all) if (x !== o && !x.dead) cand(x, x.s, x.d, x.len, x.wid);
      cand(e, e.s, e.d, e.len, e.wid);
      const T = 1.4, amax = 1.5, b = 2.3;
      let a = amax * (1 - Math.pow(o.v / Math.max(o.vdes, 1), 4));
      if (lead) {
        const lv = lead === e ? e.v : lead.v, g = Math.max(lgap, 0.1);
        const sstar = 2 + Math.max(0, o.v * T + o.v * (o.v - lv) / (2 * Math.sqrt(amax * b)));
        a = amax * (1 - Math.pow(o.v / Math.max(o.vdes, 1), 4) - Math.pow(sstar / g, 2));
      }
      if (sc && sc.kind === 'brake') {
        if (sc.phase === 'wait') { sc.t -= dt; if (sc.t <= 0) { sc.phase = 'brake'; sc.t = 4; } }
        else if (sc.phase === 'brake') { a = -5.5; sc.t -= dt; if (o.v < 2 || sc.t <= 0) { sc.phase = 'hold'; sc.t = 1.5; } }
        else { a = o.v > 2.5 ? -2 : 0; sc.t -= dt; if (sc.t <= 0) o.sc = null; }
      }
      o.a = clamp(a, -9, 2); o.v = Math.max(0, o.v + o.a * dt); o.s += o.v * dt;
    }
    if (all.some(o => o.dead)) this.objs = all.filter(o => !o.dead);
  }

  /* ---------------- perception (camerad -> modeld -> radard) */
  gtLane() {
    const e = this.ego, k = this.road.k(e.s), dc = (e.lane - 1) * LW;
    return { c0: (dc - e.d) * Math.cos(e.psi), c1: -Math.tan(e.psi), c2: k / 2, q: 1, src: 'gt' };
  }
  setCvLane(res) {   // called by the UI after the computer-vision lane detector ran on the camera frame
    if (!res || !res.valid) { this.cvLane = { valid: false, t: this.t }; return; }
    const p = this.cvLane;
    if (p && p.valid && Math.abs(res.c0 - p.c0) < 2.4) {
      const a = 0.55;
      res = { c0: p.c0 + a * (res.c0 - p.c0), c1: p.c1 + a * (res.c1 - p.c1), c2: p.c2 + a * (res.c2 - p.c2), q: res.q, valid: true };
    }
    res.t = this.t; res.src = 'cv'; this.cvLane = res;
  }
  weatherRange() { return ({ clear: 130, rain: 105, fog: 55, night: 85 })[this.cfg.weather] || 130; }

  percTick(dt) {
    const e = this.ego, cam = CAM, road = this.road;
    let lane;
    if (this.cfg.perception === 'cv') {
      if (this.cvLane && this.cvLane.valid && this.t - this.cvLane.t < 0.4) { lane = this.cvLane; this.laneLost = 0; this.lastLane = lane; }
      else { this.laneLost += dt; lane = this.lastLane || this.gtLane(); }
    } else { lane = this.gtLane(); this.laneLost = 0; }
    const sh = e.shiftEff;
    const path = { c0: lane.c0 + sh, c1: lane.c1, c2: lane.c2 };
    const pathY = x => path.c0 + path.c1 * x + path.c2 * x * x;
    const range = this.weatherRange();
    const p = {};
    const upd = (o, x, xm, ym, vrelM, vyM, conf) => {
      let tr = this.tracks.get(o.id);
      if (!tr) {
        tr = { id: o.id, cls: o.type, x: xm, y: ym, vrel: vrelM, vy: vyM, len: o.len, wid: o.wid, hgt: o.hgt, age: 0, miss: 0, lastT: this.t, pend: [], pT: 0, color: o.color };
        this.tracks.set(o.id, tr);
      } else {
        tr.x += 0.4 * (xm - tr.x); tr.y += 0.5 * (ym - tr.y);
        tr.vrel += 0.3 * (vrelM - tr.vrel); tr.vy += 0.3 * (vyM - tr.vy); tr.age++; tr.lastT = this.t;
      }
      tr.conf = conf; tr.miss = 0;
    };
    for (const tr of this.tracks.values()) tr.x += tr.vrel * dt;   // predict step (all tracks)
    const radar = o => [(o.type === 'ped' ? 0 : o.v) - e.v * Math.cos(e.psi) + this.gauss() * 0.15,
      (o.paused > 0 ? 0 : o.vd) - e.v * Math.sin(e.psi) + this.gauss() * 0.25];
    if (this.cfg.detector === 'yolo') {
      // real neural detector boxes (from the browser) fused with simulated radar velocity, as in openpilot's radard
      const Y = this.yolo;
      if (Y && Y.fresh) {
        Y.fresh = false; this.yoloStats = { det: Y.dets.length, matched: 0, fp: 0 };
        const pe = {};
        const objs = this.objs.map(o => { road.toEgo(this.pose, o.s - o.len / 2, o.d, pe); return { o, x: pe.x, y: pe.y }; });
        const used = new Set();
        for (const d of Y.dets) {
          if (d.vb - cam.vh < 1.5) continue;
          const xm = cam.f * cam.h / (d.vb - cam.vh), ym = (cam.cx - d.u) * xm / cam.f;
          let best = null, bd = 1e9;
          for (const c of objs) {
            if (used.has(c.o.id)) continue;
            const dx = Math.abs(c.x - xm), dy = Math.abs(c.y - ym);
            if (dx < Math.max(6, 0.2 * xm) && dy < 2.4) { const dd = dx / 3 + dy; if (dd < bd) { bd = dd; best = c; } }
          }
          if (!best) { this.yoloStats.fp++; continue; }
          used.add(best.o.id); this.yoloStats.matched++;
          const [vr, vy] = radar(best.o);
          upd(best.o, best.x, xm, ym, vr, vy, d.conf);
        }
        this.yoloStats.miss = objs.filter(c => c.x > 4 && c.x < 90 && Math.abs(c.y) < 12 && !used.has(c.o.id)).length;
      }
    } else {
      for (const o of this.objs) {
        road.toEgo(this.pose, o.s - o.len / 2, o.d, p);
        const x = p.x, y = p.y, rmax = o.type === 'ped' ? range * 0.6 : range;
        if (x < 3 || x > rmax) continue;
        const u = cam.cx - cam.f * y / x; if (u < -60 || u > 700) continue;
        if (this.rnd() < 0.02 + 0.5 * Math.pow(x / rmax, 3)) continue;
        const vb = cam.vh + cam.f * cam.h / x + this.gauss() * 1.0;
        if (vb - cam.vh < 1.5) continue;
        const xm = cam.f * cam.h / (vb - cam.vh), um = u + this.gauss() * 1.5, ym = (cam.cx - um) * xm / cam.f;
        const [vr, vy] = radar(o);
        upd(o, x, xm, ym, vr, vy, clamp(0.99 - 0.5 * Math.pow(x / rmax, 2) + this.gauss() * 0.02, 0.3, 0.99));
      }
    }
    const missMax = this.cfg.detector === 'yolo' ? 24 : 8;
    for (const [id, tr] of this.tracks) {
      tr.miss = Math.round((this.t - tr.lastT) / 0.05);
      if (tr.miss > missMax || tr.x < -2) this.tracks.delete(id);
    }
    // lead selection + cut-in classifier labels (online logistic regression)
    let lead = null; const list = [];
    for (const tr of this.tracks.values()) {
      if (tr.miss > missMax * 0.75) continue;
      const yrel = tr.y - pathY(tr.x);
      tr.yrel = yrel; tr.vyrel = tr.vy;
      tr.inPath = Math.abs(yrel) < tr.wid / 2 + 1.0;
      tr.toward = -sgn(yrel) * tr.vy;
      tr.f = [1, Math.abs(yrel) / 3.7, tr.toward / 1.5, tr.x / 60, clamp(-tr.vrel / 10, -1, 1)];
      if (tr.inPath && tr.conf > 0.45 && tr.age >= 1 && tr.x > 0 && (!lead || tr.x < lead.x)) lead = tr;
      // labelling
      tr.pT += dt;
      if (tr.inPath) { for (const q of tr.pend) this.ai.pmTrain(q.f, 1); tr.pend.length = 0; }
      else {
        if (tr.pT >= 0.25 && Math.abs(yrel) < 8 && tr.x < 90) { tr.pT = 0; tr.pend.push({ t: this.t, f: tr.f.slice() }); }
        while (tr.pend.length && this.t - tr.pend[0].t > 3) this.ai.pmTrain(tr.pend.shift().f, 0);
      }
      tr.pMerge = tr.inPath ? 1 : this.ai.pmProb(tr.f);
      list.push(tr);
    }
    // generative future sampler
    const fut = []; let risk = 0, tcol = 9;
    const K = 16, ts = [0.5, 1, 1.5, 2, 2.5, 3, 3.5];
    for (const tr of list) {
      if (tr.x > 105 || tr.x < 3 || Math.abs(tr.yrel) > 9) continue;
      const va = tr.cls === 'ped' ? 0 : e.v + tr.vrel, isPed = tr.cls === 'ped';
      let hits = 0; const samples = [];
      for (let k = 0; k < K; k++) {
        const merge = !tr.inPath && this.rnd() < tr.pMerge;
        let vy;
        if (merge) vy = -sgn(tr.yrel) * Math.max(0.5, (Math.abs(tr.yrel) - 0.8) / (1.2 + this.rnd() * 1.8));
        else if (tr.inPath) vy = tr.vyrel * (0.2 + 0.4 * this.rnd());
        else vy = tr.vyrel * (0.5 + 0.5 * this.rnd());
        if (isPed && !merge && !tr.inPath) vy = this.rnd() < 0.3 ? 0 : tr.vyrel;
        const u = this.rnd();
        const ao = isPed ? 0 : (u < 0.75 ? 0 : (u < 0.9 ? -(3 + 3 * this.rnd()) : 1.5));
        const pts = []; let hit = false, tc = 9;
        for (const t of ts) {
          const vo = Math.max(0, va + ao * t), xo = tr.x + (va * t + 0.5 * ao * t * t) + (ao < 0 && va + ao * t < 0 ? (va * va) / (2 * -ao) - (va * t + 0.5 * ao * t * t) : 0);
          const xe = Math.max(0, e.v * t + 0.5 * Math.min(e.aPlan, 0.5) * t * t);
          let yo = tr.yrel + vy * t;
          if (merge && Math.abs(yo) < 0.3) yo = 0.3 * sgn(yo);
          pts.push([xo, pathY(xo) + yo]);
          if (Math.abs(xo - xe) < (tr.len + 4.6) / 2 + 1.0 && Math.abs(yo) < (tr.wid + 1.9) / 2 + 0.15) { hit = true; tc = Math.min(tc, t); }
        }
        if (hit) { hits++; tcol = Math.min(tcol, tc); }
        samples.push({ pts, hit });
      }
      tr.risk = hits / K; risk = Math.max(risk, tr.risk); fut.push({ id: tr.id, samples });
    }
    this.fut = fut;
    this.P = { lane, path, tracks: list, lead, risk, tcol, t: this.t, laneLost: this.laneLost };
  }

  /* ---------------- planning + control (plannerd -> controlsd) */
  inputs() {
    const e = this.ego, P = this.P, ln = P.lane;
    const la = clamp(0.9 * e.v + 4, 8, 35);
    const y_t = P.path.c0 + P.path.c1 * la + P.path.c2 * la * la;
    const lead = P.lead;
    const kapRoad = Math.abs(2 * ln.c2), vlim = Math.sqrt(2.3 / Math.max(kapRoad, 1e-4));
    return {
      la, y_t, c0: P.path.c0, c1: P.path.c1, c2: P.path.c2,
      leadOn: lead ? 1 : 0, gap: lead ? Math.max(lead.x - 2.3, 0.1) : 100, dv: lead ? -lead.vrel : 0,
      vset: Math.max(6, Math.min(e.setSpeed, vlim)), risk: P.risk,
    };
  }
  feats(I) {
    const e = this.ego;
    return [I.c0 / 1.5, I.c1 * 8, I.c2 * 300, I.y_t / 3, e.v / 30, (I.vset - e.v) / 10,
      I.leadOn, Math.min(I.gap, 80) / 80, I.dv / 10, Math.min(I.risk, 1)];
  }
  expert(I) {
    const e = this.ego, v = e.v;
    const kap = 2 * I.y_t / (I.la * I.la + I.y_t * I.y_t);
    const T = 1.4, s0 = 2.0, amax = 1.6, b = 2.0;
    const free = 1 - Math.pow(v / Math.max(I.vset, 0.5), 4);
    let inter = 0;
    if (I.leadOn) {
      const sstar = s0 + Math.max(0, v * T + v * I.dv / (2 * Math.sqrt(amax * b)));
      inter = Math.pow(sstar / Math.max(I.gap, 0.3), 2);
    }
    return { kap, acc: clamp(amax * (free - inter), -4.5, 2.0) };
  }

  laneFree(lane) {
    if (lane < 0 || lane > 2) return false;
    const e = this.ego, c = (lane - 1) * LW;
    for (const o of this.objs) {
      if (o.type === 'ped') continue;
      const near = Math.abs(o.d - c) < 2.4 || o.tgtLane === lane;
      if (!near) continue;
      const ds = o.s - e.s, gap = Math.abs(ds) - (o.len + e.len) / 2;
      if (ds >= 0 && gap < 14 + Math.max(0, e.v - o.v) * 2.6) return false;
      if (ds < 0 && gap < 11 + Math.max(0, o.v - e.v) * 3.2) return false;
    }
    return true;
  }
  requestLaneChange(dir) {
    const e = this.ego; if (!e.engaged || e.tgtLane !== e.lane || e.lcCool > 0) return false;
    const tl = e.lane + dir;
    if (tl < 0 || tl > 2) { this.alert('No lane available', 'warn'); return false; }
    if (!this.laneFree(tl)) { this.alert('Lane change blocked - vehicle in blind spot', 'warn'); return false; }
    e.tgtLane = tl; e.lcCool = 6; this.stats.lc++; this.alert('Lane change ' + (dir > 0 ? 'left' : 'right'), 'info'); return true;
  }
  autoLane() {
    const e = this.ego, P = this.P; if (!e.engaged || e.tgtLane !== e.lane || e.lcCool > 0) return;
    const lead = P.lead;
    if (lead && lead.x < 75 && e.setSpeed - (e.v + lead.vrel) > 3) {
      const order = this.rnd() < 0.5 ? [1, -1] : [-1, 1];
      for (const dir of order) if (this.laneFree(e.lane + dir)) { this.requestLaneChange(dir); return; }
    } else if (!lead && e.lane > 0 && this.t > 8 && this.laneFree(e.lane - 1)) {
      const c = (e.lane - 2) * LW; let ok = true;
      for (const o of this.objs) if (o.type !== 'ped' && Math.abs(o.d - c) < 2.4 && o.s > e.s && o.s < e.s + 110 && o.v < e.setSpeed - 2) ok = false;
      if (ok && this.rnd() < 0.15) this.requestLaneChange(-1);
    }
  }

  injectNoise(dt) {   // data-collection disturbances so the learner sees recovery situations (DAgger-style)
    const n = this.noiseState, e = this.ego;
    n.ou += (-n.ou * 0.8 + this.gauss() * 0.0035) * dt * 2;
    n.nextDist -= dt;
    if (n.nextDist <= 0) {
      n.nextDist = 4 + this.rnd() * 6;
      const r = this.rnd();
      if (r < 0.45) { e.d += (this.rnd() - 0.5) * 1.8; e.psi += (this.rnd() - 0.5) * 0.06; }
      else if (r < 0.75) e.setSpeed = (65 + this.rnd() * 55) / 3.6;
      else this.requestLaneChange(this.rnd() < 0.5 ? 1 : -1);
    }
  }

  controlStep(dt) {
    const e = this.ego, ai = this.ai; if (!this.P) return;
    if (e.lcCool > 0) e.lcCool -= dt;
    // lane bookkeeping (localisation)
    const nl = clamp(Math.round(e.d / LW) + 1, 0, 2);
    if (nl !== e.lane) { e.shiftEff -= (nl - e.lane) * LW; e.lane = nl; }
    if (e.tgtLane === e.lane && Math.abs(e.shiftEff) < 0.05 && Math.abs(e.d - (e.lane - 1) * LW) < 0.35) e.shiftEff = 0;
    const want = (e.tgtLane - e.lane) * LW;
    e.shiftEff += clamp(want - e.shiftEff, -0.9 * dt, 0.9 * dt);
    this.autoT += dt;
    if (this.autoT > 0.5) { this.autoT = 0; if (this.cfg.autoPass) this.autoLane(); }
    if (this.cfg.noise) this.injectNoise(dt);

    const I = this.inputs(); this.I = I;
    const ex = this.expert(I), f = this.feats(I);
    const out = ai.net.forward(f); this.nnOut = [out[0] / 100, out[1] * 3.5]; this.f = f; this.ex = ex;
    if (e.engaged) {
      this.lblT += dt;
      if (this.lblT >= 0.1) { this.lblT = 0; if (this.cfg.collect && e.v > 2) ai.push(f, [ex.kap * 100, ex.acc / 3.5]); }
    }
    if (!e.engaged) {   // manual driving
      const inp = this.input;
      const dCmd = inp.steer * 0.05;
      e.delta += clamp((dCmd - e.delta) * 0.25, -0.5 * dt, 0.5 * dt);
      const aT = inp.accel > 0 ? 2.0 : (inp.accel < 0 ? -5 : -0.4 - 0.0008 * e.v * e.v);
      e.aCmd += clamp(aT - e.aCmd, -14 * dt, 5 * dt); e.aPlan = e.aCmd; return;
    }
    if (Math.abs(this.input.steer) > 0.5 || this.input.accel < 0) { this.disengage('Driver override'); return; }

    let kap = ex.kap, acc = ex.acc;
    if (this.cfg.policy === 'nn') { kap = this.nnOut[0]; acc = clamp(this.nnOut[1], -4.5, 2.0); }
    if (this.cfg.noise) kap += this.noiseState.ou;
    e.aPlan = acc;
    // safety supervisor (independent of the learned policy)
    let aeb = false; e.ttc = 99;
    if (I.leadOn && I.dv > 0.4) {
      e.ttc = I.gap / I.dv;
      if (e.ttc < 1.5 && I.gap < 45) { acc = -8; aeb = true; }
      else if (e.ttc < 2.6) { if (!this.fcwOn) this.stats.fcw++; this.fcwOn = true; } else this.fcwOn = false;
      this.stats.minTTC = Math.min(this.stats.minTTC, e.ttc);
    }
    if (!aeb && I.risk > 0.3) acc = Math.min(acc, -(1.5 + 4 * I.risk));
    if (aeb) { if (!this.aebOn) { this.stats.aeb++; this.alert('AUTOMATIC EMERGENCY BRAKING', 'crit'); } e.aCmd = -8; }
    else { e.aCmd += clamp(acc - e.aCmd, -12 * dt, 2.2 * dt); }
    this.aebOn = aeb;
    // lateral
    const kmax = 3.2 / Math.max(e.v * e.v, 9);
    kap = clamp(kap, -kmax, kmax);
    const dCmd = clamp(Math.atan(WB * kap), -0.3, 0.3);
    e.delta += clamp((dCmd - e.delta) * Math.min(1, dt / 0.1), -0.4 * dt, 0.4 * dt);
    if (this.cfg.perception === 'cv' && this.laneLost > 1.5) this.disengage('Lane lines lost - TAKE CONTROL', true);
  }
  disengage(msg, crit) {
    const e = this.ego; if (!e.engaged) return;
    e.engaged = false; this.stats.dis++; this.alert(msg, crit ? 'crit' : 'warn');
    e.tgtLane = e.lane; e.shiftEff = 0;
  }
  engage() {
    const e = this.ego; e.engaged = true; e.tgtLane = e.lane; e.shiftEff = 0; e.aCmd = e.a; this.laneLost = 0;
    this.alert('openpilot engaged', 'info');
  }

  egoStep(dt) {
    const e = this.ego, R = this.road, k = R.k(e.s), jPrev = e.a;
    e.a += (e.aCmd - e.a) * Math.min(1, dt / 0.3);
    e.v = Math.max(0, e.v + e.a * dt); if (e.v === 0 && e.a < 0) e.a = 0;
    const sd = e.v * Math.cos(e.psi) / Math.max(0.2, 1 - k * e.d);
    e.s += sd * dt; e.d += e.v * Math.sin(e.psi) * dt;
    e.psi += (e.v * Math.tan(e.delta) / WB - k * sd) * dt;
    this.stats.dist += e.v * dt; if (e.engaged) this.stats.engT += dt;
    if (e.engaged) { this.stats.jerkSum += Math.abs((e.a - jPrev) / dt); this.stats.jerkN++; }
    this.updatePose();
    if (Math.abs(e.d) > ROAD_HALF + 2.5 && !this.crashed) { this.crashed = { kind: 'road departure' }; this.stats.crashes++; this.alert('ROAD DEPARTURE', 'crit'); }
  }
  collide() {
    const e = this.ego; if (this.crashed) return;
    for (const o of this.objs) {
      if (Math.abs(o.s - e.s) < (o.len + e.len) / 2 && Math.abs(o.d - e.d) < (o.wid + e.wid) / 2) {
        this.crashed = { kind: 'collision with ' + (o.type === 'ped' ? 'pedestrian' : o.type), id: o.id };
        this.stats.crashes++; this.alert('COLLISION', 'crit'); return;
      }
    }
  }

  step(dt) {
    if (this.crashed) return;
    this.t += dt;
    this.events();
    this.trafficStep(dt);
    this.trafT += dt; if (this.trafT > 0.5) { this.trafT = 0; this.manageTraffic(); }
    this.perTimer += dt; if (this.perTimer >= 0.05) { this.perTimer -= 0.05; this.percTick(0.05); this.perFlag = true; }
    this.controlStep(dt);
    this.egoStep(dt);
    this.collide();
    this.histT += dt;
    if (this.histT >= 0.1) {
      this.histT = 0; const e = this.ego;
      this.hist.push({ t: this.t, v: e.v * 3.6, vs: Math.min(e.setSpeed, this.I ? this.I.vset : e.setSpeed) * 3.6, a: e.a, d: e.d - (e.lane - 1) * LW, delta: e.delta,
        nnK: this.nnOut ? this.nnOut[0] : 0, exK: this.ex ? this.ex.kap : 0, nnA: this.nnOut ? this.nnOut[1] : 0, exA: this.ex ? this.ex.acc : 0 });
      if (this.hist.length > 400) this.hist.shift();
    }
  }
}

/* ------------- scenario "prompt -> config" (keyword grammar; drop-in slot for an LLM) */
function parsePrompt(text) {
  const s = (text || '').toLowerCase(), cfg = { events: {}, weather: 'clear', density: 10, curve: 1 }, tags = [];
  const has = (...w) => w.some(x => s.includes(x));
  if (has('night', 'dark', 'midnight')) { cfg.weather = 'night'; tags.push('night'); }
  else if (has('fog', 'mist', 'haze')) { cfg.weather = 'fog'; tags.push('fog'); }
  else if (has('rain', 'storm', 'wet', 'downpour')) { cfg.weather = 'rain'; tags.push('rain'); }
  if (has('cut in', 'cut-in', 'cutin', 'merge', 'swerve', 'aggressive')) { cfg.events.cutin = true; tags.push('cut-in'); }
  if (has('brake', 'sudden stop', 'slam', 'slows', 'phantom')) { cfg.events.hardbrake = true; tags.push('hard-brake lead'); }
  if (has('stopped', 'broken down', 'breakdown', 'accident', 'crash', 'debris', 'obstacle')) { cfg.events.stopped = true; tags.push('stopped vehicle'); }
  if (has('pedestrian', 'jaywalk', 'person', 'child', 'deer', 'animal')) { cfg.events.ped = true; tags.push('crossing pedestrian'); }
  if (has('dense', 'heavy traffic', 'rush', 'busy', 'congest', 'crowd')) { cfg.density = 34; tags.push('dense traffic'); }
  else if (has('empty', 'quiet', 'light traffic', 'deserted')) { cfg.density = 3; tags.push('light traffic'); }
  if (has('curvy', 'curve', 'winding', 'twisty', 'sharp', 'mountain')) { cfg.curve = 1.8; tags.push('winding road'); }
  else if (has('straight', 'flat')) { cfg.curve = 0.25; tags.push('straight road'); }
  const m = s.match(/(\d{2,3})\s*(km\/h|kph|mph)/);
  if (m) { let v = +m[1]; if (m[2] === 'mph') v *= 1.609; cfg.setSpeed = clamp(v, 30, 140) / 3.6; tags.push(Math.round(v) + ' ' + m[2]); }
  if (!tags.length) tags.push('nominal highway');
  return { cfg, tags };
}

/* ------------- fast headless data generation + pre-training for the neural policy */
function genEpisode(ai, seed, seconds) {
  const r = mulberry32(seed * 31 + 5);
  const ev = {}; if (r() < 0.6) ev.cutin = true; if (r() < 0.4) ev.hardbrake = true; if (r() < 0.25) ev.stopped = true; if (r() < 0.3) ev.ped = true;
  const w = new World(ai, {
    seed, curve: [0.3, 1, 1.6, 2][Math.floor(r() * 4)], density: [4, 10, 24, 36][Math.floor(r() * 4)], events: ev,
    setSpeed: (70 + r() * 55) / 3.6, noise: true, autoPass: r() < 0.5, collect: true,
  });
  const dt = 0.02; let n = 0;
  while (w.t < seconds) {
    w.step(dt); n++;
    if (w.crashed) { w.reset(Object.assign({}, w.cfg, { seed: seed + 1000 + n })); }
    else if (!w.ego.engaged) w.engage();
  }
}

const Sim = { World, Road, MLP, makeAI, parsePrompt, genEpisode, CAM, DIM, LW, NLANES, ROAD_HALF, WB, NFEAT, mulberry32, clamp };
root.Sim = Sim;
if (typeof module !== 'undefined' && module.exports) module.exports = Sim;
})(typeof window !== 'undefined' ? window : globalThis);
