/* Real neural object detector: YOLOv8n (COCO) via onnxruntime-web, run on the rendered dashcam frame. */
(function (root) {
'use strict';
const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/ort.min.js';
const KEEP = { 0: 'ped', 2: 'car', 3: 'car', 5: 'truck', 7: 'truck' };   // COCO ids: person, car, motorcycle, bus, truck
const BASE = document.currentScript.src.replace(/js\/yolo\.js.*$/, '');
const S = 640, PAD_Y = (640 - 360) / 2;
let session = null, loading = null, lctx = null;

function loadScript(src) { return new Promise((res, rej) => { const s = document.createElement('script'); s.src = src; s.onload = res; s.onerror = () => rej(new Error('failed to load ' + src)); document.head.appendChild(s); }); }

async function init(status) {
  if (session) return session;
  if (loading) return loading;
  loading = (async () => {
    status && status('loading onnxruntime-web…');
    if (!root.ort) await loadScript(ORT_URL);
    root.ort.env.wasm.numThreads = 1;
    status && status('loading yolov8n.onnx (12 MB)…');
    session = await root.ort.InferenceSession.create(BASE + 'models/yolov8n.onnx', { executionProviders: ['wasm'] });
    const c = document.createElement('canvas'); c.width = c.height = S; lctx = c.getContext('2d', { willReadFrequently: true });
    status && status('ready');
    return session;
  })();
  try { return await loading; } catch (e) { loading = null; throw e; }
}

function iou(a, b) {
  const x1 = Math.max(a.x0, b.x0), y1 = Math.max(a.y0, b.y0), x2 = Math.min(a.x1, b.x1), y2 = Math.min(a.y1, b.y1);
  const i = Math.max(0, x2 - x1) * Math.max(0, y2 - y1), u = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - i;
  return u > 0 ? i / u : 0;
}

async function infer(srcCanvas, sx, sy, sw, sh, scale, confThr) {
  // draw region (sx,sy,sw,sh) scaled by `scale` into the 640x640 letterbox (top-left aligned, centred vertically)
  const dw = sw * scale, dh = sh * scale, oy = (S - dh) / 2, ox = (S - dw) / 2;
  lctx.fillStyle = 'rgb(114,114,114)'; lctx.fillRect(0, 0, S, S);
  lctx.filter = 'blur(0.8px)'; lctx.drawImage(srcCanvas, sx, sy, sw, sh, ox, oy, dw, dh); lctx.filter = 'none';
  const px = lctx.getImageData(0, 0, S, S).data, n = S * S, inp = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) { inp[i] = px[i * 4] / 255; inp[n + i] = px[i * 4 + 1] / 255; inp[2 * n + i] = px[i * 4 + 2] / 255; }
  const out = await session.run({ [session.inputNames[0]]: new root.ort.Tensor('float32', inp, [1, 3, S, S]) });
  const o = out[session.outputNames[0]], d = o.data, N = o.dims[2], res = [];
  for (let i = 0; i < N; i++) {
    let bc = -1, bs = confThr;
    for (const c in KEEP) { const sc = d[(4 + +c) * N + i]; if (sc > bs) { bs = sc; bc = +c; } }
    if (bc < 0) continue;
    const cx = (d[i] - ox) / scale + sx, cy = (d[N + i] - oy) / scale + sy, w = d[2 * N + i] / scale, h = d[3 * N + i] / scale;
    res.push({ cls: KEEP[bc], coco: bc, conf: bs, x0: cx - w / 2, y0: cy - h / 2, x1: cx + w / 2, y1: cy + h / 2 });
  }
  return res;
}

/* Two passes: full frame + 2x zoomed road-centre crop (finds distant vehicles). Returns boxes in camera pixels. */
async function detect(camCanvas, confThr) {
  await init();
  confThr = confThr || 0.3;
  const full = await infer(camCanvas, 0, 0, 640, 360, 1, confThr);
  const zoom = await infer(camCanvas, 160, 130, 320, 100, 2, confThr + 0.05);
  const cand = full.concat(zoom.filter(z => z.y1 < 205)).sort((a, b) => b.conf - a.conf);   // crop only trusted for small/far boxes
  const res = [];
  for (const c of cand) if (!res.some(r => iou(r, c) > 0.45)) res.push(c);
  for (const r of res) { r.u = (r.x0 + r.x1) / 2; r.vb = r.y1; r.w = r.x1 - r.x0; r.h = r.y1 - r.y0; }
  return res;
}

root.Yolo = { init, detect };
})(window);
