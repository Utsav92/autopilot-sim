# AutoPilot Sim

Browser-based autonomous-driving simulator whose architecture mirrors [commaai/openpilot](https://github.com/commaai/openpilot)
(`camerad -> modeld -> radard -> plannerd -> controlsd -> card`), plus an optional bridge to run the real openpilot in MetaDrive.

## Run

```
node server.js                              # http://localhost:8811  (static files + LLM endpoint)
ANTHROPIC_API_KEY=sk-... node server.js     # also enables Claude-generated scenarios
```
(`python -m http.server 8811` also works for everything except LLM scenarios.)
URL params: `?cv=1&nn=1&yolo=1&scenario=night+rain+cut-in`.
Headless checks: `node test/headless.js` (planner + NN in 6 scenarios), `test/cvtest.html` (CV lane keeping per weather),
`test/yolotest.html`, `test/yololoop.html` (closed loop with YOLO) via headless Chrome `--dump-dom`.

## What is real

| Technique | Where | Status |
|---|---|---|
| Computer vision | `js/cv.js`: threshold -> paint runs -> vehicle masking -> RANSAC quadratic lane-centre fit, Sobel edge view | Real, on rendered pixels |
| Object detection | `js/yolo.js`: **YOLOv8n** (COCO) through onnxruntime-web on the rendered frame, fused with simulated radar velocity (like openpilot's radard). Toggle "Object detection". Model: `models/yolov8n.onnx` (from Hyuto/yolov8-onnxruntime-web). Also a simulated detector option. | Real network; the images are synthetic so recall is partial (see below) |
| Deep learning | `js/sim.js`: MLP 10-32-32-2 behaviour-cloned from the planner, DAgger-style, trained online with Adam | Real |
| Machine learning | Online logistic-regression cut-in classifier, self-labelled | Real |
| Generative sampling | Multi-hypothesis future trajectories -> collision risk -> pre-emptive braking | Hand-parameterised mixture, driven by the classifier |
| Scenario generation | Keyword grammar by default; **Claude** via `server.js` when `ANTHROPIC_API_KEY` is set (`SCENARIO_MODEL` to override, default `claude-sonnet-5-5`). Output is validated and clamped server-side. | LLM path tested only against a mock upstream (no key available here) |
| Real openpilot | `bridge/` (WSL + MetaDrive) streams openpilot's `modelV2`/`carState`/`radarState` into the "Real openpilot" panel | Scripts written against openpilot's `tools/op.sh` + `tools/sim`; relay tested against a fake cereal module only |

YOLO note: COCO-trained YOLO is run on a cartoon-ish renderer, so it finds roughly 60% of vehicles within 80 m in clear weather and much
less at night; misses are coasted by the tracker. The renderer was upgraded (wheels, rear window, plates, shadows, pedestrian shape) for this.

## Real openpilot in the browser (bridge/)

Needs Windows 11 + WSL2 with Ubuntu, ~15 GB and time. **Nothing here is run automatically** - installing WSL changes system config.

1. Elevated PowerShell: `powershell -ExecutionPolicy Bypass -File bridge\setup_wsl.ps1`
   (installs Ubuntu-24.04 if there is no distro - reboot if asked, then re-run; then clones openpilot, `tools/op.sh setup`, `tools/op.sh build`).
2. In WSL: `bash /mnt/c/Users/Utsav/autopilot-sim/bridge/run_sim.sh` - starts the JSON relay on :8765 and `tools/op.sh sim`
   (MetaDrive bridge + openpilot). In that terminal press `2` to engage, `1`/`2` to change speed, `S` to disengage.
3. In the web UI, "Real openpilot" panel -> Connect.

The relay (`bridge/openpilot_relay.py`) is stdlib-only and serves `GET /state`.
