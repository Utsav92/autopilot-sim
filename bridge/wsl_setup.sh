#!/usr/bin/env bash
# Run inside WSL (Ubuntu 22.04/24.04). Installs openpilot + MetaDrive sim dependencies and builds it.
# Needs ~15 GB disk and a while (dependency install + scons build).
set -euo pipefail
cd "$HOME"
if [ ! -d openpilot ]; then
  git clone --recurse-submodules https://github.com/commaai/openpilot.git
fi
cd openpilot
tools/op.sh setup     # installs system deps + python env (uv); includes MetaDrive for the sim
tools/op.sh build     # scons build of native openpilot components
echo "Done. Next: bridge/run_sim.sh"
