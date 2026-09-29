#!/usr/bin/env bash
# Run inside WSL after wsl_setup.sh. Starts: openpilot (manager) + MetaDrive bridge + the JSON relay for the browser.
# Keyboard focus must be in this terminal for the bridge: press 2 to engage, 1/2 for speed, S to disengage, q to quit.
# (openpilot/tools/sim/README.md)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/openpilot"
source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH="$PWD"
python "$HERE/openpilot_relay.py" --port 8765 &
RELAY=$!
trap 'kill $RELAY 2>/dev/null || true' EXIT
tools/op.sh sim      # = run_bridge.py (MetaDrive) + launch_openpilot.sh (manager, SIMULATION=1)
