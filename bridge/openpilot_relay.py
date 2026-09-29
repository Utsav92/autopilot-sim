#!/usr/bin/env python3
"""Relay real openpilot output (cereal messages) to the browser UI as JSON over HTTP.

Run INSIDE the openpilot checkout (WSL), with openpilot's venv active and the sim running:
    cd ~/openpilot && source .venv/bin/activate   # or: tools/op.sh venv
    PYTHONPATH=. python /path/to/autopilot-sim/bridge/openpilot_relay.py [--port 8765]

Browser (Windows) reaches it at http://localhost:8765/state (WSL2 forwards localhost).
Stdlib only - no extra pip packages. Field names follow openpilot/cereal/log.capnp.
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import openpilot.cereal.messaging as messaging

SERVICES = ['modelV2', 'carState', 'selfdriveState', 'radarState', 'longitudinalPlan']
STATE = {'t': 0, 'alive': False}
LOCK = threading.Lock()


def xyz(d):
  return {'x': list(d.x), 'y': list(d.y)}


def poll():
  sm = messaging.SubMaster(SERVICES)
  while True:
    sm.update(50)
    s = {'t': time.time(), 'alive': all(sm.alive[k] for k in ('modelV2', 'carState'))}
    if sm.seen['carState']:
      cs = sm['carState']
      s['car'] = {'vEgo': cs.vEgo, 'aEgo': cs.aEgo, 'steeringAngleDeg': cs.steeringAngleDeg, 'brakePressed': cs.brakePressed}
    if sm.seen['selfdriveState']:
      ss = sm['selfdriveState']
      s['selfdrive'] = {'enabled': ss.enabled, 'active': ss.active, 'alert': ss.alertText1}
    if sm.seen['modelV2']:
      m = sm['modelV2']
      s['model'] = {
        'path': xyz(m.position),
        'laneLines': [xyz(l) for l in m.laneLines],
        'roadEdges': [xyz(e) for e in m.roadEdges],
        'leads': [{'prob': l.prob, 'x': l.x[0], 'v': l.v[0]} for l in m.leadsV3 if l.prob > 0.3],
      }
    if sm.seen['radarState']:
      r = sm['radarState'].leadOne
      s['radar'] = {'present': r.status if hasattr(r, 'status') else True, 'dRel': r.dRel, 'yRel': r.yRel, 'vRel': r.vRel, 'vLead': r.vLead}
    if sm.seen['longitudinalPlan']:
      lp = sm['longitudinalPlan']
      s['plan'] = {'accels': list(lp.accels)[:10], 'speeds': list(lp.speeds)[:10]}
    with LOCK:
      STATE.clear()
      STATE.update(s)


class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
    with LOCK:
      body = json.dumps(STATE).encode()
    self.send_response(200)
    self.send_header('Content-Type', 'application/json')
    self.send_header('Access-Control-Allow-Origin', '*')
    self.end_headers()
    self.wfile.write(body)

  def log_message(self, *a):
    pass


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('--port', type=int, default=8765)
  args = ap.parse_args()
  threading.Thread(target=poll, daemon=True).start()
  print(f'openpilot relay on http://0.0.0.0:{args.port}/state')
  ThreadingHTTPServer(('0.0.0.0', args.port), Handler).serve_forever()
