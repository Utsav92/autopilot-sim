import time
from types import SimpleNamespace as N
class SubMaster:
  def __init__(self, services):
    self.alive = {s: True for s in services}; self.seen = {s: True for s in services}
  def update(self, ms): time.sleep(ms / 1000)
  def __getitem__(self, k):
    xs = [float(i) for i in range(33)]
    xyz = N(x=xs, y=[0.01 * i for i in xs], z=[0]*33)
    return {
      'carState': N(vEgo=25.0, aEgo=0.1, steeringAngleDeg=1.5, brakePressed=False),
      'selfdriveState': N(enabled=True, active=True, alertText1=''),
      'modelV2': N(position=xyz, laneLines=[xyz]*4, roadEdges=[xyz]*2, leadsV3=[N(prob=0.9, x=[40.0], v=[20.0])]),
      'radarState': N(leadOne=N(dRel=40.0, yRel=0.0, vRel=-5.0, vLead=20.0)),
      'longitudinalPlan': N(accels=[0.0]*10, speeds=[25.0]*10),
    }[k]
