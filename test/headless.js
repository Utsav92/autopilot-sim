const Sim = require('../js/sim.js');
const t0 = Date.now();
const ai = Sim.makeAI(7);
for (let i = 0; i < 10; i++) Sim.genEpisode(ai, i + 1, 90);
console.log('gen ms', Date.now() - t0, 'samples', ai.n, 'pm w', ai.pm.w.map(x => x.toFixed(2)).join(','));
const t1 = Date.now();
for (let i = 0; i < 40; i++) ai.trainSteps(50, 64, i < 25 ? 3e-3 : 1e-3);
console.log('train ms', Date.now() - t1, 'loss', ai.lastLoss);

function run(name, cfg, secs) {
  const w = new Sim.World(ai, Object.assign({ collect: false }, cfg));
  let maxOff = 0, n = 0;
  while (w.t < secs && !w.crashed) {
    w.step(0.02);
    if (w.ego.tgtLane === w.ego.lane) maxOff = Math.max(maxOff, Math.abs(w.ego.d - (w.ego.lane - 1) * 3.7));
    if (!w.ego.engaged) { w.engage(); n++; }
  }
  console.log(name.padEnd(28), 'crash:', w.crashed ? w.crashed.kind : 'no', 't=', w.t.toFixed(0), 'maxOff', maxOff.toFixed(2), 'minTTC', w.stats.minTTC.toFixed(1), 'aeb', w.stats.aeb, 'reeng', n, 'v', (w.ego.v * 3.6).toFixed(0));
}
for (const pol of ['expert', 'nn']) {
  run(pol + ' highway', { policy: pol, seed: 3 }, 120);
  run(pol + ' curvy dense', { policy: pol, seed: 4, curve: 1.8, density: 30 }, 120);
  run(pol + ' cutin', { policy: pol, seed: 5, events: { cutin: true } }, 120);
  run(pol + ' hardbrake', { policy: pol, seed: 6, events: { hardbrake: true } }, 120);
  run(pol + ' stopped autopass', { policy: pol, seed: 7, events: { stopped: true }, autoPass: true }, 120);
  run(pol + ' ped', { policy: pol, seed: 8, events: { ped: true } }, 120);
}
