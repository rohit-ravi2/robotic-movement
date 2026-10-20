"""Stage 7 clip for the progression page: the robust deep jump (jump14_robust seed 2; lands at BOTH 1 ms and 0.5 ms), replayed at 1 ms."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, progression as PG
R = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results")
which = sys.argv[1] if len(sys.argv) > 1 else "robust"
RB = json.load(open(f"{R}/phase14/jump14_{which}.json"))
r = max([x for x in RB if x["ok_all"]], key=lambda x: min(m["v"] for m in x["meas"]))
log, o, m = PG.rec_landed(r)
L = o.get("landing", {}); v = o["v"]
sub = f"Phase 14 · takeoff {v:.2f} m/s, crouch {100*o['mech']['crouch_depth']:.0f} cm, landed {'yes' if L.get('ok') else 'no'} (1 ms; also lands at 0.5 ms)"
PG.render(m, log, dict(z=1.0, dist=4.4, az=90, el=-5), "Deep crouch + landing that holds at both timesteps", sub, f"{R}/viz/progression/g1j_7.mp4")
json.dump(dict(seed=r["seed"], v_1ms=v, v_05ms=r["meas"][1]["v"], crouch=o["mech"]["crouch_depth"], landed=bool(L.get("ok")), rot=o["mech"]["flight_rotation"], source=f"jump14_{which}.json"),
          open(f"{R}/viz/progression/g1j_7.json", "w"), indent=1)
print("stage 7", r["seed"], round(v, 2), round(r["meas"][1]["v"], 2), L.get("ok"))
