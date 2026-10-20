"""H1 landed stage for the progression page: best APRIOR jump landed at BOTH timesteps (jump14_h1.json), replayed at 1 ms."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, progression as PG
R = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results")
which = sys.argv[1] if len(sys.argv) > 1 else "h1"; sid = sys.argv[2] if len(sys.argv) > 2 else "h1j_4"
RB = json.load(open(f"{R}/phase14/jump14_{which}.json"))
r = max([x for x in RB if x["ok_all"] and x["method"] == "APRIOR"], key=lambda x: min(m["v"] for m in x["meas"]))
log, o, m = PG.rec_landed(r, "H1"); L = o.get("landing", {}); v = o["v"]
sub = f"Phase 14b{'' if sid == 'h1j_4' else ' robust'} · takeoff {v:.2f} m/s, crouch {100*o['mech']['crouch_depth']:.0f} cm, landed {'yes' if L.get('ok') else 'no'} (1 ms; also at 0.5 ms)"
PG.render(m, log, dict(z=1.1, dist=4.8, az=90, el=-5), ("Deep-crouch prior + landing controller (H1)" if sid == "h1j_4" else "H1: deep jump landed (warm start, speed floor)"), sub, f"{R}/viz/progression/{sid}.mp4")
json.dump(dict(seed=r["seed"], v_1ms=v, v_05ms=r["meas"][1]["v"], crouch=o["mech"]["crouch_depth"], landed=bool(L.get("ok")), rot=o["mech"]["flight_rotation"], source=f"jump14_{which}.json"),
          open(f"{R}/viz/progression/{sid}.json", "w"), indent=1)
print("h1 landed stage", r["seed"], round(v, 2), round(r["meas"][1]["v"], 2), L.get("ok"))
