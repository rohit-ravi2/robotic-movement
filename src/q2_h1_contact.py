"""Q2 (2026-10-17): replay the stored H1 sprint rungs with STOCK contact (STOCK_CONTACT=1, solref 0.02) vs the 0.004 override,
on fresh plants at 1 ms and 0.5 ms (each process builds its own plants; ACT_TAU / contact are read at import).
usage: STOCK_CONTACT=0|1 SPRINT_LAG=<s> SPRINT_EXTRAP=1 python q2_h1_contact.py <ladder json> <v> <out json>"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from engine import sprint as SP, core
import branch_i as BI

path, v, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
L = json.load(open(path)); R = [x for x in L["rungs"] if x["v"] == v][0]
P = SP.Sprint("H1", v, stride=dict(R["stride"]))
sol = np.unique(P.S[core.DT_OPT]["m"].geom_solref, axis=0).tolist()
res = {}
for dt in (core.DT_OPT, core.DT_CHECK):
    s = SP.summarize(P.run(np.asarray(R["z"]), dt)); res[str(dt)] = s
    print(dt, s["valid"], round(s["value"], 4), s["reasons"], "slip", s["slip_max"], "os", s["overspeed"], flush=True)
ok = bool(all(res[k]["valid"] for k in res) and min(res[k]["value"] for k in res) >= 0.97*v)
rec = dict(source=path, v=v, stock_contact=os.environ.get("STOCK_CONTACT") == "1", geom_solref=sol, act_tau=SP.ACT_TAU, extrap=bool(os.environ.get("SPRINT_EXTRAP")),
           stored_meas=R["meas"], replay=res, ok_both_dt=ok, contact=SP.SUPPLIED["contact"])
os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(rec, open(out, "w"), indent=1, default=float)
print("OK_BOTH_DT", ok)
