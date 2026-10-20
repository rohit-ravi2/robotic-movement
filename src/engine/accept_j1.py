"""J1 acceptance: the stored jump (robust4 seed 1) and throw (ladder APRIOR best) reproduce through the Task interface."""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from engine import core, task as TK
from tasks.jump import JumpTask
from tasks.throw import ThrowTask
R4 = {r["seed"]: r for r in json.load(open(f"{core.RES}/phase14/jump14_robust4.json"))}
jt = JumpTask("G1"); d = jt.demand_default(); S = jt.plant(d)
z = jt.warm(S, d, R4[1]); out = TK.measure(jt, d, z)
print("JUMP  stored 3.5565 / 3.4964  ->", [round(m["value"], 4) for m in out["meas"]], "ok_all", out["ok_all"], "landed", [m["extra"]["landing"]["ok"] for m in out["meas"]])
TL = json.load(open(f"{core.RES}/phase11/throw_g1_ladder2400.json"))
ta = max([r for r in TL if r["method"] == "APRIOR" and r["valid_all_dt"]], key=lambda r: r["speed"])
tt = ThrowTask("G1"); dT = dict(tt.demand_default(), omega=ta["omega"]); S = tt.plant(dT)
z = tt.warm(S, dT, ta); out = TK.measure(tt, dT, z)
print("THROW stored %.4f ->" % ta["speed"], [round(m["value"], 4) for m in out["meas"]], "ok_all", out["ok_all"], "reasons", [m["reasons"] for m in out["meas"]])
