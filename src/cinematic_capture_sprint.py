"""Cinematic renders (2026-10-18): replay a sprint ladder rung at 1 ms (recorded) and 0.5 ms, check both against the stored values,
and save the qpos trace + the rendered model (.mjb) for engine/cinematic.py. Env (SPRINT_LAG, SPRINT_RATINGS, ...) must match the ladder.
usage: python cinematic_capture_sprint.py <ladder json> <rung> <tag>  ->  results/demo/cinematic/cache/<tag>.npz/.mjb/.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco
L = json.load(open(sys.argv[1])); ix = int(sys.argv[2]); tag = sys.argv[3]
os.environ.setdefault("SPRINT_LAG", str(L["supplied"]["act_tau"]))
from engine import sprint as SP, core
assert abs(SP.ACT_TAU - float(L["supplied"]["act_tau"])) < 1e-9
rw = L["rungs"][ix]; st_ = rw["stride"] if rw.get("stride_extrapolated") else None
if st_ is not None and os.environ.get("RENDER_EXACT_STRIDE"):   # as engine/render_sprint.py: the exact extrapolated stride, not the rounded record
    from engine import sprint_ceiling as SC; st_ = dict(SP.select_stride(SC.body_spec(L["robot"]), float(rw["stride"]["extrapolated_from"])), extrapolated_from=rw["stride"]["extrapolated_from"])
Pb = SP.Sprint(L["robot"], rw["v"], float(L.get("scale", 1.0)), stride=st_)
r = Pb.run(np.asarray(rw["z"]), core.DT_OPT, record=True); r2 = Pb.run(np.asarray(rw["z"]), core.DT_CHECK)
chk = dict(stored=[rw["meas"][0]["value"], rw["meas"][1]["value"]], replay=[round(float(r["value"]), 4), round(float(r2["value"]), 4)], valid=[bool(r["valid"]), bool(r2["valid"])])
chk["match"] = bool(abs(chk["replay"][0] - chk["stored"][0]) < 1e-3 and abs(chk["replay"][1] - chk["stored"][1]) < 1e-3 and all(chk["valid"]))
out = f"{os.path.dirname(os.path.abspath(__file__))}/../results/demo/cinematic/cache/{tag}"
tr = r["trace"]; np.savez(out + ".npz", Q=np.asarray(tr["qpos"]), t=np.asarray(tr["t"])); mujoco.mj_saveModel(Pb.S[core.DT_OPT]["m"], out + ".mjb", None)
json.dump(dict(src=sys.argv[1], rung=ix, robot=L["robot"], v=rw["v"], lag=SP.ACT_TAU, ratings=SP.RATINGS, check=chk), open(out + ".json", "w"), indent=1); print(tag, chk)
