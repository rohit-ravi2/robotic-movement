"""Cinematic renders (2026-10-18): replay a construct ledger at 1 ms (recorded, engine.construct.replay) and at 0.5 ms on a fresh plant,
check both against the stored meas, and save the qpos trace, the model (.mjb) and the release/ball state for engine/cinematic.py.
usage: python cinematic_capture_construct.py <ledger json> <tag>  ->  results/demo/cinematic/cache/<tag>.npz/.mjb/.json"""
import os, sys, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco
from engine import construct as C, core

def replay_dt(path, dt):
    """As engine.task.measure did at the check dt: the reference is rebuilt on the fresh plant (the pickled reference is the 1 ms one)."""
    led = json.load(open(path)); task, demand = C.make_task(led["robot"], C.SCENARIOS[led["scenario"]]); task.plant(demand, core.DT_OPT); S = task.plant(demand, dt, fresh=True)   # same plant order as engine.task.measure (the fresh plant's start offset depends on it)
    p = task.unpack(S, demand, np.asarray(led["z"])); rp = path.replace(".json", "_ref.pkl")
    if os.path.exists(rp) and os.environ.get("PIN_REF_CHECK"): dr = demand.get("direction"); task._refs[(id(S["m"]), None if dr is None else tuple(np.round(dr, 4)), demand.get("capacity_mode", "power"))] = pickle.load(open(rp, "rb"))
    if led["result"].get("amp") is not None: p["amp"] = float(led["result"]["amp"]); p["tscale"] = float(led["result"]["tscale"])
    return task.rollout(S, demand, p, ctrl_fn=task.controller(S, demand, p), record=False)

if __name__ == "__main__":
    path, tag = sys.argv[1], sys.argv[2]; led = json.load(open(path)); st = [m_["value"] for m_ in led["result"]["meas"]]
    task, demand, S, p, rec = C.replay(path); r2 = replay_dt(path, core.DT_CHECK)
    chk = dict(stored=st, replay=[round(float(rec["value"]), 3), round(float(r2["value"]), 3)], valid=[bool(rec["valid"]), bool(r2["valid"])])
    chk["match"] = bool(abs(chk["replay"][0] - st[0]) < 2e-3 and abs(chk["replay"][1] - st[1]) < 2e-3 and all(chk["valid"]))
    m = S["m"]; d = mujoco.MjData(m); tr = rec["trace"]; Q = np.asarray(tr["q"]); V = np.asarray(tr["qvel"])
    d.qpos[:] = Q[-1]; d.qvel[:] = V[-1]; mujoco.mj_forward(m, d); v6 = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, S["site"], v6, 0)
    info = dict(src=path, scenario=led["scenario"], robot=led["robot"], seed=led["seed"], check=chk, dt=float(tr["dt"]), release_time=float(tr["release_time"]), n=len(Q),
                site_pos=d.site_xpos[S["site"]].tolist(), site_vel=v6[3:].tolist(), site_speed=float(np.linalg.norm(v6[3:])), extra={k: rec["extra"][k] for k in ("elev", "az", "overspeed", "ball_speed", "ball_elev", "ball_az") if k in rec["extra"]})
    if "ball_dof" in S:
        a = S["ball_dof"]; jq = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]); info.update(ball_qadr=jq, ball_pos=Q[-1, jq:jq+3].tolist(), ball_vel=V[-1, a:a+3].tolist())
    out = f"{os.path.dirname(os.path.abspath(__file__))}/../results/demo/cinematic/cache/{tag}"
    np.savez(out + ".npz", Q=Q, t=np.arange(len(Q))*float(tr["dt"])); mujoco.mj_saveModel(m, out + ".mjb", None); json.dump(info, open(out + ".json", "w"), indent=1, default=float)
    print(tag, chk, "site speed", round(info["site_speed"], 3), {k: round(v, 3) for k, v in info["extra"].items()})
