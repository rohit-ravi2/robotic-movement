"""M5 WALKING GAIT LIBRARY: collect the walking solutions (prior gaits and gain searches) with their validations into one reusable file,
and load any entry back as a ready-to-run walker (warm start for follow-on jobs).

  python engine/walk_library.py build            -> results/movements/M5_walking/gait_library.json (+ technique per entry, recorded rollout)
  from engine import walk_library as WL; P, z = WL.load_gait("G1", 0.6); r = P.run(z)          # replay
  WL.warm_z("G1", 1.0) -> the z (search vector) of the nearest stored gait of that robot, to seed a new search

Entry schema: robot, v (commanded, m/s), turn (rad), source (json path), kind (prior | gain search), stride (the spec-derived stride
record: s, delta, hr, T, Tds, cadence, ...), z (full search vector: sprint knots + amp/tscale/lean + 9 controller gains), params
(decoded: amp, tscale, lean, gains), meas (1 ms and 0.5 ms summaries), validation (fresh-plant runs and perturbations, if run),
technique (measured cadence, step length, double-support fraction, duty factor, arm swing, trunk pitch, energy per metre), git.
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from engine import walk as W, core

LIB = f"{W.RES}/gait_library.json"


def technique(P, z, dt=core.DT_OPT):
    """Measured technique of a gait on one recorded rollout (steady part, after the transient cycles)."""
    r = P.run(z, dt, record=True); tr = r["trace"]; t = np.asarray(tr["t"]); keep = t >= W.N_TRANS*P.R["Tc"]*P.unpack(z)["tscale"]
    S = P.S[dt]; types = S["types"]; qp = np.asarray(tr["qpos"]); qa = S["jarr"]["qad"]
    arm = {tn: round(float(np.ptp(qp[keep, qa[i]])), 3) for i, tn in enumerate(types) if ("shoulder_pitch" in tn or "elbow" in tn)} if keep.any() else {}
    pitch = [W.rpy_mat(W.SP.T.np.eye(3))[1]] if False else []
    if keep.any():
        import mujoco
        d = mujoco.MjData(S["m"]); pitch = []
        for q in qp[keep][::10]:
            d.qpos[:] = q; mujoco.mj_kinematics(S["m"], d); pitch.append(W.rpy_mat(d.xmat[1].reshape(3, 3))[1])
    load = np.asarray(tr["load"])[keep] if keep.any() else np.zeros((0, 2))
    duty = [round(float(np.mean(load[:, i] > 0.02)), 3) for i in (0, 1)] if len(load) else None
    ds_frac_time = round(float(np.mean((load[:, 0] > 0.02) & (load[:, 1] > 0.02))), 3) if len(load) else None
    tq = np.asarray(tr["tq"])[keep] if keep.any() else np.zeros((0, len(types))); tau = S["jarr"]["tau"]
    util = {tn: round(float(np.abs(tq[:, i]).max()/tau[i]), 3) for i, tn in enumerate(types)} if len(tq) else {}
    top = sorted(util.items(), key=lambda kv: -kv[1])[:4]
    return dict(torque_util_max=util, torque_util_top=top, overspeed=r["overspeed"], speed=r["value"], valid=r["valid"], reasons=r["reasons"], cadence_spm=r["cadence_spm"], step_len=r["step_len"], stride_len=(2*r["step_len"] if r["step_len"] else None),
                ds_min_s=r["ds_min"], ds_med_s=r["ds_med"], ds_frac_per_step=r["ds_frac"], ds_frac_time=ds_frac_time, duty_per_leg=duty, flight_max_s=r["flight_max"],
                arm_swing_ptp_rad=arm, pelvis_pitch_mean=(round(float(np.mean(pitch)), 3) if len(pitch) else None), J_per_m=r["J_per_m"], cot=r["cot"], slip_max=r["slip_max"], yaw_end=r["yaw_end"])


def _entries():
    out = []
    for robot in ("G1", "H1"):
        for pth in sorted(glob.glob(f"{W.RES}/{robot}/*.json")):
            if pth.endswith("_validation.json"): continue
            g = json.load(open(pth))
            if "z" not in g or "stride" not in g: continue
            r1, r2 = g.get("r1"), g.get("r2")
            bn = os.path.basename(pth)
            kind = ("prior" if g.get("kind", "").startswith("prior") else "side study: stride override (not the spec-preferred stride)" if "_stride_" in bn
                    else "robust gain search (perturbations in the score)" if "_rob" in bn else "gain search" if "_g" in bn else "search (v1 construction, no gain block)")
            val = json.load(open(pth.replace(".json", "_validation.json"))) if os.path.exists(pth.replace(".json", "_validation.json")) else None
            out.append(dict(robot=robot, v=g["v"], turn=g.get("turn", 0.0), source=os.path.relpath(pth, W.RES), kind=kind, stride=g["stride"], z=g["z"], params=g.get("params"),
                            meas=[r1, r2] if r1 else None, score=g.get("score"), git=g.get("git"), validation=None if val is None else {k: val.get(k) for k in ("nominal_valid_both_dt", "perturb_pass", "perturb_n", "robust", "perturb_pass_2mrad_set", "robust_2mrad_set", "ok")},
                            valid_both_dt=bool(val["nominal_valid_both_dt"]) if val else bool(r1 and r2 and r1.get("valid") and r2.get("valid"))))
    return out


def build(with_technique=True):
    E = _entries()
    if with_technique:
        for e in E:
            if not e["valid_both_dt"]: continue
            P = W.Walk(e["robot"], e["v"], e["turn"], stride=e["stride"]); z = np.asarray(e["z"], float)
            if len(z) < P.n(): z = np.concatenate([z, np.zeros(P.n() - len(z))])
            e["technique"] = technique(P, z)
            print(e["robot"], e["v"], e["turn"], e["kind"], e["technique"]["speed"], e["technique"]["cadence_spm"], flush=True)
    lib = dict(note=__doc__, supplied=W.SUPPLIED, entries=E)
    json.dump(lib, open(LIB, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else (bool(o) if isinstance(o, np.bool_) else float(o)))
    return lib


def best(robot, v, turn=0.0):
    E = [e for e in json.load(open(LIB))["entries"] if e["robot"] == robot and abs(e["v"] - v) < 1e-9 and abs(e["turn"] - turn) < 1e-6 and e["valid_both_dt"]]
    if not E: return None
    def key(e):
        v = e.get("validation") or {}
        return (not v.get("robust", False), -(v.get("perturb_pass", 0) + v.get("perturb_pass_2mrad_set", 0)), e["score"] if e["score"] is not None else 0.0)
    return sorted(E, key=key)[0]


def load_gait(robot, v, turn=0.0):
    e = best(robot, v, turn)
    if e is None: raise KeyError(f"no valid stored gait for {robot} at {v} m/s, turn {turn}")
    P = W.Walk(robot, v, turn, stride=e["stride"]); z = np.asarray(e["z"], float)
    if len(z) < P.n(): z = np.concatenate([z, np.zeros(P.n() - len(z))])
    return P, z


def warm_z(robot, v):
    E = [e for e in json.load(open(LIB))["entries"] if e["robot"] == robot and e["valid_both_dt"]]
    if not E: return None
    return min(E, key=lambda e: abs(e["v"] - v))["z"]


if __name__ == "__main__":
    if sys.argv[1] == "build": build(with_technique="--no-tech" not in sys.argv)
