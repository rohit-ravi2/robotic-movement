"""Adversarial verifier for K2 (lag-aware stand). FRESH plants (HC_FRESH default on) at 1 ms and 0.5 ms.
A: pot 44 kg head carry at 20 ms lag (refine.json pot_lag20ms z) nominal + 2 NEW perturbations (sigma 0.03, rng 4242).
B: unloaded stand at 20 ms lag (Q1 pot z) nominal + 2 NEW perturbations of the stand params (sigma 0.03, rng 4243).
Every rollout also records the peak |actuator force| of each knee and the peak over all body joints (vendor max 120 N.m, K1).
C: plant-settle check: torso z of q_settled for box 36/32 kg at both dt (the claimed 0.5 ms settle artifact).
Output results/overnight/verify_k2/verify_k2.json. usage: python verify_k2.py [workers]"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp, mujoco
import k2_lag as K
OUT = f"{K.ROOT}/results/overnight/verify_k2/verify_k2.json"


def job(a):
    import branch_i as BI
    pk = {"knee": 0.0, "all": 0.0, "all_name": None}; orig = BI.backemf_brake
    def wrap(m, d, S, *aa, **kw):
        if "kn" not in pk:
            nm = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i][0])) or "" for i in range(m.nu)]
            pk["kn"] = [i for i in range(m.nu) if "knee" in nm[i]]; pk["body"] = [i for i in range(m.nu) if "hand" not in nm[i]]; pk["nm"] = nm
        f = np.abs(d.actuator_force)
        pk["knee"] = max(pk["knee"], float(f[pk["kn"]].max())); j = pk["body"][int(np.argmax(f[pk["body"]]))]
        if f[j] > pk["all"]: pk["all"] = float(f[j]); pk["all_name"] = pk["nm"][j]
        return orig(m, d, S, *aa, **kw)
    BI.backemf_brake = wrap
    try: key, r = K.job(a)
    finally: BI.backemf_brake = orig
    r["peak_knee_Nm"] = pk["knee"]; r["peak_body_Nm"] = pk["all"]; r["peak_body_joint"] = pk["all_name"]; return key, r


def settle(a):
    obj, mass, dt = a
    from engine import headcarry as HC, dynpred_headcarry as D
    D.register_objects(); S, _ = HC.plant(obj, mass, dt, fresh=True); m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d)
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link"); return f"settle|{obj}{mass:g}|{dt}", dict(torso_z=float(d.xpos[tb][2]), dt=float(m.opt.timestep))


if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 2; os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    zp = np.asarray(json.load(open(f"{K.OUT}/refine.json"))["pot_lag20ms"]["z"])
    zs = np.asarray(json.load(open(f"{K.ROOT}/results/overnight/Q1/lag_maxima.json"))["head_pot_lag10ms"]["lag_max_z"])[:14]
    rA = np.random.default_rng(4242); rB = np.random.default_rng(4243)
    ZA = [("nom", zp)] + [(f"p{i}", np.clip(zp + 0.03*rA.standard_normal(zp.size), -1, 1)) for i in range(2)]
    ZB = [("nom", zs)] + [(f"p{i}", np.clip(zs + 0.03*rB.standard_normal(zs.size), -1, 1)) for i in range(2)]
    jobs = []
    for dt in (0.001, 0.0005):
        jobs += [(f"A|pot44|20ms|{dt}|{t}", "head", "pot", 44.0, z.tolist(), dt, 0.020, K.COMP, True, True, 0.0) for t, z in ZA]
        jobs += [(f"B|stand|20ms|{dt}|{t}", "head", "pot", 2.0, z.tolist(), dt, 0.020, K.COMP, False, True, 0.0) for t, z in ZB]
    jobs = [j for j in jobs if j[0] not in out]
    sj = [(o, mm, dt) for o, mm in (("box", 36.0), ("box", 32.0)) for dt in (0.001, 0.0005) if f"settle|{o}{mm:g}|{dt}" not in out]
    with mp.get_context("fork").Pool(W, maxtasksperchild=2) as pool:
        for k, r in pool.imap_unordered(settle, sj):
            out[k] = r; print(k, r, flush=True); json.dump(K.J(out), open(OUT, "w"), indent=1)
        for k, r in pool.imap_unordered(job, jobs):
            out[k] = r; print(k, r["t_held"], r["reason"], "valid", r.get("valid_slip"), "unl", r.get("unloaded_ok"), "ft", r["foot_travel_max"], "pd", r["pelvis_drift"], "knee", round(r["peak_knee_Nm"], 1), "peak", round(r["peak_body_Nm"], 1), r["peak_body_joint"], flush=True)
            json.dump(K.J(out), open(OUT, "w"), indent=1)
