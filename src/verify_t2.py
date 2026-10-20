"""VERIFY T2 dynamic predictors (2026-10-17, adversarial): fresh-plant replays (one spawned process per rollout, maxtasksperchild=1)
of two headline T2 results at 1 ms and 0.5 ms from their stored parameters, with an independent runtime-plant diff vs the stock
Menagerie model (verify_t0.phys/compare):
  crate   head_crate_63kg.json: t2_gate.rollout_gated (slip + support gate), plus 2 small z perturbations (sigma 0.03) per dt
  h1      ladder_H1_T2_lag10ms.json rung 3.5 (stored z and stored extrapolated stride), SPRINT_LAG=0.010
  sled    plant diff only of the T2 sled plant (mu 0.6), not covered by the T0 verifier
Output results/overnight/verify_t2/verify_t2.json."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import get_context

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); MEN = os.environ.get("MENAGERIE_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "models", "mujoco_menagerie"))
OUT = f"{ROOT}/results/overnight/verify_t2"; os.makedirs(OUT, exist_ok=True)


def job(a):
    kind, dt, k, z = a; t0 = time.time()
    import mujoco, verify_t0 as V
    if kind == "crate":
        from engine import headcarry as HC, dynpred_headcarry as D
        import t2_gate as G
        D.register_objects()
        r = G.rollout_gated("crate", 63.0, "head", np.asarray(z), 10.0, dt)
        S, path = HC.plant("crate", 63.0, dt); stock = mujoco.MjModel.from_xml_path(f"{MEN}/unitree_g1/scene_with_hands.xml")
        dev = V.compare(V.phys(stock), V.phys(S["m"])); rat = V.ratings(S, stock)
        keep = {x: r.get(x) for x in ("t_held", "reason", "head_frac", "tilt_max_deg", "foot_travel_max", "pelvis_drift", "foot_slide_max", "com_drift_max", "joint_sat", "gate_hc", "slip_ok", "valid_slip")}
        return dict(kind=kind, k=k, dt=dt, res=keep, plant_dev=dev, timestep=S["m"].opt.timestep,
                    ratings_mismatch={n: v for n, v in rat.items() if v["stock"] is not None and abs(v["tau_used"] - v["stock"]) > 1e-6}, secs=time.time() - t0)
    if kind == "h1":
        os.environ["SPRINT_LAG"] = "0.010"; os.environ["SPRINT_EXTRAP"] = "1"
        from engine import sprint as SP
        L = json.load(open(f"{ROOT}/results/sprint/ladder_H1_T2_lag10ms.json")); R = [x for x in L["rungs"] if x["v"] == 3.5][0]
        P = SP.Sprint("H1", 3.5, stride=dict(R["stride"])); S = P.S[dt]
        stock = mujoco.MjModel.from_xml_path(f"{MEN}/unitree_h1/scene.xml")
        dev = V.compare(V.phys(stock), V.phys(S["m"])); rat = V.ratings(S, stock)
        s = SP.summarize(P.run(np.asarray(R["z"]), dt))
        return dict(kind=kind, k=k, dt=dt, act_tau=SP.ACT_TAU, res={x: s[x] for x in ("value", "valid", "reasons", "slip_max", "overspeed", "periodic")},
                    stored=[(m_["value"], m_["valid"]) for m_ in R["meas"]], plant_dev=dev,
                    ratings_mismatch={n: v for n, v in rat.items() if v["stock"] is not None and abs(v["tau_used"] - v["stock"]) > 1e-6}, secs=time.time() - t0)
    if kind == "sled":
        import leverage_probes as LP
        LP.SLED_MU = 0.6; LP._L3.clear(); S = LP.l3_plant(8.0, dt)
        stock = mujoco.MjModel.from_xml_path(f"{MEN}/unitree_g1/scene_mjx.xml")
        dev = V.compare(V.phys(stock), V.phys(S["m"])); rat = V.ratings(S, stock)
        return dict(kind=kind, k=k, dt=dt, plant_dev=dev, scene=S.get("scene"),
                    ratings_mismatch={n: v for n, v in rat.items() if v["stock"] is not None and abs(v["tau_used"] - v["stock"]) > 1e-6}, secs=time.time() - t0)


if __name__ == "__main__":
    C = json.load(open(f"{ROOT}/results/overnight/T2/head_crate_63kg.json")); z0 = np.asarray(C["best_z"]); rng = np.random.default_rng(20261017)
    zs = [("exact", z0)] + [(f"pert{i}", np.clip(z0 + 0.03*rng.standard_normal(z0.size), -1, 1)) for i in range(2)]
    jobs = [("sled", 0.001, "plant", None)] + [("h1", dt, "exact", None) for dt in (0.001, 0.0005)] + [("crate", dt, k, z.tolist()) for k, z in zs for dt in (0.001, 0.0005)]
    res = []
    with get_context("spawn").Pool(2, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(job, jobs):
            res.append(r); print(json.dumps({x: v for x, v in r.items() if x != "plant_dev"}, default=str)[:700], "| dev:", r["plant_dev"][:10], flush=True)
            json.dump(res, open(f"{OUT}/verify_t2.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("done")
