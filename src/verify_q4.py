"""VERIFY Q4 predictor fixes (2026-10-17, adversarial). Fresh-plant replays (one spawned process per rollout, maxtasksperchild=1),
2 workers, at 1 ms and 0.5 ms, each with an independent runtime-plant diff vs the stock Menagerie model (verify_t0.phys/compare):
  medbox  Q4/head_medbox_45kg.json best_z: t2_gate.rollout_gated (slip + T1 support gate) exact + 2 perturbations (sigma 0.03) per dt,
          plus the law-off ablation at 1 ms
  sled    Q4/sled_push_mu0.5_6.5kg.json best_z, seed 0: sled_dyn.evaluate (fresh process, plant built at dt) exact + 2 perturbations per dt,
          instrumented with the standing-task support gate the sled runs did not apply (cumulative horizontal foot travel per foot,
          max horizontal pelvis displacement from t=0), plus foot-floor pair friction vs stock scene_mjx.xml
  probe   mechanism/probes.txt equal-mu*m*g pair (T2 mu0.6 8 kg push strategy at mu 0.4 / 12.5 kg and mu 0.25 / 20.6 kg), 1 ms
Output results/overnight/verify_q4/verify_q4.json."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import get_context

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); MEN = os.environ.get("MENAGERIE_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "models", "mujoco_menagerie"))
Q4 = f"{ROOT}/results/overnight/Q4"; OUT = f"{ROOT}/results/overnight/verify_q4"; os.makedirs(OUT, exist_ok=True)


def job(a):
    kind, dt, k, z, extra = a; t0 = time.time()
    import mujoco, verify_t0 as V
    if kind in ("medbox", "medbox_lawoff"):
        from engine import headcarry as HC, dynpred_headcarry_v2 as D2
        import t2_gate as G
        D2.register_objects()
        r = G.rollout_gated("medbox", 45.0, "head", np.asarray(z), 10.0, dt, law=(kind == "medbox"))
        S, path = HC.plant("medbox", 45.0, dt); stock = mujoco.MjModel.from_xml_path(f"{MEN}/unitree_g1/scene_with_hands.xml")
        dev = V.compare(V.phys(stock), V.phys(S["m"])); rat = V.ratings(S, stock)
        keep = {x: r.get(x) for x in ("t_held", "reason", "head_frac", "tilt_max_deg", "e_max", "foot_travel_max", "pelvis_drift", "foot_slide_max",
                                      "com_drift_max", "joint_sat", "act_lag", "gate_hc", "slip_ok", "valid_slip")}
        return dict(kind=kind, k=k, dt=dt, res=keep, plant_dev=dev, timestep=S["m"].opt.timestep,
                    ratings_mismatch={n: v for n, v in rat.items() if v["stock"] is not None and abs(v["tau_used"] - v["stock"]) > 1e-6}, secs=time.time() - t0)
    if kind in ("sled", "probe"):
        mu, mass = extra
        import leverage_probes as LP
        LP.SLED_MU = mu; LP._L3.clear()
        import sled_dyn as SD
        orig = LP.stand_rollout; sup = dict(prev=None, travel=[0.0, 0.0], p0=None, drift=0.0)

        def wrapped(S, zz, T, step_cb=None, **kw):
            m = S["m"]; fs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("left_foot", "right_foot")]
            pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

            def cb(t, d):
                p = [d.site_xpos[i][:2].copy() for i in fs]
                if sup["prev"] is not None:
                    for j in range(2): sup["travel"][j] += float(np.linalg.norm(p[j] - sup["prev"][j]))
                sup["prev"] = p
                if sup["p0"] is None: sup["p0"] = d.xpos[pel][:2].copy()
                sup["drift"] = max(sup["drift"], float(np.linalg.norm(d.xpos[pel][:2] - sup["p0"])))
                return step_cb(t, d) if step_cb else None
            return orig(S, zz, T, cb, **kw)
        LP.stand_rollout = wrapped
        r = SD.evaluate((mass, 1, np.asarray(z), 4.0, dt, False, 0))
        out = dict(kind=kind, k=k, dt=dt, mu=mu, payload=mass, res={x: r.get(x) for x in ("disp", "reason", "t_held", "com_lean_mean", "util_rms")},
                   foot_travel=sup["travel"], foot_travel_max=max(sup["travel"]), pelvis_drift=sup["drift"],
                   support_gate=bool(max(sup["travel"]) <= 0.02 and sup["drift"] <= 0.05), secs=time.time() - t0)
        if kind == "sled":
            S = LP.l3_plant(mass, dt); stock = mujoco.MjModel.from_xml_path(f"{MEN}/unitree_g1/scene_mjx.xml")
            out["plant_dev"] = V.compare(V.phys(stock), V.phys(S["m"])); rat = V.ratings(S, stock)
            out["ratings_mismatch"] = {n: v for n, v in rat.items() if v["stock"] is not None and abs(v["tau_used"] - v["stock"]) > 1e-6}
            pf = lambda mm: {mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_PAIR, i): mm.pair_friction[i].tolist() for i in range(mm.npair)}
            out["pairs_plant"] = pf(S["m"]); out["pairs_stock"] = pf(stock); out["timestep"] = S["m"].opt.timestep
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(20261018)
    zh = np.asarray(json.load(open(f"{Q4}/head_medbox_45kg.json"))["best_z"])
    zs = np.asarray(json.load(open(f"{Q4}/sled_push_mu0.5_6.5kg.json"))["best_z"])
    zp = np.asarray(json.load(open(f"{ROOT}/results/overnight/T2/sled_push_mu0.6_8kg.json"))["best_z"])
    pert = lambda z0: [("exact", z0)] + [(f"pert{i}", np.clip(z0 + 0.03*rng.standard_normal(z0.size), -1, 1)) for i in range(2)]
    jobs = [("medbox", dt, k, z.tolist(), None) for k, z in pert(zh) for dt in (0.001, 0.0005)]
    jobs += [("medbox_lawoff", 0.001, "exact", zh.tolist(), None)]
    jobs += [("sled", dt, k, z.tolist(), (0.5, 6.5)) for k, z in pert(zs) for dt in (0.001, 0.0005)]
    jobs += [("probe", dt, f"mu{mu}_{m}kg", zp.tolist(), (mu, m)) for mu, m in ((0.4, 12.5), (0.25, 20.6)) for dt in (0.001, 0.0005)]
    res = []
    with get_context("spawn").Pool(2, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(job, jobs):
            res.append(r); print(json.dumps({x: v for x, v in r.items() if x not in ("plant_dev", "pairs_plant", "pairs_stock")}, default=str)[:600],
                                 "| dev:", str(r.get("plant_dev"))[:200], flush=True)
            json.dump(res, open(f"{OUT}/verify_q4.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("done")
