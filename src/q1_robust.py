"""Q1 robust rung (2026-10-17): the 10 ms lag maxima that were fragile under sigma 0.03 perturbation (pot 40, hands box 5, hands pot 7)
are searched one rung lower (q1_lag.search_mass, warm-started from the fragile solution) and the found solution gets the same 2 x 2
perturbation check. Output results/overnight/Q1/robust_rung.json. usage: python q1_robust.py [workers]"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
import q1_lag as Q, q1_extra as X

if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 4; L = json.load(open(f"{Q.OUT}/lag_maxima.json")); out = {}; rng = np.random.default_rng(20261018)
    with mp.get_context("fork").Pool(W, maxtasksperchild=8) as pool:
        for tag, st, obj, mass in (("head_pot", "head", "pot", 36.0), ("hands_box", "hands", "box", 4.0), ("hands_pot", "hands", "pot", 6.0)):
            r = Q.search_mass(pool, tag, st, obj, mass, 0.010, L[f"{tag}_lag10ms"]["lag_max_z"], 120, time.time() + 1800)
            o = dict(mass=mass, valid=r["valid"], evals=r["evals"], z=r["z"], wcap=r["wcap"], pert={})
            if r["valid"]:
                z0 = np.asarray(r["z"]); jobs = []
                for i in range(2):
                    zp = np.clip(z0 + 0.03*rng.standard_normal(z0.size), -1, 1).tolist()
                    jobs += [(f"{i}|{dt}", st, obj, mass, zp, dt, 0.010, Q.cap_of(zp)) for dt in (0.001, 0.0005)]
                jobs += [(f"lawoff|{dt}", st, obj, mass, r["z"], dt, 0.010, Q.cap_of(r["z"])) for dt in (0.001, 0.0005)]
                for k, rr in pool.map(X.job, jobs[:4]): o["pert"][k] = rr
                o["n_pert_valid"] = sum(bool(v.get("valid_slip")) for v in o["pert"].values())
            out[tag] = o; Q.log(f"R {tag} {mass:g} kg: valid {r['valid']} perturbations valid {o.get('n_pert_valid')}/4")
            json.dump(Q.J(out), open(f"{Q.OUT}/robust_rung.json", "w"), indent=1)
