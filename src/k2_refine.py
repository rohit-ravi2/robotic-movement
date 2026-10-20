"""K2 bracket refinement: for each ladder (box/pot x 15/20 ms) one rung at max + 4 kg, warm-started from the max z, budget 48 evals
(pop 6); if valid: 3 perturbations (sigma 0.03, rng 20261018) at both dt and the law-off ablation. Output results/overnight/K2/refine.json."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
import k2_lag as K, k2_ladder as KL

if __name__ == "__main__":
    P = f"{K.OUT}/refine.json"; out = json.load(open(P)) if os.path.exists(P) else {}; t_end = time.time() + 3600*0.9
    with mp.get_context("fork").Pool(6, maxtasksperchild=6) as pool:
        for lag in (0.020, 0.015):
            for obj in ("box", "pot"):
                key = f"{obj}_lag{int(lag*1000)}ms"
                if key in out: continue
                L = json.load(open(f"{K.OUT}/ladder_{key}.json")); m = L["max"] + 4.0
                r = KL.rung(pool, obj, m, lag, L["max_z"], 48, t_end, pop=6); r["lag"] = lag
                if r["valid"]:
                    rng = np.random.default_rng(20261018); z0 = np.asarray(r["z"]); zs = [np.clip(z0 + 0.03*rng.standard_normal(z0.size), -1, 1).tolist() for _ in range(3)]
                    R = pool.map(KL.gated, [(obj, m, zp, dt, lag, True) for zp in zs for dt in (0.001, 0.0005)] + [(obj, m, r["z"], dt, lag, False) for dt in (0.001, 0.0005)], chunksize=1)
                    r["robust"] = dict(draws=[[KL.brief(R[2*i]), KL.brief(R[2*i+1])] for i in range(3)], n_both=sum(bool(R[2*i].get("valid_slip") and R[2*i+1].get("valid_slip")) for i in range(3)))
                    r["lawoff"] = dict(r1=R[-2], r05=R[-1])
                out[key] = r; json.dump(K.J(out), open(P, "w"), indent=1)
                K.log(f"R {key} {m:g} kg: valid {r['valid']} warm_ok {r['warm_ok']} evals {r['evals']} robust {r.get('robust', {}).get('n_both')}", "refine.status")
                ok = r["valid"]; os.system(f"~/bin/notify 'K2 refine {key} {m:g} kg: valid {ok}' 2>/dev/null")
