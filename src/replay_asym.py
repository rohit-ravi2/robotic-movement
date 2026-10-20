"""Replay every stored asym_budget solution through the CORRECTED rollout (floor-only contacts,
momentum convention) to size the measurement bias. Optima are NOT re-optimised here."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import branch_i as BI, phase8_campaign as PC
from multiprocessing import Pool
R = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase8")
DT = float(sys.argv[2]) if len(sys.argv) > 2 else None
SRC = sys.argv[1] if len(sys.argv) > 1 else "asym_budget"
d = json.load(open(f"{R}/{SRC}.json"))["runs"]
_init = {}
def rep(r):
    S = BI.setup()
    if DT and not _init:
        # dt-converged MEASUREMENT: replay the stored control at a finer timestep
        S["m"].opt.timestep = DT; S["m"].opt.iterations = 50
        for k in ("q_settled", "v_settled", "a_hold"): S.pop(k, None)
        BI.settle(S); _init[1] = True
    o = BI.rollout(np.array(r["x"]), set(S["types"]), PC.morph(r.get("morph", "BASELINE")))
    return dict(morph=r.get("morph", "BASELINE"), task=str(r.get("task")), target=r.get("target"), iters=r["iters"], seed=r["seed"], v_old=r["v"], valid_old=r["valid"],
                v_new=o["v"], valid_new=o["valid"], reasons=o["validity"]["reasons"], mom=o["validity"]["momentum_error"],
                PR_old=r.get("participation"), PR_new=o["participation"], work_new=o["work"],
                flight_rotation=o["mech"]["flight_rotation"], T_push=o["mech"]["T_push"])
if __name__ == "__main__":
    with Pool(8) as p: out = p.map(rep, d)
    json.dump(out, open(f"{R}/{SRC}_replay{'_dt'+str(DT) if DT else ''}.json", "w"), indent=1, default=float)
    for c in sorted({o["morph"] for o in out}):
        g = [o for o in out if o["morph"] == c]
        dv = np.array([o["v_new"] - o["v_old"] for o in g])
        print(f"{c:<16} valid old {sum(o['valid_old'] for o in g)}/{len(g)} new {sum(o['valid_new'] for o in g)}/{len(g)}  v_new - v_old: mean {dv.mean():+.3f} max {dv.max():+.3f}")
        for key in sorted({(o["iters"], o["task"], o["target"]) for o in g}, key=str):
            gg = [o for o in g if (o["iters"], o["task"], o["target"]) == key]
            v = [o["v_new"] for o in gg if o["valid_new"]]; vo = [o["v_old"] for o in gg]
            print(f"     {key}: old median {np.median(vo):.3f} -> dt-converged valid {len(v)}/{len(gg)} median {np.median(v) if v else float('nan'):.3f} best {max(v) if v else float('nan'):.3f}")
