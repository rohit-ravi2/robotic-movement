"""Phase 9 FAITHFUL replay of every stored Phase 8 control (defects A1, A2, A4 fixed):
  * the control's own optimisation-time knot schedule (phase8_schedule), not a dt-warped one
  * 1 ms timestep with held-servo gains rebuilt (apply_dt) and the start re-settled
  * exact 40 ms first-order lag; flight = zero foot force sustained 20 ms
Gate for reporting: every physics check (START, ENERGY, FOOT_ONLY, TOOK_OFF, MOMENTUM);
CONTACT_ENERGY is recorded, not gating (the 1 ms replay is the test it approximates)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import branch_i as BI, phase8_campaign as PC, mocap_baseline as MB
from multiprocessing import Pool
P8 = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase8"); OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase9")
DTOPT = dict(repro=0.004, pr_demand=0.004, pr_duration=0.004, asym_budget=0.004, asym_budget_v2=0.002,
             asym_cem=0.002, ablations=0.002, spec_map=0.002, baseline=0.004, baseline_budget400=0.004)
_ready = {}

def S_for(start=None):
    S = BI.setup(start)
    key = id(S)
    if key not in _ready: BI.apply_dt(S, 0.001); _ready[key] = True
    return S

def one(args):
    name, r = args
    dto = r.get("dt") or DTOPT[name]
    if name.startswith("baseline"):
        c = MB.load_clip(r["clip"], "retargeted_g1_all.pkl"); S = S_for(c["start"])
        x = np.array(r["x"]) if "x" in r else np.zeros(len(S["types"])*c["nk"])
        nk = len(x)//len(S["types"]); kd = BI.phase8_schedule(c["horizon"], dto, nk)
        om = r.get("omega", 0.0) if r["method"] != "OURS" else 0.0
        o = MB.run(S, c, x, om, r.get("tscale", 1.0), knot_dt=kd)
        extra = dict(clip=r["clip"], method=r["method"], lam=r.get("lam"), task=r.get("task"), set=r.get("set"),
                     tscale=r.get("tscale"), deviation=o["deviation"])
    else:
        S = S_for(); x = np.array(r["x"]); nk = len(x)//len(S["types"])
        kd = BI.phase8_schedule(BI.HOR, dto, nk)
        o = BI.rollout(x, set(S["types"]), PC.morph(r.get("morph", "BASELINE")), knot_dt=kd)
        extra = dict(morph=r.get("morph", "BASELINE"), task=str(r.get("task")), target=r.get("target"))
    ok = not [z for z in o["validity"]["reasons"] if z != "CONTACT_ENERGY"]
    return dict(name=name, seed=r.get("seed"), iters=r.get("iters"), v8=r["v"], v=o["v"], valid=bool(ok and not o.get("bound_violation")),
                reasons=o["validity"]["reasons"], mom=o["validity"]["momentum_error"], PR=o["participation"],
                work=o["work"], rot=o["mech"]["flight_rotation"], T_push=o["mech"]["T_push"], n_flights=o["n_flights"], **extra)

if __name__ == "__main__":
    names = sys.argv[1:] or list(DTOPT)
    for n in names:
        p = f"{P8}/{n}.json"
        if not os.path.exists(p): continue
        runs = json.load(open(p)); runs = runs["runs"] if isinstance(runs, dict) else runs
        with Pool(int(os.environ.get("NPROC", "12"))) as pool: out = pool.map(one, [(n, r) for r in runs], chunksize=1)
        json.dump(out, open(f"{OUT}/{n}_faithful.json", "w"), indent=1, default=float)
        v8 = np.array([o["v8"] for o in out]); v = np.array([o["v"] for o in out])
        print(f"{n}: {len(out)} runs, valid {sum(o['valid'] for o in out)}; v_faithful - v_phase8: median {np.median(v-v8):+.3f} [p10 {np.quantile(v-v8,.1):+.3f}, p90 {np.quantile(v-v8,.9):+.3f}]", flush=True)
