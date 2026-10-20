"""Phase 10 C3 (Phase 8 sections 51-53): RE-OPTIMISATION under perturbed physics. Do the NS-04
(analytic prior >> native) and NS-02 (native ~ mocap-seeded) orderings survive friction, torque-limit,
friction-impedance (impratio) and joint-limit-softness changes? Each variant is applied to the plant
BEFORE settle and optimisation, and measured at 1 ms under the same variant. Model state is restored
after every job (workers are reused)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, branch_i as BI, mocap_baseline as MB
from multiprocessing import Pool
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase10"); PKL = "retargeted_g1_all.pkl"
IT = int(os.environ.get("SENS_ITERS", "100"))
CLIPS = ["jumps1_subject1@1166", "obstacles5_subject3@1585"]
VARIANTS = {"base": {}, "mu0.6": {"mu": 0.6}, "mu1.2": {"mu": 1.2}, "tau0.85": {"tau": 0.85}, "tau1.15": {"tau": 1.15},
            "limit_tc0.05": {"limit_tc": 0.05}}
# impratio 10 (elliptic) is EXCLUDED: the G1 MJX plant cannot settle under it (max|qdot| 27 rad/s at 1 ms,
# 4 rad/s at 0.5 ms vs 0.0000 at the model's impratio 1) -> an unstable instrument, not a physics variant.

def apply(S, v):
    m = S["m"]; fl = S["floor"]; saved = dict(pf=m.pair_friction.copy(), js=m.jnt_solref.copy(), ir=m.opt.impratio, cone=m.opt.cone)
    fp = [i for i in range(m.npair) if fl in (m.pair_geom1[i], m.pair_geom2[i])]
    if "mu" in v: m.pair_friction[fp, 0] = v["mu"]; m.pair_friction[fp, 1] = v["mu"]
    if "impratio" in v: m.opt.impratio = v["impratio"]; m.opt.cone = 1
    if "limit_tc" in v: m.jnt_solref[:, 0] = v["limit_tc"]
    return saved

def restore(S, saved):
    m = S["m"]; m.pair_friction[:] = saved["pf"]; m.jnt_solref[:] = saved["js"]; m.opt.impratio = saved["ir"]; m.opt.cone = saved["cone"]

def job(a):
    v = VARIANTS[a["variant"]]; c = MB.load_clip(a["clip"], PKL)
    S = BI.setup(c["start"]); BI.apply_dt(S, 0.004); saved = apply(S, v)
    S.pop("q_settled", None); BI.settle(S)             # re-settle under the variant
    orig = BI.rollout
    if "tau" in v:
        caps = {t: (v["tau"], 1.0) for t in S["types"]}
        BI.rollout = lambda params, act, cs=None, *aa, **kk: orig(params, act, caps, *aa, **kk)
    try:
        ref = MB.analytic_clip(c) if a["method"] == "APRIOR" else c; t0 = time.time()
        if a["method"] == "NATIVE":
            om = 0.0; r = MB.solve_b1(S, c, 0.0, a["seed"], 0.0, IT, 16, task="T2", native=True); ts = 1.0
        else:
            om, _ = MB.tune_gains(S, ref, 1.0); r = MB.solve_b1(S, ref, 0.0, a["seed"], om, IT, 16, task="T2"); ts = r["tscale"]
        x = np.array(r["x"]); BI.apply_dt(S, 0.001); S.pop("q_settled", None); BI.settle(S)
        o = MB.run(S, ref, x, om, ts)
        ok = not [z for z in o["validity"]["reasons"] if z != "CONTACT_ENERGY"] and not o.get("bound_violation")
        return dict(a, v=o["v"], valid=bool(ok), reasons=o["validity"]["reasons"], rot=o["mech"]["flight_rotation"],
                    grf=o["grf_bw"], secs=time.time()-t0)
    finally:
        BI.rollout = orig; restore(S, saved); BI.apply_dt(S, 0.004); S.pop("q_settled", None)

if __name__ == "__main__":
    jobs = [dict(clip=cl, method=m, variant=vn, seed=s) for vn in VARIANTS for cl in CLIPS for m in ("APRIOR", "NATIVE", "B1") for s in range(1, 5)]
    with Pool(int(os.environ.get("NPROC", "12"))) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/sens10.json", "w"), default=float)
    for vn in VARIANTS:
        line = [vn]
        for m in ("APRIOR", "NATIVE", "B1"):
            g = [r["v"] for r in res if r["variant"] == vn and r["method"] == m and r["valid"]]
            line.append(f"{m} {np.median(g) if g else float('nan'):.2f} ({len(g)}/8)")
        print(" | ".join(line), flush=True)
