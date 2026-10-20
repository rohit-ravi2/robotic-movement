"""Phase 8 dynamic campaign on the fully corrected G1 plant (D4-D8 + P8 measurement fix).
Every campaign: plant contract verified first (fail fast), manifest written before running,
every result stamped with lineage. Usage: python phase8_campaign.py <campaign>"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, branch_i as BI, contract_g1 as CG, lineage as LN, cma
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); OUT = f"{ROOT}/results/phase8"
CHAIN = ["ankle_pitch_joint", "knee_joint", "hip_pitch_joint", "waist_pitch_joint",
         "shoulder_pitch_joint", "elbow_joint"]
MULTISET = [2.2, 1.7, 1.3, 0.9, 0.6, 0.45]; MEAN = float(np.mean(MULTISET))


def morph(name):
    if name == "BASELINE": return None
    if name == "PROXIMAL_STRONG": o = list(MULTISET)
    elif name == "DISTAL_STRONG": o = list(reversed(MULTISET))
    elif name == "MIDDLE_STRONG": o = [0.6, 1.3, 2.2, 1.7, 0.9, 0.45]
    elif name == "UNIFORM": o = [MEAN]*6
    elif name.startswith("R="):                       # continuous proximal/distal ratio family
        r = float(name[2:]); z = np.linspace(1, -1, 6); k = np.log(2.2/0.45)/2
        e = np.exp(r*k*z); o = list(6*MEAN*e/e.sum())
    elif name.startswith("EQ:"):                       # EQ:<parent>:<type> equalise one group
        _, parent, t = name.split(":"); sc = morph(parent); sc[t] = (MEAN, MEAN); return sc
    elif name.startswith("FAC:"):                      # FAC:knee=S,hip=W  (S=2.2, W=0.45)
        sc = {t: (MEAN, MEAN) for t in CHAIN}
        for kv in name[4:].split(","):
            t, lv = kv.split("="); v = 2.2 if lv == "S" else 0.45
            sc[{"knee": "knee_joint", "hip": "hip_pitch_joint", "ankle": "ankle_pitch_joint"}[t]] = (v, v)
        return sc
    else: raise ValueError(name)
    return {t: (s, s) for t, s in zip(CHAIN, o)}


def cem(obj, n, seed, iters, popsize, elite=0.2, sigma0=0.35):
    """Independent optimizer: diagonal-Gaussian cross-entropy method (no covariance learning,
    no step-size adaptation), clipped to [-1, 1]."""
    rng = np.random.default_rng(seed); mu = np.zeros(n); sd = np.full(n, sigma0)
    best = (np.inf, None); ne = max(2, int(elite*popsize))
    for _ in range(iters):
        X = np.clip(mu + sd*rng.standard_normal((popsize, n)), -1, 1)
        f = np.array([obj(x) for x in X]); idx = np.argsort(f)[:ne]
        if f[idx[0]] < best[0]: best = (f[idx[0]], X[idx[0]].copy())
        mu = 0.7*mu + 0.3*X[idx].mean(0); sd = np.maximum(0.7*sd + 0.3*X[idx].std(0), 0.02)
    return best[1]


def set_dt(S, dt):
    """Optimisation timestep. 4 ms (the MJX default) let strong-ankle morphologies exploit
    contact numerics: replayed at 1 ms, PROXIMAL_STRONG optima lost 0.48 m/s on average. All
    mechanism campaigns optimise at 2 ms and are MEASURED by a 1 ms replay (replay_asym.py)."""
    if dt and abs(S["m"].opt.timestep - dt) > 1e-12:
        BI.apply_dt(S, dt)


def job(a):
    kind = a["kind"]; S = BI.setup(); sc = morph(a.get("morph", "BASELINE"))
    set_dt(S, a.get("dt"))
    t0 = time.time(); task = a.get("task", "T1")
    if a.get("optimizer", "cma") == "cem":
        n = len(S["types"])*BI.NK
        def obj(x):
            r = BI.rollout(x, set(S["types"]), sc); vr = r["validity"]
            pen = max(0, vr["contact_energy"]-vr["energy_tol"])*0.05 + max(0, vr["energy_residual"]-vr["energy_tol"])*0.05 + (0.5 if vr["nonfoot_contact"] else 0)
            return -r["height"] + pen
        x = cem(obj, n, a["seed"], a["iters"], a["popsize"])
        r = BI.rollout(x, set(S["types"]), sc, record=True)
    else:
        x, r = BI.solve(set(S["types"]), a["seed"], target=a.get("target"), caps_scale=sc,
                        iters=a["iters"], popsize=a["popsize"], task=task,
                        x0=a.get("x0"), sigma0=a.get("sigma0", 0.35))
    tr = r.pop("trace", None)
    out = dict(a, x0=None, secs=time.time()-t0, x=np.asarray(x).tolist(), **r)
    if tr is not None and a.get("keep_trace", True): out["trace"] = tr
    return out


def run(name, jobs, question, interpretation, workers=14):
    S = BI.setup()
    ver = {}
    for m_ in sorted({j.get("morph", "BASELINE") for j in jobs}):
        ver[m_] = CG.verify_or_die(S, morph(m_))      # fail fast on any plant-contract failure
    ph = CG.plant_hash(S)
    LN.manifest(f"{OUT}/{name}.manifest.json", question=question, robot="unitree_g1 29dof MJX (torque motors + implicit holds)",
                plant_hash=ph, contract_hashes={k: v["contract_hash"] for k, v in ver.items()},
                optimizer=sorted({(j.get("optimizer", "cma"), j["iters"], j["popsize"]) for j in jobs}),
                n_jobs=len(jobs), expected_interpretation=interpretation,
                acceptance="only results whose TrajectoryValidityReport is valid AND pass the PCTV-8 per-trajectory bound enter statistics")
    t0 = time.time()
    with Pool(workers) as p: res = p.map(job, jobs, chunksize=1)
    res = [LN.stamp(r, plant_hash=ph, asset=LN.file_hash(BI.MJCF)) for r in res]
    json.dump(dict(name=name, plant_hash=ph, secs=time.time()-t0, runs=res),
              open(f"{OUT}/{name}.json", "w"), default=float)
    print(f"{name}: {len(res)} solves in {time.time()-t0:.0f}s; valid {sum(r['valid'] for r in res)}", flush=True)
    return res


CAMPAIGNS = {}
def campaign(f): CAMPAIGNS[f.__name__] = f; return f

@campaign
def repro():
    jobs = [dict(kind="max", task=t, seed=s, iters=400, popsize=24) for t in ("T1", "T2") for s in range(1, 17)]
    run("repro", jobs, "Corrected G1 maximal jump: distribution over 16 seeds, tasks T1 (vertical takeoff velocity) and T2 (+ landable flight rotation)",
        "stable if valid fraction high and seed spread small relative to Phase 7's 27-38 cm claim")

@campaign
def pr_demand():
    tg = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0]
    jobs = ([dict(kind="demand", task="T1", target=t, seed=s, iters=300, popsize=20) for t in tg for s in range(1, 9)]
            + [dict(kind="demand", task="reserve", target=t, seed=s, iters=300, popsize=20) for t in tg for s in range(1, 9)])
    run("pr_demand", jobs, "Does effective dimensionality (participation ratio of push-off positive work) rise with demanded takeoff velocity? Two demand formulations: target-matching (as Phases 6-7) and target-as-constraint with torque-reserve cost",
        "null if the demand-linked change is within seed spread under both formulations and across window/metric variants")

@campaign
def asym_budget():
    jobs = [dict(kind="max", task="T1", morph=c, seed=s, iters=it, popsize=20)
            for c in ("PROXIMAL_STRONG", "DISTAL_STRONG", "MIDDLE_STRONG", "UNIFORM")
            for it in (25, 50, 100, 200, 400, 800, 1600) for s in range(1, 9)]
    jobs.sort(key=lambda j: -j["iters"])               # longest first for load balance
    run("asym_budget", jobs, "Proximal-strong vs distal-strong at matched capacity multiset: mechanical ceiling or optimizer accessibility? Accessibility curves over a 64x budget range",
        "A: prox better at all budgets, both plateau (mechanical); B: distal catches up; C: gap shrinks; D: unstable")

@campaign
def asym_budget_v2():
    jobs = [dict(kind="max", task="T1", morph=c, seed=s, iters=it, popsize=20, dt=0.002)
            for c in ("PROXIMAL_STRONG", "DISTAL_STRONG", "MIDDLE_STRONG", "UNIFORM")
            for it in (25, 100, 400, 1600) for s in range(1, 7)]
    jobs.sort(key=lambda j: -j["iters"])
    run("asym_budget_v2", jobs, "Asymmetry accessibility curves, optimised at dt=2 ms (4 ms let strong-ankle bodies exploit contact numerics); measured by 1 ms replay",
        "A/B/C/D on dt-converged takeoff velocities")

@campaign
def asym_cem():
    jobs = [dict(kind="max", task="T1", morph=c, seed=s, iters=400, popsize=20, optimizer="cem", dt=0.002)
            for c in ("PROXIMAL_STRONG", "DISTAL_STRONG", "UNIFORM", "BASELINE") for s in range(1, 7)]
    run("asym_cem", jobs, "Does the morphology ordering survive an independent optimizer (diagonal CEM)?", "ordering preserved or not")

@campaign
def ablations():
    jobs = []
    for parent in ("PROXIMAL_STRONG", "DISTAL_STRONG"):
        for t in CHAIN:
            jobs += [dict(kind="max", task="T1", morph=f"EQ:{parent}:{t}", seed=s, iters=300, popsize=20, dt=0.002) for s in range(1, 5)]
    for k in ("S", "W"):
        for h in ("S", "W"):
            jobs += [dict(kind="max", task="T1", morph=f"FAC:knee={k},hip={h}", seed=s, iters=300, popsize=20, dt=0.002) for s in range(1, 5)]
    run("ablations", jobs, "Per-group equalisation from PROX and DIST; 2x2 knee x hip factorial with others uniform", "additive vs interacting group effects")

@campaign
def spec_map():
    jobs = [dict(kind="max", task="T1", morph=f"R={r:+.2f}", seed=s, iters=300, popsize=20, dt=0.002)
            for r in np.linspace(-1, 1, 9) for s in range(1, 5)]
    run("spec_map", jobs, "Continuous proximal/distal capacity ratio at fixed capacity sum: does movement change smoothly and interpretably with the specification?", "smooth descriptor changes vs phase transitions")

@campaign
def pr_duration():
    jobs = [dict(kind="demand", task=("duration", Tc), target=t, seed=s, iters=300, popsize=20)
            for t in (0.5, 1.1, 1.7) for Tc in (0.06, 0.10, 0.14) for s in range(1, 5)]
    run("pr_duration", jobs, "Does PR track DEMAND or push-off DURATION? Target velocity crossed with a prescribed push-off time",
        "if PR follows duration at fixed demand, the demand-PR slope is a duration effect")

if __name__ == "__main__":
    for c in sys.argv[1:]: CAMPAIGNS[c]()
