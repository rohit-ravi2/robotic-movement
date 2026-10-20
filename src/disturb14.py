"""Phase 14c step 2: disturbance robustness of the LANDED deep jumps (robust4, both robots), by REPLAY (no re-optimisation).
Pushes on the torso (0.1 s) at mid-flight (takeoff + 0.10 s) and just after touchdown (+0.02 s), forward/back/left/right,
0.15 / 0.30 / 0.45 body weight; plus floor friction 0.6 and 1.2 (model is 1.0). A replay counts if the jump still lands and
stands (same rule as jump14) at 1 ms. Output: results/phase14/disturb14.json."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, branch_i as BI, mocap_baseline as MB, aprior_standard as AS, balance as B, jump13 as J13
from multiprocessing import Pool
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase14"); G = 9.80665
_S = {}

def setup(robot):
    if robot not in _S:
        if robot == "G1": c = AS.clip0(); S = BI.setup(c["start"]); par = json.load(open(f"{OUT}/balance_tuning.json"))["best"]
        else:
            import aprior_h1 as AH; c = AH.clip0(); S = BI.setup(start=AH.Q0, robot="H1"); par = json.load(open(f"{OUT}/balance_tuning_h1.json"))["best"]
        BI.apply_dt(S, 0.001); _S[robot] = (c, S, par)
    return _S[robot]

def replay(robot, r, disturb=None, mu=None):
    c, S, par = setup(robot); m = S["m"]; ref = MB.analytic_clip(c, robot=robot)
    fl = S["floor"]; saved = m.geom_friction[fl].copy(); pair = [i for i in range(m.npair) if fl in (m.pair_geom1[i], m.pair_geom2[i])]
    saved_p = m.pair_friction[pair].copy()
    if mu is not None: m.geom_friction[fl, 0] = mu; m.pair_friction[pair, 0] = mu; m.pair_friction[pair, 1] = mu
    try:
        ctl = B.BalanceController(S, np.array(S["q_settled"]), params=par)
        fn = MB.Tracker(S, ref, omega=r["omega"], tscale=r["tscale"])
        o = BI.rollout(np.array(r["x"]), set(S["types"]), None, S=S, ctrl_fn=fn, horizon=J13.H_LAND, score="best", knot_dt=J13.KNOT_DT,
                       landing=dict(controller=ctl), disturb=disturb)
    finally:
        m.geom_friction[fl] = saved; m.pair_friction[pair] = saved_p
    L = o["landing"]; return dict(v=o["v"], ok=bool(J13.score(o) < 0), land_ok=L["ok"], t_to=o["mech"]["T_push"], td=L["touchdown_t"], tilt=L["max_tilt"], up=L["up_end"], z=L["z_end"], vend=L["v_end"])

def job(a):
    robot, r = a["robot"], a["r"]; t0 = time.time()
    nom = replay(robot, r); M = setup(robot)[1]["M"]; BW = M*G
    out = dict(robot=robot, seed=r["seed"], v=nom["v"], nominal=nom, cases=[])
    if not nom["ok"]: out["secs"] = time.time() - t0; return out
    t_fl = nom["t_to"] + 0.10; t_td = nom["td"] + 0.02
    for phase, t0_ in (("flight", t_fl), ("touchdown", t_td)):
        for dname, dvec in (("fwd", (1, 0, 0)), ("back", (-1, 0, 0)), ("left", (0, 1, 0)), ("right", (0, -1, 0))):
            for frac in (0.15, 0.30, 0.45):
                F = np.array(dvec, float)*frac*BW
                rr = replay(robot, r, disturb=[(t0_, 0.10, F)])
                out["cases"].append(dict(kind="push", phase=phase, dir=dname, bw=frac, **rr))
    for mu in (0.6, 1.2):
        rr = replay(robot, r, mu=mu); out["cases"].append(dict(kind="friction", mu=mu, **rr))
    out["secs"] = time.time() - t0; return out

if __name__ == "__main__":
    jobs = []
    for robot, f in (("G1", "jump14_robust4.json"), ("H1", "jump14_h1_robust4.json")):
        for r in json.load(open(f"{OUT}/{f}")):
            if r["ok_all"] and min(mm["v"] for mm in r["meas"]) >= 2.0: jobs.append(dict(robot=robot, r=r))
    if sys.argv[1:2] == ["smoke"]:
        a = jobs[0]; c, S, par = setup(a["robot"]); t = time.time(); print(replay(a["robot"], a["r"]), round(time.time() - t, 1), "s")
        print(replay(a["robot"], a["r"], disturb=[(0.9, 0.1, np.array([0.3*S["M"]*G, 0, 0]))])); sys.exit()
    with Pool(int(os.environ.get("NPROC", "4"))) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/disturb14.json", "w"), default=float)
    for robot in ("G1", "H1"):
        g = [r for r in res if r["robot"] == robot and r["nominal"]["ok"]]
        for phase in ("flight", "touchdown"):
            for frac in (0.15, 0.30, 0.45):
                cs = [c for r in g for c in r["cases"] if c["kind"] == "push" and c["phase"] == phase and c["bw"] == frac]
                print(f"{robot} push {phase:9s} {frac:.2f} BW: survived {sum(c['ok'] for c in cs)}/{len(cs)}", flush=True)
        for mu in (0.6, 1.2):
            cs = [c for r in g for c in r["cases"] if c["kind"] == "friction" and c["mu"] == mu]
            print(f"{robot} friction {mu}: survived {sum(c['ok'] for c in cs)}/{len(cs)}  v {sorted(round(c['v'], 2) for c in cs)}", flush=True)
