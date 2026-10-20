"""Phase 14 final: the jump WITH the balance/landing controller. Push-off re-optimised so that only jumps that land
(feet-only contact after takeoff, upright, standing, settled at 2.6 s) count. APRIOR (deep-crouch prior) vs NATIVE,
optimised at 1 ms with the tuned controller (results/phase14/balance_tuning.json), validated at 1 ms AND 0.5 ms."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cma, branch_i as BI, mocap_baseline as MB, aprior_standard as AS, balance as B, jump13 as J13
from multiprocessing import Pool
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase14")
ROBOT = os.environ.get("ROBOT", "G1"); TAG = "" if ROBOT == "G1" else "_" + ROBOT.lower()
PAR = json.load(open(os.environ.get("PAR_FILE", f"{OUT}/balance_tuning{TAG}.json")))["best"]      # each robot's own tuned landing controller

def clip0():
    if ROBOT == "G1": return AS.clip0()
    import aprior_h1; return aprior_h1.clip0()

def setup(c):
    return BI.setup(c["start"]) if ROBOT == "G1" else BI.setup(start=c["start"], robot="H1")

MARGIN = float(os.environ.get("STOP_MARGIN", "0")) or None      # robust5: hardware joint-stop margin (rad) during push-off
SLIP = float(os.environ.get("SLIP_CAP", "0")) or None           # robust6: no-slip push-off, |f_t| <= SLIP * mu * f_n

def run(S, ref, x, om, ts, ctl):
    fn = MB.Tracker(S, ref, omega=om, tscale=ts) if om > 0 else None
    return BI.rollout(x, set(S["types"]), None, S=S, ctrl_fn=fn, horizon=J13.H_LAND, score="best", knot_dt=J13.KNOT_DT,
                      landing=dict(controller=ctl), stop_margin=MARGIN, slip_cap=SLIP)

def job(a):
    t0 = time.time(); c = clip0(); S = setup(c); BI.apply_dt(S, 0.001)
    ref = MB.analytic_clip(c, robot=ROBOT) if a["method"] == "APRIOR" else c
    om = MB.tune_gains(S, ref, 1.0)[0] if a["method"] == "APRIOR" else 0.0
    ctl = B.BalanceController(S, np.array(S["q_settled"]), params=PAR)
    nT = len(S["types"]); n = nT*ref["nk"] + (1 if a["method"] == "APRIOR" else 0)
    unpack = lambda z: (z[:nT*ref["nk"]], 0.87 + 0.3*z[-1] if a["method"] == "APRIOR" else 1.0)
    robust = a.get("robust", False)
    if robust:                                            # a SEPARATE 0.5 ms plant, built once: a jump must land under BOTH
        S2 = BI._build(c["start"], {}, ROBOT); BI.apply_dt(S2, 0.0005)
        ref2 = MB.analytic_clip(c, robot=ROBOT) if a["method"] == "APRIOR" else c
        ctl2 = B.BalanceController(S2, np.array(S2["q_settled"]), params=PAR)
    def obj(z):
        x, ts = unpack(z); o1 = run(S, ref, x, om, ts, ctl); sc = J13.score(o1)
        if not robust: return sc
        o2 = run(S2, ref2, x, om, ts, ctl2); sc2 = J13.score(o2)
        if a.get("floor_always"):                          # robust4: the takeoff floor applies to EVERY solution, landed or not
            vmin = min(o1["v"], o2["v"]); floor = a["floor"]
            if vmin < floor: return 200.0 + (floor - vmin) + (0.0 if sc < 0 else sc - 100.0) + (0.0 if sc2 < 0 else sc2 - 100.0)
        if sc < 0 and sc2 < 0:                              # both land: score by the WORSE takeoff (robust2 fix: min() rewarded the better one)
            worst = max(sc, sc2); floor = a.get("floor")
            # robust3: hop basin rejected (landed at both dt but takeoff < floor). The penalty must exceed every
            # landing-failure score (100 + viol, up to ~125) because CMA-ES is RANK-based: at 50 the first robust3
            # run was byte-identical to robust2 (hops still ranked above deep jumps that failed to land).
            if floor and -worst < floor: return 200.0 + (floor + worst)
            return worst
        return 100.0 + (0.0 if sc < 0 else sc - 100.0) + (0.0 if sc2 < 0 else sc2 - 100.0)
    x0, sig = np.zeros(n), 0.35
    if a.get("warm"):                                     # robust2: refine the 1 ms-landed jump14 solution of the same seed (stay in the deep basin)
        WF = os.environ.get("WARM_FILE", f"{OUT}/jump14{TAG}.json")        # H1 robust3: warm from Phase 10 aprior_h1.json (T2H)
        W = [r for r in json.load(open(WF)) if r["method"] == a["method"] and r["seed"] == a["seed"] and r.get("task", "T2H") == "T2H"][0]
        xw = np.asarray(W["x"]); x0 = np.concatenate([xw, [(W["tscale"] - 0.87)/0.3]]) if a["method"] == "APRIOR" else xw
        x0 = np.clip(x0, -1, 1); sig = a.get("sigma", 0.10)
    es = cma.CMAEvolutionStrategy(x0, sig, {"bounds": [-1, 1], "seed": a["seed"], "verbose": -9, "maxiter": a["iters"], "popsize": 16})
    while not es.stop():
        X = es.ask(); es.tell(X, [obj(z) for z in X])
    x, ts = unpack(np.asarray(es.result.xbest)); out = []
    for dt in (0.001, 0.0005):
        BI.apply_dt(S, dt); ctl_v = B.BalanceController(S, np.array(S["q_settled"]), params=PAR)
        o = run(S, ref, x, om, ts, ctl_v); out.append(dict(dt=dt, v=o["v"], ok=J13.score(o) < 0, landing=o["landing"],
                                                           reasons=o["validity"]["reasons"], crouch=o["mech"]["crouch_depth"], rot=o["mech"]["flight_rotation"]))
    BI.apply_dt(S, 0.001)
    return dict(a, x=np.asarray(x).tolist(), omega=om, tscale=ts, meas=out, ok_all=all(m["ok"] for m in out), v=out[0]["v"], secs=time.time()-t0)

if __name__ == "__main__":
    if sys.argv[1:2] == ["smoke"]:
        r = job(dict(method="APRIOR", seed=1, iters=2)); print("smoke", round(r["secs"]), "s", r["meas"]); sys.exit()
    mode = sys.argv[1] if sys.argv[1:2] else ""
    robust = mode.startswith("robust"); warm = robust and mode != "robust"
    lvl = int(mode[6:7]) if (robust and mode[6:7].isdigit()) else 0          # robust3, robust4, robust5(b), robust6 ...
    floor = float(os.environ.get("FLOOR", "2.0")) if lvl >= 3 else None
    iters = int(os.environ.get("ITERS", "0")) or (100 if warm else 200)
    jobs = [dict(method=m, seed=s, iters=iters, robust=robust, warm=warm, floor=floor, floor_always=(lvl >= 4))
            for m in (("APRIOR",) if robust else ("APRIOR", "NATIVE")) for s in range(1, 7)]
    with Pool(int(os.environ.get("NPROC", "12"))) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/jump14{TAG}{('_' + mode) if mode else ''}.json", "w"), default=float)
    for m in ("APRIOR", "NATIVE"):
        g = [r for r in res if r["method"] == m]; ok = [r["v"] for r in g if r["ok_all"]]
        print(f"{m}: landed at 1 AND 0.5 ms {len(ok)}/{len(g)}  takeoff of landed jumps {sorted(round(v, 2) for v in ok)}", flush=True)
