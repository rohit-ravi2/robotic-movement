"""TRACK K1 (2026-10-18): G1 sprint under Unitree's PUBLISHED limits (SPRINT_RATINGS=published: min of MJCF peak, URDF effort and the
vendor's 120 N.m joint maximum -> knee 120, hip roll 88, ankle/waist r,p 35) with and without the opt-in FLIGHT gate
(SPRINT_FLIGHTGATE=0.040). Stock Menagerie G1 otherwise; no hardware change.

  python k1_sprint.py replay <ladder json> <rung ix> <lag> <ratings> <gate> <dt> <pert> [stride_from]  -> one JSON line (fresh process)
  python k1_sprint.py ladder <lag> <gate> <seed> <v_start> <workers> <iters> <pop>                     bidirectional ladder, warm from 3.21
  python k1_sprint.py robust <json list of (ladder json, ix)>                                         fresh-plant perturbation table

Warm start: z of results/sprint/ladder_G1_cont_lag2ms_c2.json rung 2 (3.21 m/s) with ITS stride (the planner's 2.0 m/s stride at knee 139,
extrapolated). Under knee 120 the ceiling planner itself has no feasible stride above 1.4 m/s, so every rung above 1.4 keeps that stride
(SPRINT_EXTRAP, flagged stride_extrapolated); the stride is a controller timing choice, the plant carries the published ratings.
Ladder: first rung v_start; if valid, rungs up by 0.25 until one fails; if not valid, rungs down by 0.25 (each warm from the last tried z)
until one is valid or 1.0 is reached."""
import os, sys, json, subprocess, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); PY = sys.executable
WARM = f"{ROOT}/results/sprint/ladder_G1_cont_lag2ms_c2.json"; WARM_IX = 2
OUT = f"{ROOT}/results/sprint/k1"


def warm_stride(SP, SC, v_from=2.0):
    """The 3.21 solution's stride: the ceiling planner's stride at v_from on the knee-139 body (identical under peak and URDF effort)."""
    return dict(SP.select_stride(SC.body_spec("G1"), v_from), extrapolated_from=v_from)


def child_replay(pth, ix, dt, pert, stride_from):
    import numpy as np
    from engine import sprint as SP, sprint_ceiling as SC
    R = json.load(open(pth))["rungs"][ix]
    st = warm_stride(SP, SC, float(R["stride"].get("extrapolated_from", stride_from))) if R.get("stride_extrapolated") else None
    _is = SP.initial_state
    def initial_state(S, Rr, p):
        q0, v0, ph = _is(S, Rr, p); q0 = np.array(q0, float); v0 = np.array(v0, float)
        if pert == "vx": v0[0] += 0.05
        if pert == "vxm": v0[0] -= 0.05
        if pert.startswith("qnoise"): q0[7:] += 2e-3*np.random.default_rng(int(pert[6:] or 1)).standard_normal(len(q0) - 7)
        return q0, v0, ph
    SP.initial_state = initial_state
    P = SP.Sprint("G1", R["v"], stride=st); S = P.S[dt]
    r = P.run(np.asarray(R["z"]), dt); s = SP.summarize(r)
    tau = {t: float(S["jarr"]["tau"][i]) for i, t in enumerate(S["types"]) if t.startswith("left_") and ("knee" in t or "hip" in t or "ankle" in t)}
    print("JSON" + json.dumps(dict(s, ratings=SP.RATINGS, act_tau=SP.ACT_TAU, gate=SP.FLIGHT_GATE, timestep=float(S["m"].opt.timestep), tau_left_leg=tau), default=float))


def replay(pth, ix, lag, ratings, gate, dt, pert, stride_from="2.0"):
    lag_ = lag*1.25 if pert == "lag125" else lag
    env = dict(os.environ, SPRINT_LAG=str(lag_), SPRINT_RATINGS=ratings, SPRINT_FLIGHTGATE=str(gate), OMP_NUM_THREADS="1")
    o = subprocess.run(["nice", "-n", "10", PY, os.path.abspath(__file__), "child", pth, str(ix), str(dt), pert, stride_from], cwd=f"{ROOT}/src", env=env, capture_output=True, text=True)
    L = [l for l in o.stdout.splitlines() if l.startswith("JSON")]
    return {**(json.loads(L[0][4:]) if L else dict(error=o.stderr[-600:])), **dict(src=os.path.basename(pth), rung=ix, lag=lag_, ratings_job=ratings, gate_job=gate, dt=dt, pert=pert)}


def ladder_bidir(lag, gate, seed, v0, workers, iters, pop):
    os.environ.update(SPRINT_LAG=str(lag), SPRINT_RATINGS="published", SPRINT_FLIGHTGATE=str(gate), SPRINT_EXTRAP="1", SPRINT_SEED=str(seed))
    sys.path.insert(0, f"{ROOT}/src")
    from engine import sprint as SP, sprint_ceiling as SC
    assert SP.RATINGS == "published" and abs(SP.ACT_TAU - lag) < 1e-12 and abs(SP.FLIGHT_GATE - gate) < 1e-12
    st0 = (warm_stride(SP, SC), 2.0); z = json.load(open(WARM))["rungs"][WARM_IX]["z"]
    cond = f"_K1_pub_lag{lag*1000:g}ms_{'gate' if gate > 0 else 'nogate'}_s{seed}"; log = []
    def rung(v, z_):
        r = SP.ladder("G1", [v], iters, pop, workers, tag=f"{cond}_r{v:g}", seed=seed, z_init=z_, st_init=st0)[0]
        log.append(dict(v=v, ok=r.get("ok_both_dt"), meas=[(m["value"], m["reasons"], m["flight_min_step"]) for m in r.get("meas", [])], file=f"ladder_G1{cond}_r{v:g}.json"))
        json.dump(dict(cond=cond, lag=lag, gate=gate, seed=seed, rungs=log), open(f"{OUT}/cond{cond}.json", "w"), indent=1); return r
    os.makedirs(OUT, exist_ok=True)
    r = rung(v0, z)
    if r.get("ok_both_dt"):
        v = v0
        while r.get("ok_both_dt"): z = r["z"]; v += 0.25; r = rung(v, z)
    else:
        v = v0; zl = r.get("z", z)
        while not r.get("ok_both_dt") and v > 1.0 + 1e-9: v -= 0.25; r = rung(v, zl); zl = r.get("z", zl)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "child":
        sys.path.insert(0, f"{ROOT}/src"); child_replay(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6])
    elif cmd == "replay":
        a = sys.argv[2:]; print(json.dumps(replay(a[0], int(a[1]), float(a[2]), a[3], float(a[4]), float(a[5]), a[6], *(a[7:8] or ["2.0"])), default=float))
    elif cmd == "ladder":
        a = sys.argv[2:]; ladder_bidir(float(a[0]), float(a[1]), int(a[2]), float(a[3]), int(a[4]), int(a[5]), int(a[6]))
    elif cmd == "robust":
        from concurrent.futures import ThreadPoolExecutor
        items = json.loads(sys.argv[2]); nw = int(sys.argv[3]) if len(sys.argv) > 3 else 4; jobs = []
        for pth, ix, lag, gate in items:
            for pert in ("none", "vx", "vxm", "qnoise1", "qnoise2", "lag125"):
                for dt in (0.001, 0.0005): jobs.append((pth, ix, lag, "published", gate, dt, pert))
        with ThreadPoolExecutor(nw) as ex: res = list(ex.map(lambda j: replay(*j), jobs))
        for r in res: print(r["src"], r["dt"], r["pert"], r.get("value"), r.get("valid"), r.get("reasons"), r.get("flight_min_step"), r.get("error", "")[-200:], flush=True)
        json.dump(res, open(f"{OUT}/robust_{time.strftime('%H%M%S')}.json", "w"), indent=1, default=float)
