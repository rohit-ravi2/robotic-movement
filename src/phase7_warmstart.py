"""Phase 7 — warm-start dynamic solves from the analytic pose pair.

F-015 measured that raising per-seed CMA-ES budget 3.3x left seed spread unchanged (0.803 vs
0.805 m/s), and at 500 iterations with 6 seeds the full-body spread was 3.284 m/s. The spread is
seed multi-modality, not under-iteration, so per-seed budget is the wrong lever.

The pose bank already holds, per joint-group subset, an optimized crouch pose and takeoff pose.
Those are exactly the initialization these problems need: start the robot at the analytic crouch
and seed the control with the crouch-to-takeoff extension direction, so every seed begins in the
same basin instead of searching for one.

Permitted under Phase 7 Part IV ("warm start from the analytic pose pair"). No constraint is
relaxed and no objective is changed; only the initial guess.
"""
import sys, os, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, cma, branch_i as BI

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
G = 9.80665
T2G = {"ankle_pitch_joint": "ankles", "knee_joint": "knees", "hip_pitch_joint": "hips",
       "waist_pitch_joint": "waist", "shoulder_pitch_joint": "shoulders",
       "elbow_joint": "elbows"}

def analytic_pair():
    """Full-subset crouch and takeoff type-angles from the pose bank."""
    D = pickle.load(open(f"{ROOT}/results/pose_bank.pkl", "rb"))["G1"]
    full = max(D["bank"], key=lambda r: len(r["groups"]))
    return full["t0"], full["tT"]

def settled_from(t0):
    """Place G1 at the analytic crouch pose, drop to ground, settle under a PD hold."""
    S = BI.setup(); m = S["m"]
    rb = S["rb"]
    rb.set_pose(t0); rb.drop_to_ground()
    q = np.array(rb.d.qpos)
    d = mujoco.MjData(m); d.qpos[:] = q; d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(int(0.4/m.opt.timestep)):
        for t in S["types"]:
            for (jid, dof, qad, tau, om, sgn, aid) in S["tmap"][t]:
                d.ctrl[aid] = float(np.clip(80.0*(q[qad]-d.qpos[qad])-6.0*d.qvel[dof], -tau, tau))
        mujoco.mj_step(m, d)
    return d.qpos.copy(), d.qvel.copy()

def warm_control(t0, tT):
    """Control guess: drive each joint along the crouch-to-takeoff direction on a ramp."""
    S = BI.setup(); types = S["types"]
    x = np.zeros((len(types), BI.NK))
    ramp = np.clip(np.linspace(-0.1, 1.0, BI.NK), 0.0, 1.0)
    for i, t in enumerate(types):
        g = T2G.get(t)
        d0 = float(t0.get(g if g in t0 else t, 0.0)) if isinstance(t0, dict) else 0.0
        d1 = float(tT.get(g if g in tT else t, 0.0)) if isinstance(tT, dict) else 0.0
        # pose bank keys are group names via caps; fall back to the type key
        for key in (t, T2G.get(t, "")):
            if key in t0: d0 = float(t0[key])
            if key in tT: d1 = float(tT[key])
        s = np.sign(d1 - d0)
        x[i, :] = 0.7*s*ramp
    return x.ravel()

def solve_warm(seed, q_settled, v_settled, x0, iters=300, popsize=20, sigma=0.25, target=None):
    S = BI.setup()
    BI._S["q_settled"] = q_settled; BI._S["v_settled"] = v_settled
    def obj(x):
        r = BI.rollout(x, set(S["types"]))
        pen = 0.0
        if r["max_penetration"] > 0.012: pen += (r["max_penetration"]-0.012)**2*4.0e4
        if r["grf_spike_bw"] > 25.0: pen += (r["grf_spike_bw"]-25.0)**2*0.02
        return (-r["height"] + pen) if target is None else (abs(r["v"]-target)+pen)
    es = cma.CMAEvolutionStrategy(x0, sigma, {"bounds": [-1, 1], "seed": seed,
                                              "verbose": -9, "maxiter": iters, "popsize": popsize})
    while not es.stop():
        X = es.ask(); es.tell(X, [obj(x) for x in X])
    return es.result.xbest, BI.rollout(es.result.xbest, set(S["types"]))

if __name__ == "__main__":
    t0, tT = analytic_pair()
    print("analytic crouch keys:", sorted(t0), flush=True)
    q, v = settled_from(t0)
    x0 = warm_control(t0, tT)
    S = BI.setup()
    print(f"warm-start control norm {np.linalg.norm(x0):.2f} over {len(x0)} params", flush=True)
    base = BI.rollout(x0, set(S["types"]))
    print(f"warm guess before any optimization: v {base['v']:.3f} ht {base['height']*100:.1f} cm "
          f"valid {base['valid']} pen {base['max_penetration']*1000:.1f}mm", flush=True)
    out = []
    for seed in (1, 2, 3, 4, 5, 6):
        t = time.time()
        _, r = solve_warm(seed, q, v, x0, iters=300)
        out.append(dict(seed=seed, v=r["v"], height=r["height"], valid=bool(r["valid"]),
                        participation=r["participation"], secs=time.time()-t))
        print(f"  seed {seed}: v {r['v']:.3f} ht {r['height']*100:.1f} cm PR {r['participation']:.2f} "
              f"valid {r['valid']} [{time.time()-t:.0f}s]", flush=True)
    vv = np.array([o["v"] for o in out if o["valid"]])
    print(f"\nWARM START: best {vv.max():.3f} mean {vv.mean():.3f} SEED SPREAD {vv.max()-vv.min():.3f} m/s", flush=True)
    print(f"  cold start at 500 iters was: best 3.992 mean 2.541 spread 3.284", flush=True)
    json.dump(dict(runs=out, warm_guess=base["v"]), open(f"{ROOT}/results/warmstart.json", "w"),
              indent=1, default=float)
