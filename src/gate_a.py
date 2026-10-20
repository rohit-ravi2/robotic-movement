"""Branch G — Gate A: squat jump on the planar human model, CMA-ES, against published values."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, cma, human_planar as HP
import plant as P

G = 9.80665
NK = 14           # control knots; horizon 0.42 s -> 30 ms spacing
HOR = 0.42

CROUCH = [-1.20, 2.00, -0.50, -0.60, 0.60]   # balance-feasible; from the 295-point scan

def rollout(h, params, record=False, lock=None, squat=True, start=None):
    m, d = h.m, h.d
    q0 = h.place(CROUCH if start is None else start)
    nst = int(HOR/m.opt.timestep); per = nst//NK
    kn = np.asarray(params).reshape(5, NK)
    if lock is not None:
        for i in lock: kn[i, :] = 0.0
    a = np.zeros(5); alpha = m.opt.timestep/h.act_tau
    com0 = h.com()[2]
    peak_v = -1e9; peak_grf = 0.0; min_com = com0
    wlog = np.zeros((nst, 5)); tlog = np.zeros((nst, 5)); work = np.zeros(5)
    airborne = -1.0
    for s in range(nst):
        k = min(s//per, NK-1); k2 = min(k+1, NK-1); f = (s % per)/per
        cmd = kn[:, k] + (kn[:, k2]-kn[:, k])*f
        a += alpha*(cmd - a)
        th = np.array([d.qpos[i] for i in h.qadr]); w = np.array([d.qvel[i] for i in h.vadr])
        lim = h.tau*h.f_angle(th)
        sp = np.clip(1.0 - np.abs(w)/h.wmax, 0.0, 1.0)
        hi = lim*np.where(w > 0, sp, 1.0); lo = lim*np.where(w < 0, sp, 1.0)
        tau = np.clip(a*h.tau, -lo, hi)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        w = np.array([d.qvel[i] for i in h.vadr])
        work += np.maximum(tau*w, 0.0)*m.opt.timestep
        wlog[s] = w; tlog[s] = tau
        c = h.com()
        vcz = float(P.com_velocity(h.m, d)[2])   # P8: exact COM velocity, not cvel average
        peak_v = max(peak_v, vcz)
        min_com = min(min_com, c[2])
        fz = 0.0
        for ci in range(d.ncon):
            fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv); fz += fv[0]
        peak_grf = max(peak_grf, fz)
        if fz < 1.0 and airborne < 0 and s > 30: airborne = s*m.opt.timestep
    dip = com0 - min_com
    height = max(0.0, peak_v)**2/(2*G)
    return dict(height=height, v=peak_v, dip=dip, peak_grf_bw=peak_grf/(h.M*G),
                work=work, wlog=wlog if record else None, tlog=tlog if record else None,
                airborne=airborne, com0=com0)

def objective(h, x, lock=None, squat=True):
    r = rollout(h, x, lock=lock, squat=squat)
    pen = 0.0
    if squat and r["dip"] > 0.01: pen += (r["dip"]-0.02)**2*200.0   # squat jump: no countermovement
    return -r["height"] + pen

def solve(h, seed, lock=None, squat=True, iters=260, sigma=0.35):
    es = cma.CMAEvolutionStrategy(np.zeros(5*NK), sigma,
                                  {"bounds": [-1, 1], "seed": seed, "verbose": -9,
                                   "maxiter": iters, "popsize": 16})
    while not es.stop():
        X = es.ask(); es.tell(X, [objective(h, x, lock, squat) for x in X])
    x = es.result.xbest
    return x, rollout(h, x, record=True, lock=lock, squat=squat)


def bobbert_test(h, seeds, scale=1.5, iters=200):
    """Direction-only: strengthen every actuator WITHOUT re-optimizing the control, and check
    whether jump height falls. Published direction (Bobbert & van Soest 1994); magnitude was
    never verified so no tolerance is invented."""
    out = []
    for s in seeds:
        x, r0 = solve(h, s, iters=iters)
        hs = HP.Human(tau_scale=[scale]*5)
        r1 = rollout(hs, x)                       # same control, stronger actuators
        _, r2 = solve(hs, s, iters=iters)          # re-optimized for comparison
        out.append(dict(seed=s, base=r0["height"], strong_same_control=r1["height"],
                        strong_reoptimized=r2["height"]))
    return out

def xcorr_lag(a, b, dt, maxlag=0.12):
    a = a-a.mean(); b = b-b.mean()
    n = int(maxlag/dt); best = (0, -1e9)
    for L in range(-n, n+1):
        if L < 0: x, y = a[-L:], b[:len(b)+L]
        elif L > 0: x, y = a[:len(a)-L], b[L:]
        else: x, y = a, b
        if len(x) < 10: continue
        c = float(np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-12))
        if c > best[1]: best = (L*dt, c)
    return best

if __name__ == "__main__":
    h = HP.Human()
    SEEDS = [1, 2, 3, 4, 5]
    res = {"seeds": SEEDS, "nk": NK, "horizon": HOR, "mass": h.M}
    t0 = time.time(); runs = []
    for s in SEEDS:
        x, r = solve(h, s)
        runs.append(dict(seed=s, height=r["height"], v=r["v"], dip=r["dip"],
                         grf=r["peak_grf_bw"], work=r["work"].tolist(),
                         airborne=r["airborne"], x=x.tolist()))
        print(f"  seed {s}: height {r['height']*100:.1f} cm  v {r['v']:.2f}  dip {r['dip']*100:.1f} cm  "
              f"GRF {r['peak_grf_bw']:.2f} BW  [{time.time()-t0:.0f}s]", flush=True)
    res["squat_jump"] = runs
    best = max(runs, key=lambda z: z["height"])
    # G.1 sequencing from the best run
    _, rb = solve(h, best["seed"])
    W = rb["wlog"]; dt = h.m.opt.timestep
    seq = {}
    for i, (a, b) in enumerate([(0, 1), (1, 2)]):      # hip->knee, knee->ankle
        lag, c = xcorr_lag(W[:, a], W[:, b], dt)
        seq[f"{HP.JOINTS[a]}->{HP.JOINTS[b]}"] = dict(lag_s=lag, corr=c)
    res["sequencing"] = seq
    # G.2 arm contribution: lock shoulder+elbow
    arm = []
    for s in SEEDS[:3]:
        _, r = solve(h, s, lock=[3, 4])
        arm.append(r["height"])
    res["arms_locked_height"] = arm
    # G.3 CMJ: allow dip
    cmj = []
    for s in SEEDS[:3]:
        _, r = solve(h, s, squat=False)
        cmj.append(dict(height=r["height"], dip=r["dip"]))
    res["cmj"] = cmj
    res["bobbert"] = bobbert_test(h, SEEDS[:3])
    json.dump(res, open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/gate_a.json"), "w"),
              indent=1, default=float)
    print("wrote gate_a.json", flush=True)
