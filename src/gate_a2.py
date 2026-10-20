"""Gate A v2 — planar human squat jump, re-run with the Phase 8 measurement discipline.

Changes from gate_a.py (Phase 6), each a defect found on G1 that applies here too:
  * COM velocity from the exact COM Jacobian (P8; gate_a.py used the cvel average)
  * jump velocity = velocity AT TAKEOFF (first contact loss with upward COM velocity), not the
    peak over the whole horizon (which includes contact transients)
  * start from a HELD crouch: holding torques from the static-equilibrium LP, applied at t=0
    (gate_a.py started limp, so gravity supplied an uncontrolled countermovement)
  * momentum check (v_takeoff vs ground impulse) and energy accounting on every solve
Squat jump constraint (no dip > 1 cm below the held crouch) and literature targets unchanged."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, cma, human_planar as HP, plant as P
from multiprocessing import Pool

G = 9.80665; NK = 14; HOR = 0.42
FLIGHT_CONFIRM = 0.020   # s; same sustained-flight rule as branch_i (G1/H1)
CROUCH = [-1.17, 1.86, -0.69, -0.65, -0.75]  # Phase 8 robust certified crouch; elbow sign follows the P16 fix (same 0.75 rad flexion, now anterior). Re-certified at make_h.
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase8")


def hold_torque(h, q0):
    d = mujoco.MjData(h.m); fid = h.fid
    th = np.array([q0[i] for i in h.qadr])
    lo, hi = h.limits(th, np.zeros(5))
    c = P.static_certificate(h.m, d, q0, coll_geoms=[fid], tmax_override=np.maximum(lo, hi))
    if c["status"] != "STATIC_FEASIBLE": return None, c
    t = np.array(c["tau"])
    if np.any(t > hi + 1e-6) or np.any(-t > lo + 1e-6):     # directional (extensor/flexor) capacity
        return None, dict(c, status="TORQUE_INFEASIBLE_DIRECTIONAL")
    return np.array(c["tau"]), c


def rollout(h, params, lock=None, record=False, horizon=None, nk=None, flight_confirm=None):
    """Vectorised (Phase 9: 62 ms -> ~15 ms per rollout; identical physics and bookkeeping,
    checked against the Phase 8 implementation in tests/test_gate_a_fast.py)."""
    m = h.m; d = mujoco.MjData(m)
    q0 = h.place(CROUCH); d.qpos[:] = q0; d.qvel[:] = 0
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    mujoco.mj_forward(m, d)
    qa = np.array(h.qadr); va = np.array(h.vadr)
    P_, W_, FL_ = (np.array([t[i] for t in h.ta]) for i in range(3))
    a = np.clip(h._hold/h.tau, -1, 1) if h._hold is not None else np.zeros(5)
    NKl = nk or NK; Hh = horizon or HOR
    kn = np.asarray(params, float).reshape(5, NKl).copy()
    if lock is not None:
        for i in lock: kn[i, :] = a[i]
    nst = int(Hh/m.opt.timestep); per = max(1, nst//NKl); dt = m.opt.timestep; alpha = dt/h.act_tau
    E0 = float(d.energy[0] + d.energy[1]); z0 = float(d.subtree_com[0][2]); zmin = z0
    zmin = float(zmin)
    W = 0.0; work = np.zeros(5); imp = 0.0; vto = None; tto = None; Eto = None; Wto = None; cand = None
    fc = FLIGHT_CONFIRM if flight_confirm is None else flight_confirm
    wlog = []; peak = 0.0; pen = 0.0; tau_h = h.tau; wmax = h.wmax
    jac = np.zeros((3, m.nv))
    # precomputed per-step command (linear interpolation between knots, as before)
    ks = np.arange(nst)//per; ks = np.minimum(ks, NKl-1); k2s = np.minimum(ks+1, NKl-1)
    fs = (np.arange(nst) % per)/per
    CMD = kn[:, ks] + (kn[:, k2s]-kn[:, ks])*fs
    half_pi = 0.5*np.pi; one_m_fl = 1-FL_; QS = h.qsign; fv = np.zeros(6); qpos = d.qpos; qvel = d.qvel
    for s in range(nst):
        a += alpha*(CMD[:, s] - a)
        th = qpos[qa]; w = qvel[va]
        if h.anderson:
            lo, hi = h.limits(th, w)
        else:
            x = (QS*th - P_)/W_; x[x > 1] = 1; x[x < -1] = -1     # P15: curve in anatomical convention
            lim = tau_h*(FL_ + one_m_fl*np.cos(half_pi*x)**2)
            sp = 1.0 - np.abs(w)/wmax; sp[sp < 0] = 0; sp[sp > 1] = 1
            pos = w > 0; neg = w < 0
            hi = lim*(sp*pos + (~pos)); lo = lim*(sp*neg + (~neg))
        tau = a*tau_h; tau = np.minimum(np.maximum(tau, -lo), hi)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        w2 = qvel[va]
        W += float(tau @ (0.5*(w + w2)))*dt
        mujoco.mj_jacSubtreeCom(m, d, jac, 0); vz = float(jac[2] @ qvel)
        zc = d.subtree_com[0][2]
        if zc < zmin: zmin = zc
        fz = 0.0
        for ci in range(d.ncon):
            mujoco.mj_contactForce(m, d, ci, fv); fz += fv[0]
            pd = -d.contact[ci].dist
            if pd > pen: pen = pd
        imp += fz*dt
        if fz > peak: peak = fz
        tw = tau*w2; work += tw*(tw > 0)*dt
        if record: wlog.append(w2.copy())
        # TAKEOFF (Phase 10, defect P17 = AD2 never ported to Gate A): loss of foot contact with upward COM
        # velocity, kept only if the flight is SUSTAINED for FLIGHT_CONFIRM s; a momentary unloading that
        # re-contacts is not a takeoff (it was scored as one, at low velocity, since Phase 6).
        if cand is not None:
            if fz >= 1.0: cand = None
            elif (s+1)*dt - cand[1] >= fc - 1e-12: break
        elif fz < 1.0 and vz > 0.0:
            cand = (vz, (s+1)*dt, float(d.energy[0] + d.energy[1]), W, float(zc), imp)
            if fc <= 0: break                                   # legacy single-step rule (tests only)
    if cand is not None:
        vto, tto, Eto, Wto, zto, imp_to = cand
    else:
        vto = 0.0; tto = Hh; Eto = float(d.energy[0] + d.energy[1]); Wto = W; zto = float(d.subtree_com[0][2]); imp_to = imp
    mom_err = vto - (imp_to/h.M - G*tto)
    return dict(v=vto, height=max(vto, 0)**2/(2*G), T=tto, dip=z0 - zmin, z_takeoff=zto, z_start=z0, peak_grf_bw=peak/(h.M*G),
                work=work, energy_residual=(Eto - E0) - Wto, W=Wto, momentum_error=mom_err,
                max_pen=pen, wlog=np.array(wlog) if record else None,
                valid=bool(abs(mom_err) < 0.05 and (Eto - E0) - Wto <= max(1.0, 0.02*abs(Wto)) and vto > 0))


def make_h(scale=None):
    # MLAW_HUMAN_SPEC selects the body spec (Phase 10: spec/body_sourced.yaml); default body.yaml
    h = HP.Human(spec=os.environ.get("MLAW_HUMAN_SPEC", HP.SPEC), tau_scale=scale)
    pc = h.spec.get("protocol", {}).get("gate_a_crouch")
    if pc is not None and not os.environ.get("MLAW_KEEP_CROUCH"): CROUCH[:] = pc["value"]
    q0 = h.place(CROUCH); h._hold, h._cert = hold_torque(h, q0)
    return h


def solve(args):
    seed, lock, squat, scale, iters = args
    h = make_h(scale)
    def obj(x):
        r = rollout(h, x, lock)
        pen = 0.0
        if squat and r["dip"] > 0.01: pen += (r["dip"]-0.01)**2*200.0
        if not r["valid"]: pen += 0.1
        return -r["height"] + pen
    x0 = np.repeat((h._hold/h.tau)[:, None], NK, 1).ravel() if h._hold is not None else np.zeros(5*NK)
    es = cma.CMAEvolutionStrategy(np.clip(x0, -1, 1), 0.35, {"bounds": [-1, 1], "seed": seed, "verbose": -9,
                                                               "maxiter": iters, "popsize": 16})
    while not es.stop():
        X = es.ask(); es.tell(X, [obj(x) for x in X])
    x = es.result.xbest; r = rollout(h, x, lock, record=True)
    wl = r.pop("wlog")
    return dict(seed=seed, lock=lock, squat=squat, scale=scale, x=np.asarray(x).tolist(),
                wlog=wl.tolist(), **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()})


def xcorr_lag(a, b, dt, maxlag=0.10):
    a = a-a.mean(); b = b-b.mean(); n = int(maxlag/dt); best = (0, -1e9)
    for L in range(-n, n+1):
        x, y = (a[-L:], b[:len(b)+L]) if L < 0 else (a[:len(a)-L], b[L:]) if L > 0 else (a, b)
        if len(x) < 10: continue
        c = float(np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-12))
        if c > best[1]: best = (L*dt, c)
    return best


if __name__ == "__main__":
    h = make_h(); print("held crouch certificate:", h._cert["status"], "hold torques", np.round(h._hold, 1), flush=True)
    SEEDS = list(range(1, 9))
    jobs = ([(s, None, True, None, 300) for s in SEEDS] + [(s, [3, 4], True, None, 300) for s in SEEDS]
            + [(s, None, False, None, 300) for s in SEEDS])
    with Pool(14) as p: res = p.map(solve, jobs, chunksize=1)
    sj = [r for r in res if r["lock"] is None and r["squat"]]
    # Bobbert direction test: same control, stronger actuators (1.5x), NOT re-optimised
    bob = []
    for r in sj:
        hs = make_h([1.5]*5); rs = rollout(hs, r["x"])
        bob.append(dict(seed=r["seed"], base=r["height"], strong_same_control=rs["height"], valid=rs["valid"]))
    dt = h.m.opt.timestep; seq = []
    for r in sj:
        W = np.array(r["wlog"])
        if len(W) > 20:
            seq.append(dict(seed=r["seed"], **{f"{HP.JOINTS[a]}->{HP.JOINTS[b]}": xcorr_lag(W[:, a], W[:, b], dt) for a, b in ((0, 1), (1, 2))}))
    for r in res: r.pop("wlog", None)
    json.dump(dict(runs=res, bobbert=bob, sequencing=seq, hold=h._hold.tolist(), cert=h._cert["status"]),
              open(f"{OUT}/gate_a2.json", "w"), indent=1, default=float)
    f = lambda L: np.array([r["height"]*100 for r in L if r["valid"]])
    print("SJ height cm:", np.round(f(sj), 1), "published 34.6 +- 4.4")
    print("arms locked cm:", np.round(f([r for r in res if r['lock']]), 1))
    print("CMJ (dip allowed) cm:", np.round(f([r for r in res if not r['squat']]), 1))
    print("Bobbert same-control 1.5x:", [(round(b['base']*100, 1), round(b['strong_same_control']*100, 1)) for b in bob])
    print("sequencing lags (s, corr):", seq)
