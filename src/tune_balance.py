"""Phase 14: tune the balance/landing controller (balance.py) for ROBUST landing. Suite: drops from 10-62 cm x
perturbations (initial forward/backward COM velocity +-0.3 m/s, sideways 0.15 m/s, body pitch +-0.08 rad).
Objective: landings survived (+ small continuous margin term). CMA-ES over 9 controller parameters; each suite
evaluation runs its sims in parallel. Then a held-out validation suite (different perturbations) at 2 ms and 1 ms."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cma, branch_i as BI, balance as B, aprior_standard as AS
from multiprocessing import Pool
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase14")
ROBOT = os.environ.get("ROBOT", "G1")
# drop heights: up to each robot's deep-jump apex (G1 3.5 m/s -> 62 cm; H1 2.3-2.55 m/s -> 27-33 cm, 40 cm covers it)
H = (0.10, 0.20, 0.30, 0.45, 0.62) if ROBOT == "G1" else (0.10, 0.20, 0.30, 0.40)
TAG = os.environ.get("TAG", "" if ROBOT == "G1" else "_" + ROBOT.lower())   # TAG=_brake: retune under the back-EMF brake without overwriting the Phase 14 file
PERT = [(0, 0, 0), (0.3, 0, 0), (-0.3, 0, 0), (0, 0.15, 0), (0, 0, 0.08), (0, 0, -0.08)]       # (vx, vy, pitch)
PERT_VAL = [(0.15, 0.1, 0.04), (-0.2, -0.1, -0.05), (0.25, 0, -0.06), (-0.1, 0.15, 0.06)]
_S = {}

def unpack(z):
    z = np.clip(z, -1, 1); u = (z + 1)/2
    return dict(stiff=60.0*10**(1.2*u[0]), k_cp=3.0*u[1], k_trunk=2.0*u[2], crouch=1.2*u[3], t_down=0.05 + 0.25*u[4],
                t_hold=0.05 + 0.45*u[5], t_up=0.2 + 1.0*u[6], pre_bend=0.4*u[7], k_foot=1.5*u[8], mode="land")

def sim(args):
    params, h, pert, dt = args
    if dt not in _S:
        if ROBOT == "G1": c = AS.clip0(); S = BI.setup(c["start"])
        else:
            Q0 = np.load(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase9/h1_start_soleflat.npy")); S = BI.setup(start=Q0, robot="H1")
        BI.apply_dt(S, dt); _S[dt] = (S, np.array(S["q_settled"]))
    S, q = _S[dt]
    q0 = q.copy(); q0[2] += h
    if pert[2]:                                             # pitch the whole body about y
        a = pert[2]/2; dq = np.array([np.cos(a), 0, np.sin(a), 0]); w, x, y, zq = q0[3:7]
        q0[3:7] = [dq[0]*w - dq[2]*y, dq[0]*x + dq[2]*zq, dq[0]*y + dq[2]*w, dq[0]*zq - dq[2]*x]
    v0 = np.zeros(S["m"].nv); v0[0], v0[1] = pert[0], pert[1]
    ctl = B.BalanceController(S, q, params=params); r = B.simulate(S, ctl, q0, v0=v0, T=3.0)
    return r

def suite(params, pool, perts, dt=0.001):
    rs = pool.map(sim, [(params, h, p, dt) for h in H for p in perts])
    ok = sum(r["ok"] for r in rs); margin = sum(min(1.0, max(0.0, r["up_end"])) + min(1.0, max(0.0, r["z_end"]/0.66)) for r in rs)
    return ok, margin, rs

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); hist = []
    with Pool(int(os.environ.get("NPROC", "14"))) as pool:
        z0 = np.array([0.1, -0.33, 0.0, 0.33, 0.0, -0.2, -0.5, -0.25, 0.33])   # omega ~ 250 rad/s, k_cp 1, crouch .8
        es = cma.CMAEvolutionStrategy(z0, 0.3, {"bounds": [-1, 1], "seed": 1, "verbose": -9, "maxiter": int(os.environ.get("ITERS", "25")), "popsize": 10})
        while not es.stop():
            Z = es.ask(); F = []
            for z in Z:
                ok, mg, _ = suite(unpack(z), pool, PERT); F.append(-(ok + 0.05*mg)); hist.append((ok, float(mg), unpack(z)))
            es.tell(Z, F); print(f"iter {es.countiter}: best so far {int(-min(F) // 1)}/{len(H)*len(PERT)} landed  ({time.time()-t0:.0f}s)", flush=True)
        best = unpack(es.result.xbest)
        res = {}
        for dt in (0.001, 0.0005):
            for name, perts in (("train", PERT), ("held_out", PERT_VAL)):
                ok, mg, rs = suite(best, pool, perts, dt)
                res[f"{name}_dt{dt}"] = dict(ok=ok, n=len(rs), per_height={h: int(sum(r["ok"] for r, (hh, p) in zip(rs, [(hh, p) for hh in H for p in perts]) if hh == h)) for h in H},
                                            peak_grf_bw=float(np.median([r["peak_grf_bw"] for r in rs])))
                print(name, dt, ok, "/", len(rs), res[f"{name}_dt{dt}"]["per_height"], flush=True)
    json.dump(dict(best=best, results=res, history=hist, robot=ROBOT, heights=H), open(f"{OUT}/balance_tuning{TAG}.json", "w"), default=float, indent=1)
