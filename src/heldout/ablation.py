"""SCAFFOLD ABLATION on the held-out bodies (2026-10-16): A vs B at one rung, same knots, same gates, same CMA budget.
  B  = the sprint construction as is (engine.sprint.Sprint).
  A  = the same problem object with the BODY-DERIVED REFERENCE removed: the cycle reference Q (stance sweep + Cartesian swing + arm
       counter-swing) replaced by the standing pose at every phase, QD = 0, and the inverse-dynamics feedforward recomputed for that
       static reference by the same formula as engine.sprint.scaffold (base at v with the stride's vertical acceleration, no contact,
       stance legs zeroed). KEPT IDENTICAL (stability/feedback layers and timing): the event clock and stride timing (t_c, t_f), the
       force-controlled stance (half-sine GRF, COM height/velocity PD, attitude moment, CoP clip), Raibert placement maps PL, swing
       weights, attitude compensation, PD gains, the 8-knot periodic torque parameterisation, unpack ranges, running start velocity.
Search = engine.sprint's CMA protocol (sigma 0.1, bounds [-1, 1], worse-of-two-dt lexicographic score, PATIENCE early stop).
Measured: evals to the first candidate valid at both dt with v >= 0.97 v_rung (score < 0), best speed (min of the two dt).
usage: SPRINT_LAG=0.002 [SCALE=..] python heldout/ablation.py <robot> <v> <arm A|B> <seed> [iters] [pop] [workers] [tag]"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from engine import sprint as SP, core

G = 9.81


def ablate(Pb):
    """Arm A: the standing pose as the reference (see module doc). Mutates Pb.R in place."""
    S = Pb.S[core.DT_OPT]; R = Pb.R; m = S["m"]; types = S["types"]; nT = len(types); N = R["N"]; res = R["res"]
    Q = np.repeat(R["q_up"][None], N, 0); QD = np.zeros((N, nT)); ID = np.zeros((N, nT))
    tc, Tst, Tc, dz = R["tc"], R["Tst"], R["Tc"], R["dz"]
    if SP.ID_FF:
        dd = mujoco.MjData(m); qa_, dof_ = S["jarr"]["qad"], S["jarr"]["dof"]
        for k in range(N):
            tr_ = (k*res) % Tst; az = dz*(np.pi/tc)**2*np.sin(np.pi*tr_/tc) if tr_ < tc else -G
            dd.qpos[:] = S["q_settled"]; dd.qpos[:3] = [0, 0, 5.0]; dd.qpos[3:7] = [1, 0, 0, 0]; dd.qpos[qa_] = Q[k]
            dd.qvel[:] = 0; dd.qvel[0] = Pb.v; dd.qacc[:] = 0; dd.qacc[2] = az
            mujoco.mj_inverse(m, dd); ID[k] = dd.qfrc_inverse[dof_]
            for li_, sd_ in enumerate(("right", "left")):
                if ((k*res - (0.0 if li_ == 0 else Tst)) % Tc) < tc: ID[k, [i_ for i_, t_ in enumerate(types) if t_.startswith(sd_) and any(x_ in t_ for x_ in ("hip", "knee", "ankle"))]] = 0.0
    R.update(Q=Q, QD=QD, ID=ID, ablated="A: standing-pose reference"); return Pb


_W = {}
def _init(robot, v, scale, stride, arm):
    Pb = SP.Sprint(robot, v, scale, stride)
    _W["P"] = ablate(Pb) if arm == "A" else Pb
def _eval(z): s, r1, r2 = _W["P"].robust(z); return s, SP.summarize(r1), SP.summarize(r2)


def run(robot, v, arm, seed, iters, pop, workers, scale):
    import cma, multiprocessing as mp
    Pb = SP.Sprint(robot, v, scale); st = Pb.st
    if arm == "A": ablate(Pb)
    z0 = SP.z_prior(Pb.S[core.DT_OPT]); s0, r1, r2 = Pb.robust(z0); n_ev = 1
    best = (s0, z0, SP.summarize(r1), SP.summarize(r2)); first_valid = 1 if s0 < 0 else None; hist = [(0, float(s0))]; t0 = time.time(); last_imp = 0
    pool = mp.get_context("fork").Pool(workers, initializer=_init, initargs=(robot, v, scale, st, arm))
    es = cma.CMAEvolutionStrategy(np.asarray(z0, float), 0.1, {"bounds": [-1, 1], "seed": seed, "verbose": -9, "maxiter": iters, "popsize": pop})
    try:
        while not es.stop():
            X = es.ask(); out = pool.map(_eval, X); F = [o[0] for o in out]; es.tell(X, F)
            for i, f in enumerate(F):
                if first_valid is None and f < 0: first_valid = n_ev + i + 1
            n_ev += len(X); k = int(np.argmin(F))
            if F[k] < best[0]: best = (F[k], np.array(X[k]), out[k][1], out[k][2]); last_imp = es.countiter
            hist.append((es.countiter, float(best[0])))
            print(f"{robot} {arm} s{seed} it {es.countiter} best {best[0]:.4f} v {best[2]['value']:.3f}/{best[3]['value']:.3f} {best[2]['reasons']} first_valid {first_valid} {time.time() - t0:.0f}s", flush=True)
            if best[0] < 0 and es.countiter - last_imp >= SP.PATIENCE: break
    finally:
        pool.close(); pool.join()
    zb = np.asarray(best[1]); q1 = Pb.run(zb, core.DT_OPT); q2 = Pb.run(zb, core.DT_CHECK)
    valid = bool(q1["valid"] and q2["valid"] and min(q1["value"], q2["value"]) >= 0.97*v)
    return dict(robot=robot, v=v, arm=arm, seed=seed, scale=scale, act_tau=SP.ACT_TAU, budget=dict(iters=iters, pop=pop), evals=n_ev, secs=round(time.time() - t0),
                evals_to_first_valid=first_valid, valid_best=valid, best_speed=(min(q1["value"], q2["value"]) if valid else None),
                meas=[SP.summarize(q1), SP.summarize(q2)], prior_score=float(s0), hist=hist, z=zb.tolist(), stride=SP.stride_record(st))


if __name__ == "__main__":
    robot, v, arm, seed = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    it = int(sys.argv[5]) if len(sys.argv) > 5 else 120; pop = int(sys.argv[6]) if len(sys.argv) > 6 else 12; wk = int(sys.argv[7]) if len(sys.argv) > 7 else 6
    tag = sys.argv[8] if len(sys.argv) > 8 else ""; scale = float(os.environ.get("SCALE", 1.0))
    res = run(robot, v, arm, seed, it, pop, wk, scale)
    path = f"{core.RES}/heldout/ablation_{robot}{tag}_v{v}_{arm}_s{seed}.json"; SP.write(path, res)
    msg = f"ablation {robot}{tag} {arm} s{seed}: first valid at {res['evals_to_first_valid']} evals, best {res['best_speed']} ({res['evals']} evals, {res['secs']/60:.0f} min)"
    print(msg, flush=True); os.system(f'~/bin/notify "{msg[:150]}" >/dev/null 2>&1')
