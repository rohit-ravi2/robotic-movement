"""Task interface + generic optimizer. A Task is: a plant (per demand), a parameterisation, an optional strategy prior,
a rollout that returns a RECORD, and a scoring that is LEXICOGRAPHIC with the lessons of Phases 11-14 built in:
  * any violation ranks below every valid solution (100 + graded violation; rank-based CMA needs the ORDER right),
  * the trivial basin (value below the task floor) ranks below every violation (200 + shortfall), unconditionally,
  * robust objective = the WORSE of the faithful-dt and finer-dt rollouts,
  * every reported number is a faithful replay at DT_OPT, with the DT_CHECK replay beside it.
Record keys (all tasks): value (float, higher = better), valid (bool), reasons (list), violation (float >= 0),
cost (float, positive work J), extra (dict, task-specific), plus dt."""
import sys, os, json, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cma
from engine import core


class Task:
    name = "base"; robot = "G1"
    def sigma0(self, demand, warm): return 0.10 if (warm is not None or demand.get("prior")) else 0.35   # a prior makes z=0 a good solution: refine, don't explore
    def demand_default(self): return {}
    def plant(self, demand, dt=core.DT_OPT, fresh=False): raise NotImplementedError
    def n_params(self, S, demand): raise NotImplementedError
    def unpack(self, S, demand, z): """z in [-1,1]^n -> params dict"""; raise NotImplementedError
    def controller(self, S, demand, params): return None
    def rollout(self, S, demand, params, ctrl_fn=None, record=False): raise NotImplementedError
    def floor(self, demand): return None
    def objective(self, rec, demand): return -rec["value"]
    def warm(self, S, demand, stored): """stored solution dict -> z (for warm starts / replays)"""; raise NotImplementedError

    # ---- generic machinery
    def score(self, rec, demand):
        if rec["violation"] > 0 or not rec["valid"]:
            return 100.0 + max(rec["violation"], 1e-3) - 0.01*rec["value"]
        return self.objective(rec, demand)

    def run(self, S, demand, z, record=False):
        p = self.unpack(S, demand, z); fn = self.controller(S, demand, p)
        rec = self.rollout(S, demand, p, ctrl_fn=fn, record=record); rec["dt"] = float(S["m"].opt.timestep); rec["params"] = p
        return rec


def robust_score(task, demand, rec1, rec2):
    """Worse-of-two with the unconditional floor. rec2 may be None (non-robust)."""
    fl = task.floor(demand); recs = [r for r in (rec1, rec2) if r is not None]
    vmin = min(r["value"] for r in recs)
    s = [task.score(r, demand) for r in recs]
    pen_fail = sum((si - 100.0) for si in s if si >= 100.0)
    if fl is not None and vmin < fl: return 200.0 + (fl - vmin) + pen_fail
    if all(si < 100.0 for si in s): return max(s)
    return 100.0 + pen_fail


def optimize(task, demand, seed, iters=100, pop=16, warm=None, sigma=None, robust=True, log=None, z0=None, measure_after=True):
    S1 = task.plant(demand, core.DT_OPT); S2 = task.plant(demand, core.DT_CHECK, fresh=True) if robust else None
    n = task.n_params(S1, demand)
    if z0 is None: z0 = np.zeros(n) if warm is None else np.clip(task.warm(S1, demand, warm), -1, 1)
    z0 = np.asarray(z0, float); warm = warm if warm is not None else (z0 if np.any(z0) else None)
    sig = sigma if sigma is not None else task.sigma0(demand, warm)
    def obj(z):
        r1 = task.run(S1, demand, z); r2 = task.run(S2, demand, z) if robust else None
        return robust_score(task, demand, r1, r2)
    es = cma.CMAEvolutionStrategy(z0, sig, {"bounds": [-1, 1], "seed": seed, "verbose": -9, "maxiter": iters, "popsize": pop})
    t0 = time.time(); best = (obj(z0), np.array(z0))        # the start point is always a candidate: a warm start can never come back worse
    while not es.stop():
        X = es.ask(); F = [obj(z) for z in X]; es.tell(X, F)
        k = int(np.argmin(F))
        if F[k] < best[0]: best = (F[k], np.array(X[k]))
        if log is not None: log(es.countiter, best[0])
    z = best[1] if best[1] is not None else np.asarray(es.result.xbest)
    if not measure_after: return dict(z=np.asarray(z).tolist(), score=float(best[0]), secs=time.time() - t0)
    out = measure(task, demand, z); out.update(seed=seed, iters=iters, pop=pop, secs=time.time() - t0, score=float(best[0]))
    return out


def measure(task, demand, z, dts=(core.DT_OPT, core.DT_CHECK)):
    """Faithful replay at each dt; a solution counts only if valid at every dt and above the floor."""
    meas = []
    for i, dt in enumerate(dts):
        S = task.plant(demand, dt, fresh=(i > 0)); r = task.run(S, demand, z, record=(i == 0))
        meas.append(dict(dt=dt, value=r["value"], valid=r["valid"], reasons=r["reasons"], violation=r["violation"], cost=r["cost"], extra=r["extra"]))
    fl = task.floor(demand); vmin = min(m["value"] for m in meas)
    ok = all(m["valid"] and m["violation"] <= 0 for m in meas) and (fl is None or vmin >= fl)
    h = hashlib.sha256(np.asarray(z, float).round(10).tobytes()).hexdigest()[:12]
    return dict(task=task.name, robot=task.robot, demand=demand, z=np.asarray(z).tolist(), value=meas[0]["value"], ok_all=bool(ok), meas=meas,
                params={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in task.unpack(task.plant(demand, dts[0]), demand, z).items()}, z_hash=h)
