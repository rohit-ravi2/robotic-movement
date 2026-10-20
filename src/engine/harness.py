"""J2: exploit harness. Every task's scoring must put (valid, above floor) < (violation) < (trivial basin), be worse-of-two
across timesteps, and reject the historical exploits when their stored solutions are replayed. Run: python engine/harness.py [task]
The three historical objective defects are reconstructed as OLD scorers and must FAIL these checks (proof the harness bites)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from engine import core, task as TK

def _rec(value, valid=True, violation=0.0): return dict(value=value, valid=valid, reasons=[] if valid else ["X"], violation=violation, cost=0.0, extra={})

# ---- synthetic rank-order probes (no physics) --------------------------------------------------------------
def probe_rank_order(task, demand):
    fl = task.floor(demand) or 0.0
    deep_ok = _rec(fl + 1.5); deep_ok2 = _rec(fl + 1.4)
    deep_fail = _rec(fl + 1.5, valid=False, violation=0.5)       # e.g. lands at 1 ms, not at 0.5 ms
    hop_ok = _rec(max(0.05, fl - 1.5))                            # trivial basin, lands trivially
    hop_fail_one = _rec(max(0.05, fl - 1.5), valid=False, violation=0.02)  # the Phase 14 robust3 loophole
    S = lambda a, b: TK.robust_score(task, demand, a, b)
    checks = {
        "worse-of-two": S(deep_ok, deep_ok2) == max(task.score(deep_ok, demand), task.score(deep_ok2, demand)),
        "valid < violation": S(deep_ok, deep_ok) < S(deep_fail, deep_fail),
        "violation < hop (floor)": (fl == 0.0) or S(deep_fail, deep_fail) < S(hop_ok, hop_ok),
        "deep-fails-one < hop-fails-one": (fl == 0.0) or S(deep_ok, deep_fail) < S(hop_ok, hop_fail_one),
        "hop never beats deep": (fl == 0.0) or S(hop_ok, hop_ok) > S(deep_ok, deep_fail),
    }
    return checks

# ---- the three historical defective objectives, reconstructed ---------------------------------------------
def old_better_of_two(task, demand, a, b):          # Phase 14 robust v0
    s1, s2 = task.score(a, demand), task.score(b, demand)
    return min(s1, s2) if (s1 < 0 and s2 < 0) else 100.0
def old_floor_below_failure(task, demand, a, b):    # robust3 v0 (no-op under rank-based CMA)
    s1, s2 = task.score(a, demand), task.score(b, demand); fl = task.floor(demand) or 0.0
    if s1 < 0 and s2 < 0:
        w = max(s1, s2); return 50.0 + (fl + w) if -w < fl else w
    return 100.0 + (0.0 if s1 < 0 else s1 - 100.0) + (0.0 if s2 < 0 else s2 - 100.0)
def old_floor_only_if_landed(task, demand, a, b):   # robust3 (loophole)
    s1, s2 = task.score(a, demand), task.score(b, demand); fl = task.floor(demand) or 0.0
    if s1 < 0 and s2 < 0:
        w = max(s1, s2); return 200.0 + (fl + w) if -w < fl else w
    return 100.0 + (0.0 if s1 < 0 else s1 - 100.0) + (0.0 if s2 < 0 else s2 - 100.0)

def probe_old_objectives(task, demand):
    """Each old scorer must break at least one rank-order check (otherwise the harness does not bite)."""
    fl = task.floor(demand) or 2.0
    deep_ok = _rec(fl + 1.5); deep_fail = _rec(fl + 1.5, valid=False, violation=0.5); hop_ok = _rec(fl - 1.5); hop_fail_one = _rec(fl - 1.5, valid=False, violation=0.02)
    out = {}
    for name, f in (("better-of-two", old_better_of_two), ("floor-below-failure", old_floor_below_failure), ("floor-only-if-landed", old_floor_only_if_landed)):
        broken = []
        if f(task, demand, _rec(fl + 1.5), _rec(fl + 1.4)) != max(task.score(_rec(fl + 1.5), demand), task.score(_rec(fl + 1.4), demand)): broken.append("worse-of-two")
        if not (f(task, demand, deep_fail, deep_fail) < f(task, demand, hop_ok, hop_ok)): broken.append("violation < hop")
        if not (f(task, demand, deep_ok, deep_fail) < f(task, demand, hop_ok, hop_fail_one)): broken.append("deep-fails-one < hop-fails-one")
        out[name] = broken
    return out

# ---- replay probes: historical exploit solutions must score as violations under the task's scorer ---------
def probe_replays(task, demand, stored, label):
    S = task.plant(demand, core.DT_OPT); res = []
    for r in stored:
        z = task.warm(S, demand, r); rec = task.run(S, demand, z); res.append((rec["value"], rec["valid"], rec["violation"], task.score(rec, demand)))
    return label, res

def run(task_name):
    if task_name == "jump":
        from tasks.jump import JumpTask; task = JumpTask("G1"); demand = task.demand_default()
        hops = [r for r in json.load(open(f"{core.RES}/phase14/jump14_robust.json")) if r["v"] < 1.0][:2]
        replays = [probe_replays(task, demand, hops, "Phase 14 robust-v0 hops (must be below floor -> trivial)")]
    elif task_name == "reach":
        from tasks.reach import ReachTask; task = ReachTask("G1"); demand = dict(task.demand_default(), prior="ascent"); replays = []
    else:
        from tasks.throw import ThrowTask; task = ThrowTask("G1"); demand = task.demand_default()
        v0 = json.load(open(f"{core.RES}/phase11/throw_g1_main_v0_flatpenalty.json"))
        flings = [dict(x=r["x"], amp=r.get("amp", 0.5), tscale=r.get("tscale", 1.0)) for r in v0 if r["method"] == "APRIOR" and r.get("amp") is not None][:2]
        replays = [probe_replays(task, demand, flings, "Phase 11 v0 flings (must be violations)")] if flings else []
    ok = True
    print(f"== {task_name}: rank-order probes")
    for k, v in probe_rank_order(task, demand).items(): print(f"   {'PASS' if v else 'FAIL'} {k}"); ok &= bool(v)
    print(f"== {task_name}: historical objectives must break a check")
    for k, v in probe_old_objectives(task, demand).items(): print(f"   {'BITES' if v else 'NOT CAUGHT'} {k}: breaks {v}"); ok &= bool(v)
    for label, res in replays:
        print(f"== {task_name}: {label}")
        for v, valid, viol, sc in res:
            fl = task.floor(demand)
            flagged = (viol > 0 or not valid) or (fl is not None and v < fl)
            print(f"   value {v:.2f} valid {valid} violation {viol:.2f} score {sc:.1f} -> {'rejected' if flagged else 'ACCEPTED (BAD)'}"); ok &= flagged
    print("HARNESS", "GREEN" if ok else "RED"); return ok

def run_compose():
    """Composer acceptance (added 2026-10-10 after a silent regression): landed deep jump -> settle -> throw from the landing
    stance must stay VALID and above 20 m/s. A handoff change had dropped it to 9-15 m/s with the stance lost."""
    from engine import compose as C; from tasks.jump import JumpTask; from tasks.throw import ThrowTask
    R4 = {r["seed"]: r for r in json.load(open(f"{core.RES}/phase14/jump14_robust4.json"))}; Wt = json.load(open(f"{core.RES}/atlas/warm_throw_best.json"))[0]
    jt, tt = JumpTask("G1"), ThrowTask("G1"); dj = jt.demand_default(); dt_ = dict(tt.demand_default(), omega=Wt["omega"])
    res = C.run_sequence("G1", [dict(task=jt, demand=dj, z=jt.warm(jt.plant(dj), dj, R4[1])), dict(task=tt, demand=dt_, z=tt.warm(tt.plant(dt_), dt_, Wt))])
    sj, st = res["steps"]; ok = bool(sj["valid"] and st["valid"] and st["value"] >= 20.0)
    print(f"== compose: jump valid {sj['valid']} -> throw {st['value']:.1f} m/s valid {st['valid']} {st['reasons']} -> {'PASS' if ok else 'FAIL'}"); return ok

if __name__ == "__main__":
    names = sys.argv[1:] or ["jump", "throw", "reach", "compose"]; allok = all([(run_compose() if n == "compose" else run(n)) for n in names]); sys.exit(0 if allok else 1)
