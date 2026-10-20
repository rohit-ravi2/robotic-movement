"""J6: selector. Demand vector in -> capability (atlas record) out, by envelope containment and minimum cost; exposes the
switch surfaces (where the chosen strategy changes along a demand axis). Also imports the Phase 11 demand sweep so the
throw recruitment threshold is available before the engine's own throw sweep lands."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from engine import core, atlas as AT


def _num(v):
    if isinstance(v, (list, tuple)): return [float(x) for x in v]
    if isinstance(v, bool): return float(v)
    try: return float(v)
    except (TypeError, ValueError): return v


def demand_distance(d, q, axes):
    """Normalised distance between a record's demand and a query over the given axes (strings must match exactly)."""
    tot = 0.0
    for a in axes:
        x, y = _num(d.get(a)), _num(q.get(a))
        if isinstance(x, str) or isinstance(y, str) or x is None or y is None: tot += 0.0 if x == y else 1e9; continue
        x, y = np.atleast_1d(x), np.atleast_1d(y); tot += float(np.sum((x - y)**2))
    return float(np.sqrt(tot))


def select(robot, task, query, axes, tol=1e-6, records=None):
    """Best (lowest cost) ok_all record whose demand matches the query on `axes` within tol. None if the demand is
    outside the atlas's feasible envelope (no ok record at that point)."""
    R = records if records is not None else AT.load(robot, task)
    cands = [r for r in R if r["ok_all"] and demand_distance(r["demand"], query, axes) <= tol]
    if not cands: return None
    return min(cands, key=lambda r: r["meas"][0]["cost"])


def feasible_envelope(robot, task, axes, records=None):
    """Per demand point: whether any seed succeeded, best value, min cost, n seeds."""
    R = records if records is not None else AT.load(robot, task)
    return AT.envelope(R, axes)


def switch_surface(robot, task, axis, descriptor, records=None, level=None):
    """Along one demand axis: the median strategy descriptor (a function record -> float) of the ok solutions at each
    point, and the first axis value where it crosses `level` (None = midpoint between the extremes)."""
    R = [r for r in (records if records is not None else AT.load(robot, task)) if r["ok_all"]]
    pts = {}
    for r in R: pts.setdefault(float(_num(r["demand"][axis])), []).append(descriptor(r))
    xs = sorted(pts); ys = [float(np.median(pts[x])) for x in xs]
    if not xs: return dict(axis=axis, points=[], switch=None)
    lv = level if level is not None else 0.5*(min(ys) + max(ys)); sw = None
    for x0, x1, y0, y1 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
        if (y0 - lv)*(y1 - lv) <= 0 and y0 != y1: sw = x0 + (lv - y0)*(x1 - x0)/(y1 - y0); break
    return dict(axis=axis, points=list(zip(xs, ys)), level=lv, switch=sw)


def legs_share(r):
    w = r["meas"][0]["extra"].get("work", {}); tot = sum(w.values()) or 1e-9; return w.get("legs", 0.0)/tot


def import_phase11_demand(path=f"{core.RES}/phase11/throw_demand2.json"):
    """Phase 11 arm-first warm-started demand sweep (full chain) as atlas-shaped records (value = speed, extra.work)."""
    D = json.load(open(path)); out = []
    for r in D:
        if r.get("target") is None or "full" not in r: continue
        f = r["full"]
        if not f.get("valid_all_dt"): continue
        out.append(dict(task="throw", robot="G1", demand=dict(target_speed=float(r["target"]), prior="arm_first", rom_scale=1.0), seed=r["seed"],
                        value=f["speed"], ok_all=True, z=r.get("x_full"), meas=[dict(dt=0.001, value=f["speed"], valid=True, reasons=[], violation=0.0,
                        cost=float(sum(f["work"].values())), extra=dict(work=f["work"], participation=f["participation"]))], source="phase11/throw_demand2.json"))
    return out


if __name__ == "__main__":
    # acceptance: the throw recruitment threshold from atlas-shaped data alone
    R = import_phase11_demand()
    env = feasible_envelope("G1", "throw", ["target_speed"], records=R)
    print("Phase 11 demand sweep as atlas records:", len(R), "feasible points", {k[0]: (e["n_ok"], round(e["best"], 2), round(e["cost"])) for k, e in sorted(env.items())})
    sw = switch_surface("G1", "throw", "target_speed", legs_share, records=R)
    print("legs share vs target:", [(x, round(y, 2)) for x, y in sw["points"]], "-> switch at target ~ %.1f m/s (level %.2f)" % (sw["switch"], sw["level"]))
    q = dict(target_speed=6.0, prior="arm_first", rom_scale=1.0); best = select("G1", "throw", q, ["target_speed", "prior"], records=R)
    print("select(target 6 m/s):", None if best is None else (best["seed"], round(best["value"], 2), round(best["meas"][0]["cost"]), round(legs_share(best), 2)))
    q = dict(target_speed=30.0, prior="arm_first", rom_scale=1.0); print("select(target 30 m/s):", select("G1", "throw", q, ["target_speed", "prior"], records=R))
    # the engine's own atlas so far
    for t in ("jump", "throw", "reach"):
        env = feasible_envelope("G1", t, ["prior"] if t == "jump" else (["rom_scale", "target_speed"] if t == "throw" else ["target", "load"]))
        print(t, {str(k): (e["n_ok"], e["n"], None if e["best"] is None else round(e["best"], 2)) for k, e in env.items()})
