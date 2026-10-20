"""Phase 10 Gate A analysis, written before the results were read. Targets from spec/validation_targets.md
(all title-verified). Model height = v_to^2/2g (H_flight); H_stand = H_flight + (z_CM,takeoff - z_CM,standing),
the latter measured by replaying each solution (same definition as the Bobbert lab)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco
R = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase10")
FLIGHT = {"Kyne 2025 (n23, 18 y athletes)": (34.6, 4.4), "Kozinc 2024 (n209)": (29.11, 5.25), "Djuric 2023 (n16)": (38.0, 5.0)}
STAND = {"Bobbert 2008 P3 (n8)": (41.0, 5.0), "Bobbert 2006 two-leg (n8)": (28.3, 6.0), "Bobbert 1996 (n6)": (44.7, 4.4)}
CMJ_SJ = (3.0, 4.0)   # cm, consistent range across S1/S5/S6/S7 (validation_targets.md)

def stand_offsets(spec, xs):
    """z_CM at takeoff minus z_CM standing upright (heels down), per solution (Bobbert-lab definition)."""
    os.environ["MLAW_HUMAN_SPEC"] = spec
    import importlib, gate_a2 as GA; importlib.reload(GA)
    h = GA.make_h(); h.place(None); zs = float(h.com()[2])
    return [GA.rollout(h, np.array(x))["z_takeoff"] - zs for x in xs]

def summarize(res):
    out = {}
    for r in res:
        k = (r["extra"]["spec"], r["extra"].get("tag", ""), r["budget"], r["squat"])
        out.setdefault(k, []).append(r)
    rows = []
    for k, g in sorted(out.items()):
        h = np.array([x["height"]*100 for x in g if x["valid"]])
        rows.append(dict(spec=k[0], tag=k[1], budget=k[2], squat=k[3], n=len(g), n_valid=int(len(h)),
                         median=float(np.median(h)) if len(h) else None, best=float(h.max()) if len(h) else None,
                         q25=float(np.percentile(h, 25)) if len(h) else None, q75=float(np.percentile(h, 75)) if len(h) else None,
                         sd=float(h.std()) if len(h) else None,
                         dip=float(np.median([x["dip"] for x in g if x["valid"]])) if len(h) else None))
    return rows

if __name__ == "__main__":
    res = json.load(open(f"{R}/gate_a10_main.json"))
    rows = summarize(res)
    from gate_a10 import SPECS
    top = {}
    for r in rows:
        if r["tag"] == "ladder" and r["n_valid"]:
            top[r["spec"]] = max(top.get(r["spec"], 0), r["budget"])
    for sp, b in top.items():
        g = [x for x in res if x["extra"]["spec"] == sp and x["extra"].get("tag") == "ladder" and x["budget"] == b and x["valid"]]
        off = np.array(stand_offsets(SPECS[sp], [x["x"] for x in g]))*100
        hf = np.array([x["height"]*100 for x in g])
        for r in rows:
            if r["spec"] == sp and r["tag"] == "ladder" and r["budget"] == b:
                r["takeoff_above_stand_cm"] = float(np.median(off)); r["H_stand_median"] = float(np.median(hf + off))
    for r in rows: print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    json.dump(dict(rows=rows, targets_flight=FLIGHT, targets_stand=STAND, cmj_minus_sj=CMJ_SJ), open(f"{R}/gate_a10_summary.json", "w"), indent=1)
