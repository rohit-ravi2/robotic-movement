"""Track T1 (2026-10-17): SUPPORT GATE + STOCK HEAD-CARRY RERUN. G1H, stock menagerie model (audit: headcarry_t1_stock.py), simulator only.
Phases: J re-judge every stored Track A head-carry / hands result under the new support gate (replay best z on FRESH plants at 1 ms and
0.5 ms); L max-load ladders (head+arms vs hands-only, box and pot) from the last gate-passing standing mass upward, warm-started from
the previous valid mass, with the support gate inside the search score, until the first failing mass; A law-off ablation at each new
maximum (both dt); R render the best clean hold. Outputs results/overnight/T1/. usage: python headcarry_t1.py --phase JLAR --workers 7"""
import os, sys, json, time, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from engine import headcarry as HC
import headcarry_search as HS

A_DIR = HC.RES; OUT = os.path.join(os.path.dirname(A_DIR), "T1"); HS.OUT = OUT
STEP = {"head": 2.0, "hands": 1.0}


def keep(r): return {k: r.get(k) for k in ("t_held", "reason", "head_frac", "tilt_max_deg", "foot_travel", "foot_travel_max", "pelvis_drift", "foot_loaded_frac", "com_y_range", "palm_N_mean", "util_p95", "e_max")}


def _ev(a):
    tag, obj, mass, st, z, T, dt, sway = a; r = HC.evaluate((obj, mass, st, np.asarray(z), T, dt, sway)); return tag, dt, r


def phase_J(pool):
    path = f"{OUT}/rejudge.json"; done = json.load(open(path)) if os.path.exists(path) else {}
    files = sorted(f for f in glob.glob(f"{A_DIR}/*.json") if os.path.basename(f) not in ("predictor.json", "summary.json"))
    jobs = []
    for f in files:
        C = json.load(open(f)); tag = os.path.basename(f)[:-5]
        if "best_z" not in C or tag in done: continue
        jobs += [(tag, C["obj"], C["mass"], C["strategy"], C["best_z"], C["T"], dt, C.get("sway")) for dt in (0.001, 0.0005)]
    res = {}
    for tag, dt, r in pool.imap_unordered(_ev, jobs):
        res.setdefault(tag, {})[str(dt)] = r
        if len(res[tag]) == 2:
            C = json.load(open(f"{A_DIR}/{tag}.json")); a, b = res[tag]["0.001"], res[tag]["0.0005"]
            done[tag] = dict(strategy=C["strategy"], obj=C["obj"], mass=C["mass"], sway=C.get("sway"), stored_valid=C["valid"],
                             old_gate=[HC.gate(a, C["T"], C["strategy"], C["mass"], support=False), HC.gate(b, C["T"], C["strategy"], C["mass"], support=False)],
                             new_gate=[HC.gate(a, C["T"], C["strategy"], C["mass"]), HC.gate(b, C["T"], C["strategy"], C["mass"])],
                             r1ms=keep(a), r05ms=keep(b), best_z=C["best_z"])
            done[tag]["valid_new"] = all(done[tag]["new_gate"])
            json.dump(HS.J(done), open(path, "w"), indent=1)
            HS.log(f"rejudge {tag}: stored valid {C['valid']} -> new {done[tag]['valid_new']} (old-gate replay {done[tag]['old_gate']}); foot travel max {a['foot_travel_max']:.3f}/{b['foot_travel_max']:.3f} m, pelvis drift {a['pelvis_drift']:.3f}/{b['pelvis_drift']:.3f} m")
    return done


def phase_L(pool, budget):
    J = json.load(open(f"{OUT}/rejudge.json")); out = {}
    for st, ob in (("head", "pot"), ("head", "box"), ("hands", "box"), ("hands", "pot")):
        ok = [(v["mass"], k) for k, v in J.items() if v["strategy"] == st and v["obj"] == ob and v["sway"] is None and v["valid_new"]]
        if ok: m0, src = max(ok); z0 = J[src]["best_z"]
        else: m0, src, z0 = 0.0, None, None
        HS.log(f"ladder {st} {ob}: start above {m0:g} kg (last gate-passing stored case {src})")
        mass, prev_z, last_ok, steps = m0, z0, (m0, src), []
        while True:
            mass = round(mass + STEP[st], 3)
            r = HS.run_case(pool, st, ob, mass, budget, x0=prev_z, tag=f"{st}_{ob}_{mass:g}kg_t1")
            steps.append(dict(mass=mass, valid=r["valid"], evals=r["evals"], wall_s=r["wall_s"], b1=keep(r["best"]), b05=keep(r["check_dt0005"])))
            if not r["valid"] or mass >= 80: break
            last_ok = (mass, f"{st}_{ob}_{mass:g}kg_t1"); prev_z = r["best_z"]
        out[f"{st}_{ob}"] = dict(start_from=dict(mass=m0, case=src), steps=steps, max_valid=last_ok[0], max_case=last_ok[1], first_fail=mass)
        json.dump(HS.J(out), open(f"{OUT}/ladders.json", "w"), indent=1)
        os.system(f"~/bin/notify 'T1 ladder {st} {ob}: max {last_ok[0]:g} kg (first fail {mass:g})' 2>/dev/null")
    return out


def _abl(a):
    tag, obj, mass, st, z, dt, law = a; r = HC.rollout(obj, mass, st, np.asarray(z), 10.0, dt, False, None, law=law); r.pop("trace", None); return tag, dt, law, r


def case_z(tag):
    p = f"{OUT}/{tag}.json"
    if os.path.exists(p): C = json.load(open(p)); return C["obj"], C["mass"], C["strategy"], C["best_z"]
    v = json.load(open(f"{OUT}/rejudge.json"))[tag]; return v["obj"], v["mass"], v["strategy"], v["best_z"]


def phase_A(pool):
    L = json.load(open(f"{OUT}/ladders.json")); jobs = []
    for k, v in L.items():
        if v["max_case"] is None: continue
        ob, ms, st, z = case_z(v["max_case"]); jobs += [(v["max_case"], ob, ms, st, z, dt, law) for dt in (0.001, 0.0005) for law in (True, False)]
    out = {}
    for tag, dt, law, r in pool.imap_unordered(_abl, jobs):
        ob, ms, st, _ = case_z(tag); out.setdefault(tag, {})[f"{'law' if law else 'lawoff'}_{dt}"] = dict(keep(r), gate=HC.gate(r, 10.0, st, ms))
    json.dump(HS.J(out), open(f"{OUT}/ablation_at_max.json", "w"), indent=1); return out


def phase_R(tag):
    from engine import render_any as RA
    ob, ms, st, z = case_z(tag)
    for law in (True, False):
        r = HC.rollout(ob, ms, st, np.asarray(z), 10.0, 0.001, True, None, law=law)
        out = f"{OUT}/{tag}{'' if law else '_lawoff'}.mp4"; np.save(out.replace(".mp4", "_trace.npy"), r["trace"])
        what = ("on the head, palms balancing" if st == "head" else "between the palms") + ("" if law else " - LAW OFF (ablation)")
        RA.render(r["scene"], r["trace"], r["trace_dt"], out, title=f"G1H (stock model): {ms:g} kg {ob} {what}",
                  sub=f"held {r['t_held']:.1f}/10 s ({r['reason'] or 'gates pass'}), head {100*r['head_frac']:.0f}%, feet travel {1000*r['foot_travel_max']:.0f} mm, pelvis drift {1000*r['pelvis_drift']:.0f} mm",
                  azimuth=150, distance=2.6, elevation=-8)
        HS.log(f"render {out}: {r['t_held']} {r['reason']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=7); ap.add_argument("--phase", default="JLA"); ap.add_argument("--budget", type=int, default=600); ap.add_argument("--render", default=None)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.workers) as pool:
        if "J" in a.phase: phase_J(pool); os.system("~/bin/notify 'T1: re-judge of stored head-carry results done' 2>/dev/null")
        if "L" in a.phase: phase_L(pool, a.budget)
        if "A" in a.phase: phase_A(pool)
    if a.render: phase_R(a.render)
    HS.log("T1 driver finished " + a.phase)
