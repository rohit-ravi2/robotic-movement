"""T2: predicted vs found for the pre-registered new cases (results/overnight/T2/PREDICTIONS.md, commit 1c85c8f).
-> results/overnight/T2/summary.json and a markdown table on stdout."""
import os, json, glob
import numpy as np
T2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "overnight", "T2")
SPR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "sprint")
PRED = dict(head_tallcyl=11.29, head_crate=84.88, push_mu0_6=9.0, push_mu0_25=52.5, pull_mu0_6=48.0, pull_mu0_25=82.5, sprint_H1_lag10=3.4)
rows = []


def head(obj, ladder):
    out = []
    for m in ladder:
        p = f"{T2}/head_{obj}_{m:g}kg.json"
        if not os.path.exists(p): out.append(dict(mass=m, ran=False)); continue
        d = json.load(open(p)); c = d["check"]; lo = d.get("law_off_1ms") or {}
        out.append(dict(mass=m, ran=True, valid=d["valid"], held=[c["dt1ms"].get("t_held"), c["dt05ms"].get("t_held")], reason=[c["dt1ms"].get("reason"), c["dt05ms"].get("reason")],
                        slide=[c["dt1ms"].get("foot_slide_max"), c["dt05ms"].get("foot_slide_max")], tilt=[c["dt1ms"].get("tilt_max_deg"), c["dt05ms"].get("tilt_max_deg")],
                        joint_sat_05=c["dt05ms"].get("joint_sat"), law_off=[lo.get("t_held"), lo.get("reason")], evals=d["evals"], wall_min=d["wall_s"]/60))
    return out


def sled(nm, mu, payloads):
    out = []
    for p_ in payloads:
        p = f"{T2}/sled_{nm}_mu{mu:g}_{p_:g}kg.json"
        if not os.path.exists(p): out.append(dict(total=p_ + 1, ran=False)); continue
        d = json.load(open(p)); out.append(dict(total=p_ + 1, ran=True, valid=d["valid"], robust=d["robust"], disp_seed0=d["fresh"][0]["disp"], reason_seed0=d["fresh"][0]["reason"], wall_min=d["wall_s"]/60))
    return out


def found_of(cases, key="mass"):
    ran = [c for c in cases if c.get("ran")]
    if not ran: return None, "not run"
    v = [c[key] for c in ran if c["valid"]]
    if v: return max(v), ("largest valid tested = TOP of the ladder (lower bound)" if max(v) >= max(c[key] for c in ran) else f"largest valid tested (next rung {min([c[key] for c in ran if c[key] > max(v)]):g} invalid)")
    return None, f"none valid (< {min(c[key] for c in ran):g})"


summary = {}
for obj, lad in (("tallcyl", (5.5, 8, 11.3, 17)), ("crate", (21, 42, 63, 85))):
    cs = head(obj, lad); f, why = found_of(cs); summary[f"head_{obj}"] = dict(predicted=PRED[f"head_{obj}"], found=f, found_note=why, cases=cs)
for nm, mu, pl in (("push", 0.6, (4.0, 8.0)), ("pull", 0.6, (23.0, 47.0)), ("push", 0.25, (25.25, 51.5)), ("pull", 0.25, (40.25, 81.5))):
    cs = sled(nm, mu, pl); f, why = found_of(cs, "total"); summary[f"{nm}_mu{str(mu).replace('.', '_')}"] = dict(predicted=PRED[f"{nm}_mu{str(mu).replace('.', '_')}"], found=f, found_note=why, cases=cs)
sp = f"{SPR}/ladder_H1_T2_lag10ms.json"
if os.path.exists(sp):
    d = json.load(open(sp)); rs = []
    for r in d["rungs"]:
        ms = r.get("meas") or []
        rs.append(dict(v=r["v"], ok=r.get("ok_both_dt"), v_meas=[m_["value"] for m_ in ms], reasons=[m_["reasons"] for m_ in ms], cadence=[m_.get("cadence_hz") for m_ in ms], note=r.get("note")))
    ok = [r for r in rs if r["ok"]]
    summary["sprint_H1_lag10"] = dict(predicted=PRED["sprint_H1_lag10"], found=(min(ok[-1]["v_meas"]) if ok else None), found_note=("worse-dt speed at the highest valid rung" if ok else "no valid rung"), rungs=rs)
else:
    summary["sprint_H1_lag10"] = dict(predicted=PRED["sprint_H1_lag10"], found=None, found_note="not run")
lr = []
print("| case | predicted | found | ratio pred/found | within 1.5x | note |\n|---|---|---|---|---|---|")
for k, v in summary.items():
    f = v["found"]; r = v["predicted"]/f if f else None
    if r is None and "none valid" in v["found_note"]:
        lo = float(v["found_note"].split("< ")[1].rstrip(")")); rtxt = f"> {v['predicted']/lo:.2f}"; w = "no"; lr.append(abs(np.log(v["predicted"]/lo)))
    elif r is not None and "lower bound" in v["found_note"]:
        rtxt = f"<= {r:.2f}"; w = ("no" if r < 1/1.5 else "undetermined (<= 1)"); lr.append(abs(np.log(r)))
    elif r is not None:
        nxt = v["found_note"].split("next rung ")[1].split(" ")[0] if "next rung" in v["found_note"] else None
        rtxt = f"{r:.2f}" + (f" (true in ({v['predicted']/float(nxt):.2f}, {r:.2f}])" if nxt else ""); lo_ = v['predicted']/float(nxt) if nxt else r
        w = "yes" if (1/1.5 <= lo_ and r <= 1.5) else ("no" if (lo_ > 1.5 or r < 1/1.5) else "undetermined"); lr.append(abs(np.log(r)))
    else: rtxt, w = "n/a", "n/a"
    v["ratio"] = r; v["within_1p5"] = w
    print(f"| {k} | {v['predicted']} | {f} | {rtxt} | {w} | {v['found_note']} |")
summary["_median_abs_log_ratio"] = float(np.median(lr)) if lr else None
summary["_preregistration_commit"] = "1c85c8f"
json.dump(summary, open(f"{T2}/summary.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
print("median |log ratio| (new cases, bracket endpoints):", summary["_median_abs_log_ratio"], "-> factor", np.exp(summary["_median_abs_log_ratio"]) if lr else None)
