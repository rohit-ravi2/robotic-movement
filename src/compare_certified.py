"""Old (uncertified) vs certified pose bank: every analytic branch's headline numbers side by side."""
import json, numpy as np, sys
import os; R = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results")
NEWDIR = sys.argv[1] if len(sys.argv) > 1 else "certified"
def L(p): return json.load(open(p))
def flat(x, pre=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items(): out.update(flat(v, f"{pre}.{k}" if pre else str(k)))
    elif isinstance(x, (int, float)) and not isinstance(x, bool): out[pre] = float(x)
    elif isinstance(x, list) and x and all(isinstance(v, (int, float)) for v in x) and len(x) <= 12:
        for i, v in enumerate(x): out[f"{pre}[{i}]"] = float(v)
    return out
rows = []
for f, keys in (("analysis_BEF.json", ("B", "E", "F")), ("c2d2e3.json", ("C2", "E3")),
                ("analysis_CD.json", ("collapse_all", "collapse_interp", "sensitivity"))):
    o, n = L(f"{R}/{f}"), L(f"{R}/{NEWDIR}/{f}")
    for k in keys:
        fo, fn = flat(o.get(k, {}), k), flat(n.get(k, {}), k)
        for kk in sorted(set(fo) | set(fn)):
            a, b = fo.get(kk), fn.get(kk)
            rows.append((f, kk, a, b))
o, n = L(f"{R}/c2d2e3.json")["D2"]["counts"], L(f"{R}/{NEWDIR}/c2d2e3.json")["D2"]["counts"]
rows.append(("c2d2e3.json", "D2.counts", json.dumps(o), json.dumps(n)))
o, n = len(L(f"{R}/analysis_CD.json")["bifurcations"]), len(L(f"{R}/{NEWDIR}/analysis_CD.json")["bifurcations"])
rows.append(("analysis_CD.json", "n_bifurcations_raw", o, n))
for r in rows:
    a, b = r[2], r[3]
    if isinstance(a, float) and isinstance(b, float):
        ch = "" if abs(a-b) < 1e-9 else f"  delta {b-a:+.4g}"
        print(f"{r[0][:14]:<14} {r[1][:70]:<70} {a:>10.4g} -> {b:<10.4g}{ch}")
    else:
        print(f"{r[0][:14]:<14} {r[1][:70]:<70} {a} -> {b}")
