"""J3: atlas store. One JSONL file per (robot, task); a record = demand, solution z, params, faithful measures at both
timesteps, ok_all, value, cost, z_hash, provenance. Append-only; envelopes and costs are derived views."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from engine import core
ATLAS = f"{core.RES}/atlas"

def path(robot, task): os.makedirs(ATLAS, exist_ok=True); return f"{ATLAS}/{robot}_{task}.jsonl"

def append(robot, task, rec, **prov):
    rec = dict(rec, created=time.strftime("%Y-%m-%d %H:%M:%S"), **prov)
    with open(path(robot, task), "a") as f: f.write(json.dumps(rec, default=float) + "\n")

def load(robot, task):
    p = path(robot, task)
    if not os.path.exists(p): return []
    return [json.loads(l) for l in open(p) if l.strip()]

def key(demand): return json.dumps(demand, sort_keys=True, default=float)

def envelope(records, axes):
    """Per demand point: best faithful value among ok_all solutions, n seeds, n ok, cost of the best."""
    out = {}
    for r in records:
        k = tuple(r["demand"].get(a) for a in axes); e = out.setdefault(k, dict(n=0, n_ok=0, best=None, cost=None, seeds=[]))
        e["n"] += 1; e["seeds"].append(r.get("seed"))
        if r["ok_all"]:
            e["n_ok"] += 1
            if e["best"] is None or r["value"] > e["best"]: e["best"] = r["value"]; e["cost"] = r["meas"][0]["cost"]
    return out
