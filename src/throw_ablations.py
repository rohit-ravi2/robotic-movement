"""Small, matched-budget prior ablation campaign.

This is an executable pilot/confirmatory seed set, isolated from historical Phase 11
files. Every arm uses the same CMA-ES budget and the same two-timestep validity gate.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import branch_i as BI
import throw as T

OUT = Path(__file__).resolve().parents[1] / "results" / "throw_ablations_v1"
MODES = ("native", "power_prior", "geometry_prior", "random_prior", "flat_prior")


def flat_ref(ref):
    return dict(ref, Q=np.repeat(ref["start"][None], len(ref["Q"]), axis=0), windup=[ref["start"]], release=ref["start"].tolist())


def run(a):
    S = T.setup("G1"); BI.apply_dt(S, T.DT_OPT)
    t0 = time.time()
    if a["mode"] == "native":
        x = T.solve(S, a["seed"], iters=a["iters"], pop=16)
        r = T.measure(S, x)
        prior = None
    else:
        ref = T.windup_reference(S, capacity_mode={"power_prior": "power", "geometry_prior": "geometry", "random_prior": "random", "flat_prior": "power"}[a["mode"]], seed=a["seed"])
        if a["mode"] == "flat_prior": ref = flat_ref(ref)
        x, om, ts, amp = T.solve_prior(S, ref, a["seed"], iters=a["iters"], pop=16)
        r = T.measure(S, x, ctrl_fn_factory=lambda: T.tracker(S, T.scaled_ref(ref, amp), om, ts))
        prior = {"capacity_mode": a["mode"], "omega": om, "tscale": ts, "amp": amp}
    r.pop("trace", None)
    # final_state is useful for an in-memory controller but is not part of this
    # compact campaign artifact and contains large non-JSON NumPy arrays.
    r.pop("final_state", None)
    return {**a, "result": r, "prior": prior, "seconds": time.time() - t0}


def main():
    iters = int(os.environ.get("THROW_ABLATION_ITERS", "150"))
    seeds = [int(x) for x in os.environ.get("THROW_ABLATION_SEEDS", "0,1").split(",")]
    jobs = [dict(mode=m, seed=s, iters=iters) for m in MODES for s in seeds]
    OUT.mkdir(parents=True, exist_ok=True)
    def path(a): return OUT / f"{a['mode']}_s{a['seed']}_i{a['iters']}.json"
    pending = [a for a in jobs if not path(a).exists()]
    rows = []
    for a in jobs:
        p = path(a)
        if p.exists(): rows.append(json.loads(p.read_text()))
    with Pool(int(os.environ.get("NPROC", "2"))) as pool:
        for row in pool.imap_unordered(run, pending):
            a = {"mode": row["mode"], "seed": row["seed"], "iters": row["iters"]}
            path(a).write_text(json.dumps(row, indent=2, default=float) + "\n")
            rows.append(row)
    summary_path = OUT / f"campaign_{iters}.json"
    summary_path.write_text(json.dumps(rows, indent=2, default=float) + "\n")
    for m in MODES:
        g = [x["result"] for x in rows if x["mode"] == m]
        valid = [x["speed"] for x in g if x.get("valid_all_dt")]
        print(m, "valid", len(valid), "/", len(g), "speeds", [round(x, 3) for x in valid])
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
