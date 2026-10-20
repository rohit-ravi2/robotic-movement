"""Phase 10 Gate A on the SOURCED spec (spec/body_sourced.yaml, committed 800e519 before this run),
plus the P15-corrected INVENTED spec to separate the defect fix from the sourcing.
Each job carries its spec; the result is accepted as it lands."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase10")
SPECS = {"sourced": os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "spec/body_sourced.yaml"),
         "invented_fixed": os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "spec/body.yaml")}

def solve(job):
    seed, iters, pop, squat, extra = job
    os.environ["MLAW_HUMAN_SPEC"] = SPECS[extra["spec"]]
    import gate_a_ladder as L
    r = L.solve((seed, iters, pop, squat, {k: v for k, v in extra.items() if k != "spec"}))
    r["extra"]["spec"] = extra["spec"]; return r

def run(name, jobs, workers=int(os.environ.get("NPROC", "14"))):
    jobs = sorted(jobs, key=lambda j: -j[1]*j[2])
    with Pool(workers) as p: res = p.map(solve, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/{name}.json", "w"), default=float)
    return res

if __name__ == "__main__":
    what = sys.argv[1]
    if what == "smoke":
        t = time.time(); r = solve((1, 10, 16, True, {"spec": "sourced"}))
        print("sourced 160 evals", round(time.time()-t, 1), "s  h", round(r["height"]*100, 1), "cm valid", r["valid"])
        t = time.time(); r = solve((1, 10, 16, True, {"spec": "invented_fixed"}))
        print("invented_fixed 160 evals", round(time.time()-t, 1), "s  h", round(r["height"]*100, 1), "cm valid", r["valid"])
    if what == "attrib":
        C = {"base": {}, "vel1.5": {"vel": 1.5}, "vel2": {"vel": 2.0}, "vel3": {"vel": 3.0},
             "iso1.31": {"iso": 1.31}, "ankle1.7": {"ankle_iso": 1.7},
             "vel2_iso1.31": {"vel": 2.0, "iso": 1.31}, "vel2_iso1.31_ankle1.7": {"vel": 2.0, "iso": 1.31, "ankle_iso": 1.7}}
        jobs = [(s, 300, 16, True, {"spec": "sourced", "tag": k, "anderson_scale": v}) for k, v in C.items() for s in range(1, 9)]
        jobs += [(s, 300, 16, False, {"spec": "sourced", "tag": "cmj_" + k, "anderson_scale": v}) for k, v in C.items() if k in ("vel2", "vel2_iso1.31_ankle1.7") for s in range(1, 9)]
        run("gate_a10_attrib", jobs)
    if what == "p17":
        # re-optimisation under the P17-corrected (sustained-flight) takeoff rule
        jobs = [(s, 1200, 16, True, {"spec": "sourced", "tag": "ladder"}) for s in range(1, 17)]
        jobs += [(s, 1200, 16, True, {"spec": "invented_fixed", "tag": "ladder"}) for s in range(1, 9)]
        jobs += [(s, 1200, 16, False, {"spec": sp, "tag": "cmj"}) for sp in SPECS for s in range(1, 9)]
        for k, v in (("vel2", {"vel": 2.0}), ("vel2_iso1.31", {"vel": 2.0, "iso": 1.31})):
            jobs += [(s, 300, 16, True, {"spec": "sourced", "tag": k, "anderson_scale": v}) for s in range(1, 9)]
            jobs += [(s, 300, 16, False, {"spec": "sourced", "tag": "cmj_" + k, "anderson_scale": v}) for s in range(1, 9)]
        run("gate_a10_p17", jobs)
    if what == "main":
        jobs = []
        for it, pop, seeds in ((75, 16, 16), (300, 16, 16), (1200, 16, 16), (3000, 16, 8), (3000, 32, 8)):
            jobs += [(s, it, pop, True, {"spec": "sourced", "tag": "ladder"}) for s in range(1, seeds+1)]
        for it, pop in ((1200, 16), (3000, 32)):
            jobs += [(s, it, pop, True, {"spec": "invented_fixed", "tag": "ladder"}) for s in range(1, 9)]
        for sp in SPECS:
            jobs += [(s, 1200, 16, False, {"spec": sp, "tag": "cmj"}) for s in range(1, 9)]
        run("gate_a10_main", jobs)
