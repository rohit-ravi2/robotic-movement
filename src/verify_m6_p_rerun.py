import sys, json, time; sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "src"))
from engine import core, walk_headcarry as WH
def job(a):
    case, nm, dt = a
    g = json.load(open(f"{WH.RES}/search/{case}.json"))
    C = WH.Carry(g["obj"], g["mass"], g["v"], dt_list=(dt,), uid=2, law_base=g["law_base"])
    r = C.run(g["x"], dt, **WH.pert_kw(nm, len(C.S[dt]["types"]))); return case, nm, dt, r["valid"], r["reasons"], round(r["value"],4), r["load"]["t_lost"]
if __name__ == "__main__":
    import multiprocessing as mp
    jobs = [(c, nm, dt) for c in ["pot_20kg_v0.6_s1", "pot_15kg_v0.6_s1", "box_20kg_v0.6_s1"] for nm in WH.PERT for dt in (core.DT_OPT, core.DT_CHECK)]
    with mp.get_context("spawn").Pool(3, maxtasksperchild=1) as p: res = p.map(job, jobs)
    for x in res: print(*x, flush=True)
    json.dump(res, open(f"{WH.RES}/verify_physics/p_rerun_uid2.json", "w"), indent=1, default=float)
