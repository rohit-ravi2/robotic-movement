"""Phase 10 B3: contact / joint-limit / timestep sensitivity of stored NS-04 (APRIOR) and native
solutions. Same stored control, replayed under modified numerics. Peak GRF should scale with
stiffness if it is an impact transient; takeoff v should not move if the result is robust."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, branch_i as BI, mocap_baseline as MB
from multiprocessing import Pool
OUT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase10")
G = 9.80665

def variants(m):
    import mujoco
    ps = np.array(m.pair_solref); js = np.array(m.jnt_solref)
    fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_pairs = [i for i in range(m.npair) if fl in (m.pair_geom1[i], m.pair_geom2[i])]
    V = {"base": {}}
    for tc in (0.002, 0.008, 0.016):
        a = ps.copy(); a[floor_pairs, 0] = tc; V[f"contact_tc{tc}"] = {"pair_solref": a}
    for tc in (0.004, 0.02, 0.05):
        a = js.copy(); a[:, 0] = tc; V[f"limit_tc{tc}"] = {"jnt_solref": a}
    V["cone_elliptic"] = {"opt.cone": 1}
    V["impratio10"] = {"opt.impratio": 10.0, "opt.cone": 1}
    return V

def job(a):
    import mujoco
    c = MB.load_clip(a["clip"], "retargeted_g1_all.pkl") if a["clip"] != "knees_bent" else None
    if c is None:
        import aprior_standard as AS; c = AS.clip0()
    ref = MB.analytic_clip(c) if a["method"] in ("APRIOR",) else c
    S = BI.setup(c["start"]); V = variants(S["m"])
    out = []
    for name in a["variants"]:
        mods = V.get(name, {}); dt = 0.001
        if name == "dt0.0005": mods = {}; dt = 0.0005
        BI.apply_dt(S, dt); m = S["m"]; saved = {}
        for k, v in mods.items():
            if k.startswith("opt."): saved[k] = getattr(m.opt, k[4:]); setattr(m.opt, k[4:], v)
            else: saved[k] = np.array(getattr(m, k)); getattr(m, k)[:] = v
        o = MB.run(S, ref, np.array(a["x"]), a["omega"], a["tscale"])
        for k, v in saved.items():
            if k.startswith("opt."): setattr(m.opt, k[4:], v)
            else: getattr(m, k)[:] = v
        BI.apply_dt(S, 0.004)
        out.append(dict(variant=name, v=o["v"], grf_bw=o["grf_bw"], reasons=o["validity"]["reasons"],
                        rot=o["mech"]["flight_rotation"], lim_exc=o["validity"]["joint_limit_excursion"]))
    return dict(a, x=None, res=out)

if __name__ == "__main__":
    d = json.load(open("../results/phase9/baseline9.json"))
    names = ["base", "contact_tc0.002", "contact_tc0.008", "contact_tc0.016", "limit_tc0.004", "limit_tc0.02",
             "limit_tc0.05", "cone_elliptic", "impratio10", "dt0.0005"]
    jobs = []
    for meth in ("APRIOR", "OURS", "B1"):
        rows = [r for r in d if r["method"] == meth and r["valid"] and r["task"] == "T2"]
        for r in rows[::2] if meth == "APRIOR" else rows[::8]:
            jobs.append(dict(clip=r["clip"], method=meth, seed=r["seed"], x=r["x"], omega=r["omega"], tscale=r["tscale"], v_stored=r["v"], variants=names))
    a = json.load(open("../results/phase9/aprior_standard.json"))
    for r in [r for r in a if r["method"] == "APRIOR" and r["valid"]]:
        pass  # aprior_standard stores no omega/tscale; covered by baseline9 APRIOR arm
    print(len(jobs), "jobs", flush=True)
    with Pool(int(os.environ.get("NPROC", "12"))) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(f"{OUT}/ns04_sensitivity.json", "w"), default=float)
    for r in res:
        b = r["res"][0]
        print(r["method"], r["clip"][:10], r["seed"], " | ".join(f"{z['variant']}: v {z['v']:.2f} grf {z['grf_bw']:.1f}{'*' if [q for q in z['reasons'] if q!='CONTACT_ENERGY'] else ''}" for z in r["res"]), flush=True)
