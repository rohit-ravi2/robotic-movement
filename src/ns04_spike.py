"""Phase 10 B2: where in the movement do the analytic-prior (NS-04) pre-takeoff GRF spikes occur,
and which geoms carry them? Replays stored solutions at 1 ms with a contact-logging ctrl_fn wrapper."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, branch_i as BI, mocap_baseline as MB, aprior_standard as AS
G = 9.80665

def replay(clip, ref, x, om, ts, dt=0.001, mods=None):
    S = BI.setup(clip["start"]); BI.apply_dt(S, dt); m = S["m"]
    saved = {}
    if mods:
        for k, v in mods.items(): saved[k] = np.array(getattr(m, k) if not k.startswith("opt.") else getattr(m.opt, k[4:])); 
        for k, v in mods.items():
            if k.startswith("opt."): setattr(m.opt, k[4:], v)
            else: getattr(m, k)[:] = v
    tr = MB.Tracker(S, ref, omega=om, tscale=ts); log = []
    fl = S["floor"]
    def fn(t, d, S_):
        rows = []
        for ci in range(d.ncon):
            c = d.contact[ci]; fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv)
            o = c.geom2 if c.geom1 == fl else (c.geom1 if c.geom2 == fl else None)
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, o) if o is not None else "self:%s/%s" % (
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2))
            rows.append((nm, float(fv[0]), float(c.dist), c.pos.copy()))
        log.append((t, rows, d.qpos.copy(), d.qvel.copy()))
        return tr(t, d, S_)
    r = BI.rollout(x, set(S["types"]), None, record=True, S=S, ctrl_fn=fn, horizon=ref["horizon"], score="best")
    for k, v in saved.items():
        if k.startswith("opt."): setattr(m.opt, k[4:], v)
        else: getattr(m, k)[:] = v
    BI.apply_dt(S, 0.004)
    return r, log, S

if __name__ == "__main__":
    d = json.load(open("../results/phase9/baseline9.json"))
    r = [x for x in d if x["method"] == "APRIOR" and x["clip"].startswith("obstacles4") and x["seed"] == 5][0]
    c = MB.load_clip(r["clip"], "retargeted_g1_all.pkl"); ref = MB.analytic_clip(c)
    o, log, S = replay(c, ref, np.array(r["x"]), r["omega"], r["tscale"])
    Mg = S["M"]*G; print("v", round(o["v"], 3), "stored", round(r["v"], 3), "grf_bw", round(o["grf_bw"], 1))
    fz = np.array(o["trace"]["fz"]); i = int(np.argmax(fz[:int(o["mech"]["T_push"]/0.001)]))
    m = S["m"]; names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    for k in range(max(0, i-12), i+8, 2):
        t, rows, q, qd = log[k]
        agg = {}
        for nm, f, dist, pos in rows: agg[nm] = agg.get(nm, 0) + f
        print(f"t {t:.3f} fz/BW {fz[k]/Mg:5.1f} ", {a: round(b/Mg, 1) for a, b in agg.items() if abs(b) > 0.05*Mg})
    t, rows, q, qd = log[i]
    # joint-limit proximity at the spike
    for j in range(1, m.njnt):
        if m.jnt_limited[j]:
            qa = q[m.jnt_qposadr[j]]; lo, hi = m.jnt_range[j]
            if min(qa-lo, hi-qa) < 0.03: print("  near limit:", names[j], round(qa, 3), (round(lo, 2), round(hi, 2)), "qvel", round(qd[m.jnt_dofadr[j]], 2))
    print("base z", round(q[2], 3), "base vz", round(qd[2], 3), "takeoff", o["mech"]["T_push"])
