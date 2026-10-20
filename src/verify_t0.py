"""VERIFY T0 spec audit (2026-10-17): (1) independent diff of the ACTUAL runtime plants of two headline results against the stock
Menagerie model (robot bodies only, incl. fields spec_audit.py skips: gravcomp, gravity, density/viscosity, jnt margin/ref);
(2) fresh-plant replays at 1 ms and 0.5 ms: head-carry 34 kg pot (stored best_z) and G1 sprint 2.0 rung (2 ms lag), plus the
same sprint z with the G1 continuous (URDF-effort) ratings as a conservative replay. Output results/overnight/verify_t0/."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SPRINT_LAG", "0.002")
import numpy as np, mujoco
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); MEN = os.environ.get("MENAGERIE_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "models", "mujoco_menagerie"))
OUT = f"{ROOT}/results/overnight/verify_t0"; os.makedirs(OUT, exist_ok=True)


def robot_bodies(m):
    rb = [b for b in range(1, m.nbody) if m.body_parentid[b] == 0 and m.body_jntnum[b] and m.jnt_type[m.body_jntadr[b]] == 0][0]
    return [b for b in range(m.nbody) if m.body_rootid[b] == rb]


def phys(m):
    B = robot_bodies(m); Bs = set(B); n = lambda t, i: mujoco.mj_id2name(m, t, i)
    J = [j for j in range(m.njnt) if m.jnt_bodyid[j] in Bs]; D = [d for d in range(m.nv) if m.jnt_bodyid[m.dof_jntid[d]] in Bs]
    G = [g for g in range(m.ngeom) if m.geom_bodyid[g] in Bs and m.geom_contype[g] | m.geom_conaffinity[g]]
    return dict(bodies={n(1, b): dict(mass=m.body_mass[b], inertia=m.body_inertia[b].copy(), ipos=m.body_ipos[b].copy(), gravcomp=m.body_gravcomp[b]) for b in B},
                joints={(n(3, j) or "free"): dict(range=m.jnt_range[j].copy(), limited=int(m.jnt_limited[j]), stiffness=m.jnt_stiffness[j], margin=m.jnt_margin[j], ref=m.qpos0[m.jnt_qposadr[j]]) for j in J},
                dofs={(n(3, m.dof_jntid[d]) or "free") + f"/{d - m.jnt_dofadr[m.dof_jntid[d]]}": dict(damping=m.dof_damping[d], armature=m.dof_armature[d], frictionloss=m.dof_frictionloss[d]) for d in D},
                geoms=[dict(body=n(1, m.geom_bodyid[g]), name=n(5, g), type=int(m.geom_type[g]), size=m.geom_size[g].round(5).tolist(), friction=m.geom_friction[g].round(5).tolist(),
                            solref=m.geom_solref[g].round(5).tolist(), solimp=m.geom_solimp[g].round(5).tolist(), condim=int(m.geom_condim[g])) for g in G],
                opt=dict(gravity=m.opt.gravity.tolist(), density=m.opt.density, viscosity=m.opt.viscosity, wind=m.opt.wind.tolist(), cone=int(m.opt.cone), impratio=m.opt.impratio, noslip=int(m.opt.noslip_iterations)),
                mass=float(sum(m.body_mass[b] for b in B)))


def compare(a, b):
    dev = []
    for k, va in a["bodies"].items():
        vb = b["bodies"].get(k)
        if vb is None: dev.append(f"body {k} missing"); continue
        for f in va:
            if not np.allclose(va[f], vb[f], rtol=1e-6, atol=1e-9): dev.append(f"body {k}.{f} {np.round(va[f], 5).tolist()} -> {np.round(vb[f], 5).tolist()}")
    for grp in ("joints", "dofs"):
        for k, va in a[grp].items():
            vb = b[grp].get(k)
            if vb is None: dev.append(f"{grp} {k} missing"); continue
            for f in va:
                if not np.allclose(va[f], vb[f], rtol=1e-6, atol=1e-9): dev.append(f"{grp} {k}.{f} {np.round(va[f], 5).tolist()} -> {np.round(vb[f], 5).tolist()}")
    ga = {(g["body"], i): g for i, g in enumerate(a["geoms"])}; gb = [g for g in b["geoms"]]
    # geoms matched by (body, order within body)
    def per_body(L):
        d = {}
        for g in L: d.setdefault(g["body"], []).append(g)
        return d
    pa, pb = per_body(a["geoms"]), per_body(b["geoms"])
    for body, La in pa.items():
        Lb = [g for g in pb.get(body, []) if not (g["name"] or "").startswith("eff_")]
        if len(La) != len(Lb): dev.append(f"geoms on {body}: {len(La)} -> {len(Lb)}"); continue
        for x, y in zip(La, Lb):
            for f in ("type", "size", "friction", "solref", "solimp", "condim"):
                if x[f] != y[f]: dev.append(f"geom {body}/{x['name']}.{f} {x[f]} -> {y[f]}")
    added = [g for body, L in pb.items() for g in L if (g["name"] or "").startswith("eff_")]
    if added: dev.append(f"added eff colliders {[(g['name'], g['body']) for g in added]}")
    for f, v in a["opt"].items():
        if not np.allclose(v, b["opt"][f]): dev.append(f"opt.{f} {v} -> {b['opt'][f]}")
    if abs(a["mass"] - b["mass"]) > 1e-6: dev.append(f"robot mass {a['mass']:.4f} -> {b['mass']:.4f}")
    return dev


def ratings(S, m_stock):
    out = {}
    for i, t in enumerate(S["types"]):
        jid = S["tmap"][t][0][0]; nm = mujoco.mj_id2name(S["m"], 3, jid); js = mujoco.mj_name2id(m_stock, 3, nm)
        stock = float(m_stock.jnt_actfrcrange[js][1]) if js >= 0 and m_stock.jnt_actfrclimited[js] else None
        if stock is None and js >= 0:
            a = [x for x in range(m_stock.nu) if m_stock.actuator_trnid[x][0] == js]; stock = float(m_stock.actuator_forcerange[a[0]][1]) if a and m_stock.actuator_forcelimited[a[0]] else None
        aid = int(S["jarr"]["aid"][i]) if "aid" in S["jarr"] else None
        out[nm] = dict(tau_used=float(S["jarr"]["tau"][i]), stock=stock, plant_forcerange=float(S["m"].actuator_forcerange[aid][1]) if aid is not None else None, om=float(S["jarr"]["om"][i]))
    return out


# ------------------------------------------------------------------ jobs
def job(a):
    kind, dt = a[0], a[1]; t0 = time.time()
    if kind == "hc":
        from engine import headcarry as HC
        C = json.load(open(f"{ROOT}/results/overnight/A/head_pot_34kg.json"))
        S, path = HC.plant("pot", 34.0, dt); stock = mujoco.MjModel.from_xml_path(f"{MEN}/unitree_g1/scene_with_hands.xml")
        dev = compare(phys(stock), phys(S["m"])); rat = ratings(S, stock)
        r = HC.evaluate(("pot", 34.0, "head", np.asarray(C["best_z"]), C["T"], dt, C.get("sway")))
        keep = {k: r.get(k) for k in ("t_held", "reason", "head_frac", "tilt_max_deg", "foot_travel_max", "pelvis_drift", "util_p95")}
        return dict(kind=kind, dt=dt, gate=HC.gate(r, C["T"], "head", 34.0), gate_nosupport=HC.gate(r, C["T"], "head", 34.0, support=False), res=keep,
                    plant_dev=dev, ratings_mismatch={k: v for k, v in rat.items() if v["stock"] is not None and abs(v["tau_used"] - v["stock"]) > 1e-6},
                    timestep=S["m"].opt.timestep, scene=path, secs=time.time() - t0)
    from engine import sprint as SP
    L = json.load(open(f"{ROOT}/results/sprint/ladder_G1_lag2ms.json")); R = [x for x in L["rungs"] if x["v"] == 2.0][0]
    P = SP.Sprint("G1", 2.0); S = P.S[dt]; stock = mujoco.MjModel.from_xml_path(f"{MEN}/unitree_g1/scene_mjx.xml")
    dev = compare(phys(stock), phys(S["m"])); rat = ratings(S, stock)
    if kind == "sp_cont":   # URDF-effort (continuous) set where it differs from the MJCF peak: ankles, waist roll/pitch 35, hip roll 88
        for i, t in enumerate(S["types"]):
            if any(k in t for k in ("ankle_pitch", "ankle_roll", "waist_roll", "waist_pitch")): S["jarr"]["tau"][i] = min(S["jarr"]["tau"][i], 35.0)
            if "hip_roll" in t: S["jarr"]["tau"][i] = min(S["jarr"]["tau"][i], 88.0)
    r = P.run(np.asarray(R["z"]), dt)
    s = SP.summarize(r)
    return dict(kind=kind, dt=dt, act_tau=SP.ACT_TAU, res={k: s[k] for k in ("value", "valid", "reasons", "slip_max", "overspeed", "periodic")},
                stored=[(m_["value"], m_["valid"]) for m_ in R["meas"]], plant_dev=dev,
                ratings_mismatch={k: v for k, v in rat.items() if v["stock"] is not None and abs(v["tau_used"] - v["stock"]) > 1e-6}, secs=time.time() - t0)


if __name__ == "__main__":
    jobs = [("hc", 0.001), ("hc", 0.0005), ("sp", 0.001), ("sp", 0.0005), ("sp_cont", 0.001), ("sp_cont", 0.0005)]
    res = []
    with Pool(2, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(job, jobs):
            res.append(r); print(json.dumps({k: v for k, v in r.items() if k != "plant_dev"}, default=str)[:600], "| dev:", r["plant_dev"][:12], flush=True)
            json.dump(res, open(f"{OUT}/verify_t0.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("done")
