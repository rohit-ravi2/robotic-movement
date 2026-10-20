"""SPEC AUDIT (T0, 2026-10-17): programmatic diff of every derived model copy, and of the runtime plants, against the stock
MuJoCo Menagerie originals. No movement simulation (the runtime check calls throw.setup, whose settle is a 4 s statue hold).

Compared on the ROBOT only (bodies under the floating base, matched by order; names reported if renamed):
  body mass / inertia / ipos / iquat / pos / quat, joint range / limited / stiffness, dof armature / damping / frictionloss,
  joint actfrcrange, robot geom type / size / pos / friction / solref / solimp / condim / contype / conaffinity / priority /
  margin / gap, robot geom count, actuator gain/bias/ctrl/force ranges, contact pairs and excludes touching robot geoms,
  equality constraints, keyframe names, global options.
usage: python spec_audit.py  ->  results/overnight/spec_audit/spec_diff.json"""
import os, sys, json, glob, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco

MEN = os.environ.get("MENAGERIE_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "models", "mujoco_menagerie"))
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = f"{ROOT}/results/overnight/spec_audit"
ORIG = {"G1": f"{MEN}/unitree_g1/scene_mjx.xml", "G1H": f"{MEN}/unitree_g1/scene_with_hands.xml", "H1": f"{MEN}/unitree_h1/scene.xml",
        "Apollo": f"{MEN}/apptronik_apollo/scene.xml", "T1": f"{MEN}/booster_t1/scene.xml"}
DERIVED = {"G1H": f"{MEN}/unitree_g1/scene_hands_palm.xml", "H1": f"{MEN}/unitree_h1/scene_palm.xml",
           "Apollo": f"{ROOT}/results/heldout/models/scene_apollo_heldout.xml", "T1": f"{ROOT}/results/heldout/models/scene_t1_heldout.xml"}
TOL = 1e-6     # relative (XML round trips of derived scenes write ~6 significant digits)


def robot_bodies(m):
    root = [b for b in range(1, m.nbody) if m.body_parentid[b] == 0 and m.body_jntnum[b] and m.jnt_type[m.body_jntadr[b]] == 0]
    # the robot's floating base = the first free body in the tree (world objects are added after it)
    rb0 = root[0]; return [b for b in range(m.nbody) if m.body_rootid[b] == rb0]


def nm(m, t, i): return mujoco.mj_id2name(m, t, i) or f"#{i}"


def collect(m):
    B = robot_bodies(m); Bs = set(B)
    J = [j for j in range(m.njnt) if m.jnt_bodyid[j] in Bs]
    D = [d for d in range(m.nv) if m.jnt_bodyid[m.dof_jntid[d]] in Bs]
    G_all = [g for g in range(m.ngeom) if m.geom_bodyid[g] in Bs]
    EFF = [g for g in G_all if (mujoco.mj_id2name(m, 5, g) or "").startswith("eff_")]     # added effector colliders: reported, not index-compared
    G = [g for g in G_all if g not in EFF]
    A = [a for a in range(m.nu) if m.actuator_trntype[a] == 0 and m.jnt_bodyid[m.actuator_trnid[a][0]] in Bs]
    Gs = set(G)
    pairs = [dict(g1=nm(m, 5, m.pair_geom1[p]), g2=nm(m, 5, m.pair_geom2[p]), condim=int(m.pair_dim[p]), friction=m.pair_friction[p].round(4).tolist(),
                  solref=m.pair_solref[p].round(5).tolist()) for p in range(m.npair) if m.pair_geom1[p] in Gs or m.pair_geom2[p] in Gs]
    return dict(
        body=dict(names=[nm(m, 1, b) for b in B], mass=m.body_mass[B], inertia=m.body_inertia[B], ipos=m.body_ipos[B], iquat=m.body_iquat[B], pos=m.body_pos[B], quat=m.body_quat[B]),
        joint=dict(names=[nm(m, 3, j) for j in J], range=m.jnt_range[J], limited=m.jnt_limited[J].astype(float), stiffness=m.jnt_stiffness[J],
                   actfrcrange=m.jnt_actfrcrange[J], actfrclimited=m.jnt_actfrclimited[J].astype(float)),
        dof=dict(armature=m.dof_armature[D], damping=m.dof_damping[D], frictionloss=m.dof_frictionloss[D]),
        geom=dict(names=[nm(m, 5, g) for g in G], type=m.geom_type[G].astype(float), size=m.geom_size[G], pos=m.geom_pos[G], friction=m.geom_friction[G],
                  solref=m.geom_solref[G], solimp=m.geom_solimp[G], condim=m.geom_condim[G].astype(float), contype=m.geom_contype[G].astype(float),
                  conaffinity=m.geom_conaffinity[G].astype(float), priority=m.geom_priority[G].astype(float), margin=m.geom_margin[G], gap=m.geom_gap[G]),
        act=dict(names=[nm(m, 19, a) for a in A], gaintype=m.actuator_gaintype[A].astype(float), biastype=m.actuator_biastype[A].astype(float),
                 gainprm=m.actuator_gainprm[A][:, :3], biasprm=m.actuator_biasprm[A][:, :3], ctrlrange=m.actuator_ctrlrange[A], forcerange=m.actuator_forcerange[A]),
        pairs=pairs, nexclude=int(m.nexclude), neq=int(m.neq), eq=[(nm(m, 17, e), int(m.eq_type[e])) for e in range(m.neq)],
        keys=[nm(m, 8, k) for k in range(m.nkey)],
        opt=dict(timestep=m.opt.timestep, integrator=int(m.opt.integrator), cone=int(m.opt.cone), impratio=m.opt.impratio, iterations=int(m.opt.iterations),
                 noslip=int(m.opt.noslip_iterations), o_solref=m.opt.o_solref.tolist(), disableflags=int(m.opt.disableflags), enableflags=int(m.opt.enableflags)),
        eff=[dict(name=nm(m, 5, g), body=nm(m, 1, m.geom_bodyid[g]), type=int(m.geom_type[g]), size=m.geom_size[g].round(4).tolist(), mass_body=float(m.body_mass[m.geom_bodyid[g]])) for g in EFF],
        robot_mass=float(m.body_mass[B].sum()), n=dict(body=len(B), joint=len(J), geom=len(G), eff_collider_geoms=len(EFF), act=len(A), site=int(sum(1 for s in range(m.nsite) if m.site_bodyid[s] in Bs))))


def diff(a, b, label_a="orig", label_b="derived"):
    """Field-wise differences of two collect() dicts (robot only). Returns a list of human-readable deviations + a structured dict."""
    out = []
    for k, v in a["n"].items():
        if v != b["n"][k]: out.append(f"robot {k} count {v} -> {b['n'][k]}")
    for grp in ("body", "joint", "dof", "geom", "act"):
        names = a[grp].get("names") or a["joint"]["names"]
        for f, va in a[grp].items():
            if f == "names": continue
            vb = b[grp][f]; va = np.asarray(va, float); vb = np.asarray(vb, float)
            if va.shape != vb.shape:
                out.append(f"{grp}.{f} shape {va.shape} -> {vb.shape}"); continue
            if va.size == 0: continue
            rel = np.abs(va - vb)/np.maximum(1.0, np.abs(va)); dv = rel.reshape(len(va), -1).max(axis=1) if va.ndim > 1 else rel
            idx = np.where(dv > TOL)[0]
            if len(idx):
                ex = [(names[i] if grp not in ("dof",) else f"dof{i}", np.round(va[i], 5).tolist(), np.round(vb[i], 5).tolist()) for i in idx[:4]]
                out.append(f"{grp}.{f}: {len(idx)} of {len(va)} differ, e.g. {ex}")
    if a["body"]["names"] != b["body"]["names"]: out.append(f"body names renamed: {sum(x != y for x, y in zip(a['body']['names'], b['body']['names']))}")
    if a["joint"]["names"] != b["joint"]["names"]: out.append(f"joint names renamed: {sum(x != y for x, y in zip(a['joint']['names'], b['joint']['names']))}")
    sa = {(p["g1"], p["g2"]): p for p in a["pairs"]}; sb = {(p["g1"], p["g2"]): p for p in b["pairs"]}
    if len(sa) != len(sb) or any(sa.get(k) != v for k, v in sb.items()):
        ch = [k for k in sb if k in sa and sa[k] != sb[k]]
        out.append(f"robot contact pairs {len(sa)} -> {len(sb)}; changed {len(ch)}" + (f", e.g. {sa[ch[0]]} -> {sb[ch[0]]}" if ch else "")
                   + (f"; added e.g. {[sb[k] for k in sb if k not in sa][:2]}" if len(sb) > len(sa) else ""))
    if b["eff"]: out.append(f"ADDED effector colliders {[(e['name'], e['body'], 'sphere r=%.3f' % e['size'][0]) for e in b['eff']]}")
    if a["nexclude"] != b["nexclude"]: out.append(f"excludes {a['nexclude']} -> {b['nexclude']}")
    if a["eq"] != b["eq"]: out.append(f"equality constraints {a['eq']} -> {b['eq']}")
    if a["keys"] != b["keys"]: out.append(f"keyframes {a['keys']} -> {b['keys']}")
    for k, v in a["opt"].items():
        if v != b["opt"][k]: out.append(f"opt.{k} {v} -> {b['opt'][k]}")
    if abs(a["robot_mass"] - b["robot_mass"]) > 1e-6: out.append(f"robot mass {a['robot_mass']:.4f} -> {b['robot_mass']:.4f} kg")
    return out


def load(p): return mujoco.MjModel.from_xml_path(p)


def robot_of(path):
    f = os.path.basename(path)
    if "/unitree_h1/" in path: return "H1"
    if f.startswith("scene_md_G1H_") or f.startswith("scene_sheet_"): return "G1H"
    if f.startswith("scene_md_G1_") or f.startswith("scene_mech_"): return "G1"
    return None


def world_base(path):
    """scene_world_* files: the robot is identified by the base file's model name."""
    txt = open(path).read(2000)
    if "g1_29dof_with_hand" in txt: return "G1H"
    if "h1" in txt.lower() and "/unitree_h1/" in path: return "H1"
    return "G1"


def main():
    os.makedirs(OUT, exist_ok=True); res = dict(orig_vs_derived={}, scene_copies={}, runtime={}, ratings={})
    C = {r: collect(load(p)) for r, p in ORIG.items()}
    # 1. base copies vs the menagerie originals
    for r, p in DERIVED.items():
        res["orig_vs_derived"][r] = dict(orig=ORIG[r], derived=p, deviations=diff(C[r], collect(load(p))))
        print(r, res["orig_vs_derived"][r]["deviations"], flush=True)
    # 2. every task scene copy vs the menagerie original of its robot (robot part only)
    files = sorted(glob.glob(f"{MEN}/unitree_g1/scene_*.xml") + glob.glob(f"{MEN}/unitree_h1/scene_*.xml"))
    files = [f for f in files if os.path.basename(f) not in ("scene.xml", "scene_mjx.xml", "scene_with_hands.xml")]
    agg = {}
    for f in files:
        r = robot_of(f) or world_base(f)
        try: dv = diff(C[r], collect(load(f)))
        except Exception as e: dv = [f"LOAD_ERROR {type(e).__name__}: {str(e)[:120]}"]
        # deviations expected from the base copy (site / keyframes) are filtered into a signature
        sig = "; ".join(sorted(x for x in dv))
        agg.setdefault(r, {}).setdefault(sig, []).append(os.path.basename(f))
    for r, d in agg.items():
        res["scene_copies"][r] = [dict(n_files=len(v), deviations=(k.split("; ") if k else []), example=v[0]) for k, v in sorted(d.items(), key=lambda kv: -len(kv[1]))]
        print(r, [(x["n_files"], x["deviations"][:6]) for x in res["scene_copies"][r]], flush=True)
    # 3. runtime plants (throw.setup: the plant every quoted result builds on) vs the compiled scene it loads
    import throw as T, branch_i as BI
    for r in ("G1", "G1H", "H1", "Apollo", "T1"):
        try:
            S = T.setup(r, ball=False); m = S["m"]; src = T.CFG[r].get("mjcf") or BI.ROBOTS[r]["mjcf"]
            dv = diff(collect(load(src)), collect(m))
            res["runtime"][r] = dict(scene=src, deviations=dv, held_actuators=len(S["held"]), driven=len(S["types"]))
            # ratings used by the plant vs the model's own actuatorfrcrange
            rr = []
            for i, t in enumerate(S["types"]):
                jid = S["tmap"][t][0][0]; mod = load(src); fr = float(mod.jnt_actfrcrange[jid][1]) if mod.jnt_actfrclimited[jid] else None
                a = [x for x in range(mod.nu) if mod.actuator_trnid[x][0] == jid]; afr = float(mod.actuator_forcerange[a[0]][1]) if a and mod.actuator_forcelimited[a[0]] else None
                rr.append(dict(joint=t, tau_used=float(S["jarr"]["tau"][i]), omega_used=float(S["jarr"]["om"][i]), mjcf_jnt_actfrcrange=fr, mjcf_act_forcerange=afr))
            res["ratings"][r] = rr
            print(r, "runtime", dv[:8], flush=True)
        except Exception as e:
            res["runtime"][r] = dict(error=f"{type(e).__name__}: {e}"); print(r, "runtime ERROR", e, flush=True)
    json.dump(res, open(f"{OUT}/spec_diff.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", f"{OUT}/spec_diff.json")


if __name__ == "__main__":
    main()
