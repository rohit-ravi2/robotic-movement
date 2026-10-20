"""T1 stock-model audit (2026-10-17): compare the head-carry plant (as built by engine.headcarry.plant, at 1 ms and 0.5 ms) with
the UNMODIFIED MuJoCo Menagerie G1-with-hands model (unitree_g1/g1_with_hands.xml, git-clean), field by field, by name.
Physical parameters must match exactly; every difference is listed with its class (control / simulation / added object)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco
from engine import headcarry as HC

MEN = os.path.join(os.environ.get("MENAGERIE_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "models", "mujoco_menagerie")), "unitree_g1")


def names(m, t, n): return [mujoco.mj_id2name(m, t, i) for i in range(n)]


def compare(ms, mp):
    O = mujoco.mjtObj; out = {}
    def by(t, n_s, n_p, fields):
        ns, np_ = names(ms, t, n_s), names(mp, t, n_p); diffs = {}
        for i, nm in enumerate(ns):
            if nm is None: continue
            j = np_.index(nm) if nm in np_ else None
            if j is None: diffs[nm] = "MISSING in plant"; continue
            for f, (fs, fp) in fields.items():
                a, b = np.asarray(fs(ms, i), float), np.asarray(fp(mp, j), float)
                if a.shape != b.shape or not np.allclose(a, b, rtol=1e-9, atol=1e-12): diffs.setdefault(nm, {})[f] = (a.tolist(), b.tolist())
        return dict(n=n_s, n_differing=len(diffs), diffs=diffs)
    g = lambda attr: (lambda m, i: getattr(m, attr)[i])
    out["bodies"] = by(O.mjOBJ_BODY, ms.nbody, mp.nbody, {k: (g(k), g(k)) for k in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat")})
    out["joints"] = by(O.mjOBJ_JOINT, ms.njnt, mp.njnt, {k: (g(k), g(k)) for k in ("jnt_range", "jnt_limited", "jnt_actfrcrange", "jnt_stiffness", "jnt_axis", "jnt_pos")})
    dof = lambda attr: (lambda m, j: getattr(m, attr)[m.jnt_dofadr[j]])
    out["dofs"] = by(O.mjOBJ_JOINT, ms.njnt, mp.njnt, {k: (dof(k), dof(k)) for k in ("dof_armature", "dof_damping", "dof_frictionloss")})
    # robot geoms only (the plant adds the floor material and the load; the stock file has no floor)
    out["geoms"] = by(O.mjOBJ_GEOM, ms.ngeom, mp.ngeom, {k: (g(k), g(k)) for k in ("geom_size", "geom_friction", "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity", "geom_condim", "geom_margin", "geom_type", "geom_pos")})
    # unnamed geoms: compare in order within each body
    ug = {}
    for b in range(ms.nbody):
        nm = mujoco.mj_id2name(ms, O.mjOBJ_BODY, b); bp = mujoco.mj_name2id(mp, O.mjOBJ_BODY, nm)
        gs = [x for x in range(ms.ngeom) if ms.geom_bodyid[x] == b]; gp = [x for x in range(mp.ngeom) if mp.geom_bodyid[x] == bp]
        if len(gs) != len(gp): ug[nm] = f"geom count {len(gs)} vs {len(gp)}"; continue
        for a, c in zip(gs, gp):
            for k in ("geom_size", "geom_friction", "geom_solref", "geom_solimp", "geom_condim", "geom_type", "geom_pos", "geom_contype", "geom_conaffinity"):
                if not np.allclose(getattr(ms, k)[a], getattr(mp, k)[c]): ug.setdefault(nm, []).append(k)
    out["geoms_by_body_order"] = dict(n_bodies_differing=len(ug), diffs=ug)
    # actuators: the stock model has position/motor actuators; the plant converts them (control). Torque limit must match.
    act = {}
    for a in range(ms.nu):
        jn = mujoco.mj_id2name(ms, O.mjOBJ_JOINT, int(ms.actuator_trnid[a][0])); jp = mujoco.mj_name2id(mp, O.mjOBJ_JOINT, jn)
        ap = [k for k in range(mp.nu) if int(mp.actuator_trnid[k][0]) == jp][0]
        lim_s = float(ms.actuator_forcerange[a][1]) if ms.actuator_forcelimited[a] else (float(ms.jnt_actfrcrange[ms.actuator_trnid[a][0]][1]) if ms.jnt_actfrclimited[ms.actuator_trnid[a][0]] else None)
        lim_p = float(mp.actuator_forcerange[ap][1]); gear_s, gear_p = float(ms.actuator_gear[a][0]), float(mp.actuator_gear[ap][0])
        act[jn] = dict(stock_limit=lim_s, plant_limit=lim_p, gear=(gear_s, gear_p), match=(lim_s is not None and abs(lim_s - lim_p) < 1e-9 and gear_s == gear_p))
    out["actuator_torque_limits"] = dict(n=len(act), n_mismatch=sum(not v["match"] for v in act.values()), mismatches={k: v for k, v in act.items() if not v["match"]})
    out["opt"] = {k: (float(getattr(ms.opt, k)) if np.ndim(getattr(ms.opt, k)) == 0 else np.asarray(getattr(ms.opt, k)).tolist(), float(getattr(mp.opt, k)) if np.ndim(getattr(mp.opt, k)) == 0 else np.asarray(getattr(mp.opt, k)).tolist()) for k in ("timestep", "integrator", "cone", "impratio", "iterations", "noslip_iterations", "gravity", "o_solref")}
    return out


if __name__ == "__main__":
    ms = mujoco.MjModel.from_xml_path(f"{MEN}/g1_with_hands.xml"); res = {}
    for dt in (0.001, 0.0005):
        S, path = HC.plant("box", 32.0, dt); mp = S["m"]; res[f"dt{dt}"] = dict(scene=path, **compare(ms, mp))
    rm = sum(ms.body_mass[1:]); res["stock_robot_mass_kg"] = float(rm)
    res["note"] = ("stock = unmodified menagerie g1_with_hands.xml (menagerie git clean for that file). The plant file g1_hands_palm.xml = stock + one "
                   "massless site 'right_palm'; scene adds floor, keyframes (initial conditions), and the free load body. Actuator gain/bias are control "
                   "(position servos converted to torque motors, then the controller's own servo loops); torque limits are compared above.")
    json.dump(res, open(os.path.join(HC.RES, "..", "T1", "stock_check.json"), "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    for k in ("dt0.001", "dt0.0005"):
        r = res[k]; print(k, {s: r[s].get("n_differing", r[s].get("n_bodies_differing", r[s].get("n_mismatch"))) for s in ("bodies", "joints", "dofs", "geoms", "geoms_by_body_order", "actuator_torque_limits")}, r["opt"])
        for s in ("bodies", "joints", "dofs", "geoms"):
            for nm, v in list(r[s]["diffs"].items())[:6]: print("  ", s, nm, str(v)[:200])
        print("  geom order diffs", str(r["geoms_by_body_order"]["diffs"])[:400]); print("  act", str(r["actuator_torque_limits"]["mismatches"])[:600])
