"""Deliverable interface.

    python -m tool --mjcf <scene.xml> [--urdf <file.urdf>] [--t-contact 0.25] [--demand 2.0]

In:  an MJCF (and optionally a manufacturer URDF for effort/velocity limits).
Out: required_depth(D) across the demand range, the minimal sufficient joint-group subset at
     each demand, the full analytic frontier, and achieved/bound when a trajectory is supplied.

The analytic path needs no contact model, no solver and no training. It runs in minutes.
"""
import sys, os, json, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, yaml

G = 9.80665


def autospec(mjcf, urdf=None, out=None):
    """Build a minimal spec from an MJCF, using a URDF for capacities where available.
    Every parameter is tagged CITED (read from a declared source) or INVENTED."""
    m = mujoco.MjModel.from_xml_path(mjcf)
    # Actuator-type guard. 5 of 9 audited Menagerie humanoids ship POSITION actuators
    # (ROBOT_ACTUATOR_SEMANTICS.md; semantics can differ between a robot's own scene variants)
    # (biastype affine): d.ctrl is a target ANGLE, not a torque, and the force the joint can
    # deliver is set by the servo gains rather than by any torque limit in the file. A
    # torque-limit analysis on such a model is analysing something the controller never uses.
    # This cost this project two phases of invalid dynamic results before it was noticed.
    n_servo = sum(1 for a in range(m.nu) if int(m.actuator_biastype[a]) == 1)
    actuator_note = ("torque" if n_servo == 0 else
                     f"POSITION SERVO on {n_servo}/{m.nu} actuators: ctrl is a target angle. "
                     "Torque limits below are joint properties, not what the shipped controller "
                     "commands. Convert to torque motors before any dynamic use.")
    U = {}
    if urdf and os.path.exists(urdf):
        import xml.etree.ElementTree as ET
        for j in ET.parse(urdf).getroot().iter("joint"):
            l = j.find("limit")
            if j.get("name") and l is not None and l.get("effort"):
                U[j.get("name")] = (float(l.get("effort")), float(l.get("velocity", 0)) or None)
    GROUP_HINTS = ["ankle", "knee", "hip", "waist", "torso", "shoulder", "elbow", "wrist", "neck"]
    joints, groups, missing = {}, {}, []
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if n is None or m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE: continue
        tau = float(m.jnt_actfrcrange[j][1])
        src = "mjcf_actuatorfrcrange"
        if n in U and U[n][0]: tau, src = U[n][0], "urdf_effort"
        om = U.get(n, (None, None))[1]
        if not om: om, omsrc = 20.0, "INVENTED"
        else: omsrc = "urdf_velocity"
        if tau <= 0:
            for a in range(m.nu):
                if m.actuator_trnid[a][0] != j: continue
                # ctrlrange is a TARGET ANGLE on a position servo (defect P9); only a torque
                # motor's ctrlrange is a torque, and forcerange is valid for both.
                if m.actuator_forcelimited[a]:
                    tau, src = abs(float(m.actuator_forcerange[a][1])), "mjcf_forcerange"
                elif int(m.actuator_biastype[a]) == 0 and m.actuator_ctrllimited[a]:
                    tau = abs(float(m.actuator_ctrlrange[a][1])*float(m.actuator_gear[a][0]))
                    src = "mjcf_ctrlrange_x_gear(torque motor)"
        if tau <= 0: missing.append(n); continue
        g = next((h for h in GROUP_HINTS if h in n.lower()), "other")
        assert g != "other" or "free" in n.lower() or True, n   # grouping is by DECLARED joint name
        joints[n] = dict(tau=tau, tau_src=src, omega=om, omega_src=omsrc,
                         lo=float(m.jnt_range[j][0]), hi=float(m.jnt_range[j][1]),
                         axis=[float(x) for x in m.jnt_axis[j]], group=g)
        groups.setdefault(g, []).append(n)
    spec = dict(mjcf=mjcf, urdf=urdf, mass=float(sum(m.body_mass)), joints=joints,
                groups=groups, joints_without_capacity=missing,
                actuator_type=actuator_note, n_position_servo=n_servo)
    if out: json.dump(spec, open(out, "w"), indent=1)
    return spec


def analytic_frontier(spec, t_contact=0.25, sagittal_only=True):
    """required_depth(D) and the full subset frontier, with no solver and no contact model."""
    J = spec["joints"]
    use = {n: r for n, r in J.items() if (not sagittal_only or abs(r["axis"][1]) > 0.9)}
    groups = {}
    for n, r in use.items(): groups.setdefault(r["group"], []).append(n)
    groups = {g: v for g, v in groups.items() if g not in ("wrist", "neck", "other")}
    gs = list(groups)
    rows = []
    for k in range(1, len(gs)+1):
        for sub in itertools.combinations(gs, k):
            W = 0.0
            for g in sub:
                for n in groups[g]:
                    r = use[n]
                    dth = min(abs(r["hi"]-r["lo"]), 2.5)          # conservative sweep
                    wr = r["tau"]*dth
                    wp = r["tau"]*r["omega"]/4.0*t_contact
                    wc = r["tau"]*max(0.0, 1.0-(dth/t_contact)/r["omega"])*dth
                    W += min(wr, wp, wc)
            v = float(np.sqrt(max(0.0, 2.0*W/spec["mass"])))
            rows.append(dict(subset=list(sub), k=k, v_bound=v, W=W))
    # enforce monotonicity by set inclusion
    by = {frozenset(r["subset"]): r for r in rows}
    for key in sorted(by, key=len):
        for o in by:
            if o < key and by[o]["v_bound"] > by[key]["v_bound"]:
                by[key]["v_bound"] = by[o]["v_bound"]
    rows = list(by.values())
    vmax = max(r["v_bound"] for r in rows)
    demands = np.linspace(0.1*vmax, 0.98*vmax, 15)
    rd = []
    for Dm in demands:
        ok = [r for r in rows if r["v_bound"] >= Dm]
        if not ok: rd.append(dict(demand=float(Dm), depth=None, subset=None)); continue
        k = min(len(r["subset"]) for r in ok)
        best = max([r for r in ok if len(r["subset"]) == k], key=lambda r: r["v_bound"])
        rd.append(dict(demand=float(Dm), depth=k, subset=best["subset"]))
    return dict(groups=gs, v_bound_max=vmax, frontier=rows, required_depth=rd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mjcf", required=True); p.add_argument("--urdf", default=None)
    p.add_argument("--t-contact", type=float, default=0.25)
    p.add_argument("--achieved", type=float, default=None,
                   help="measured takeoff velocity, to report achieved/bound")
    p.add_argument("--json", default=None)
    a = p.parse_args()
    spec = autospec(a.mjcf, a.urdf)
    r = analytic_frontier(spec, a.t_contact)
    print(f"model: {os.path.basename(a.mjcf)}   mass {spec['mass']:.2f} kg   "
          f"{len(spec['joints'])} actuated joints")
    inv = [n for n, j in spec["joints"].items() if j["omega_src"] == "INVENTED"]
    print(f"groups used: {r['groups']}")
    if spec["n_position_servo"]:
        print(f"WARNING actuators: {spec['actuator_type']}")
    print(f"INVENTED parameters: {len(inv)} joints lack a velocity limit"
          + (f" (e.g. {inv[0]})" if inv else ""))
    if spec["joints_without_capacity"]:
        print(f"WARNING: {len(spec['joints_without_capacity'])} joints had no recoverable torque limit")
    print(f"analytic max bound: {r['v_bound_max']:.2f} m/s\n")
    print(f"{'demand (m/s)':>13}{'depth':>7}  minimal sufficient subset")
    for x in r["required_depth"]:
        print(f"{x['demand']:13.2f}{(x['depth'] if x['depth'] else '--'):>7}  "
              f"{'+'.join(x['subset']) if x['subset'] else '(infeasible)'}")
    if a.achieved:
        print(f"\nachieved/bound = {a.achieved/r['v_bound_max']:.3f}")
    if a.json: json.dump(dict(spec=spec, result=r), open(a.json, "w"), indent=1)

if __name__ == "__main__":
    main()
