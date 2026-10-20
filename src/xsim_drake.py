"""Phase 10 B4: replay stored G1 solutions in DRAKE (a second simulator with a different contact
formulation: compliant point contact solved by SAP) under the IDENTICAL control law
(xsim_export.controller_spec), from the same settled start. Run in ~/venvs/drake.
Both simulators' traces are scored by the same rule (flight = floor force < 1 N with upward COM
velocity, sustained 20 ms; best confirmed flight), so differences are physics, not bookkeeping."""
import sys, os, glob, json, pickle, time
import numpy as np
from pydrake.all import (DiagramBuilder, AddMultibodyPlantSceneGraph, Parser, Simulator, ContactModel,
                         DiscreteContactApproximation, PdControllerGains, RevoluteJoint)

XS = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "results/phase10/xsim")
G = 9.80665; FLIGHT_CONFIRM = 0.020
SETTLE_KP, SETTLE_KD, SETTLE_T = 3000.0, 60.0, 4.0   # settle servo only (not part of the experiment)

def score(t, vz, fz):
    """Best confirmed flight: (v_takeoff, t_takeoff, n_flights)."""
    best = (0.0, None); n = 0; i = 0; N = len(t)
    while i < N:
        if fz[i] < 1.0 and vz[i] > 0.0 and i > 0:
            j = i
            while j < N and fz[j] < 1.0: j += 1
            if (t[min(j, N-1)] - t[i]) >= FLIGHT_CONFIRM - 1e-12 or j == N:
                n += 1
                if vz[i] > best[0]: best = (float(vz[i]), float(t[i]))
            i = j
        else:
            i += 1
    return best[0], best[1], n

def build(spec, penetration_allowance=None, approx="sap", stiction=None, settle=False, contact_k=None):
    b = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(b, time_step=spec["dt"])
    plant.set_contact_model(ContactModel.kPoint)
    plant.set_discrete_contact_approximation({"sap": DiscreteContactApproximation.kSap,
                                              "lagged": DiscreteContactApproximation.kLagged,
                                              "similar": DiscreteContactApproximation.kSimilar}[approx])
    Parser(plant).AddModels(f"{XS}/g1_drake.xml")
    held = {h["joint"]: h for h in spec["held"]}
    acts = {}
    # armature: MJCF dof armature -> actuator rotor inertia with gear 1 (reflected inertia = armature)
    arm = json.load(open(f"{XS}/armature.json"))
    for jn in spec["driven"] + list(held):
        j = plant.GetJointByName(jn)
        if jn in held: lim = held[jn]["limit"]
        elif settle: lim = float(spec["tau"][spec["driven"].index(jn)])
        else: lim = 1e6
        a = plant.AddJointActuator(jn + "_act", j, effort_limit=lim)
        a.set_default_rotor_inertia(arm.get(jn, 0.0)); a.set_default_gear_ratio(1.0)
        if jn in held: a.set_controller_gains(PdControllerGains(p=held[jn]["kp"], d=held[jn]["kd"]))
        elif settle: a.set_controller_gains(PdControllerGains(p=SETTLE_KP, d=SETTLE_KD))
        acts[jn] = a
    if penetration_allowance: plant.set_penetration_allowance(penetration_allowance)
    if stiction: plant.set_stiction_tolerance(stiction)
    plant.Finalize()
    # the MJX model collides only its explicit <pair> list; every exported geom is in a FLOOR pair,
    # so robot self-collision is excluded here to match (Drake otherwise collides non-adjacent bodies)
    from pydrake.all import CollisionFilterDeclaration, GeometrySet
    inst = plant.GetModelInstanceByName("g1_drake_export")
    gids = []
    for bi in plant.GetBodyIndices(inst):
        gids += plant.GetCollisionGeometriesForBody(plant.get_body(bi))
    sg.collision_filter_manager().Apply(CollisionFilterDeclaration().ExcludeWithin(GeometrySet(gids)))
    if contact_k is not None:
        # explicit compliant point-contact stiffness (N/m) on every robot collision geom; the floor
        # is made effectively rigid so k is the contact stiffness (series combination)
        from pydrake.all import ProximityProperties, RoleAssign, CoulombFriction
        insp = sg.model_inspector()
        for g, k in [(g, contact_k) for g in gids] + [(g, 1e9) for g in plant.GetCollisionGeometriesForBody(plant.world_body())]:
            pp = ProximityProperties(insp.GetProximityProperties(g))
            if pp.HasProperty("material", "point_contact_stiffness"): pp.UpdateProperty("material", "point_contact_stiffness", k)
            else: pp.AddProperty("material", "point_contact_stiffness", k)
            sg.AssignRole(plant.get_source_id(), g, pp, RoleAssign.kReplace)
    d = b.Build()
    return d, plant

def set_start(plant, pc, sp, q0, v0):
    from pydrake.all import RigidTransform, Quaternion, SpatialVelocity
    names = sp["qpos_joint_names"]; qa = sp["jnt_qposadr"]; da = sp["jnt_dofadr"]
    fb = plant.GetBodyByName("pelvis"); qw = q0[3:7] / np.linalg.norm(q0[3:7])
    plant.SetFreeBodyPose(pc, fb, RigidTransform(Quaternion(qw), q0[0:3]))
    plant.SetFreeBodySpatialVelocity(pc, fb, SpatialVelocity(Quaternion(qw).rotation() @ v0[3:6], v0[0:3]))
    for k, jn in enumerate(names):
        if k == 0: continue
        j = plant.GetJointByName(jn); j.set_angle(pc, float(q0[qa[k]])); j.set_angular_rate(pc, float(v0[da[k]]))

def settle_drake(sp, **kw):
    """Same protocol as branch_i.settle, in Drake: every joint servo-held at the start pose for
    SETTLE_T s, then the driven joints' realised holding torques -> per-type mean -> lag state."""
    diagram, plant = build(sp, settle=True, **kw)
    sim = Simulator(diagram); ctx = sim.get_mutable_context(); pc = plant.GetMyMutableContextFromRoot(ctx)
    set_start(plant, pc, sp, sp["q_settled"], 0*sp["v_settled"])
    order = [plant.get_joint_actuator(i).joint().name() for i in plant.GetJointActuatorIndices()]
    names = sp["qpos_joint_names"]; qa = sp["jnt_qposadr"]
    tgt = {jn: float(sp["q_settled"][qa[names.index(jn)]]) for jn in order}
    plant.get_desired_state_input_port(plant.GetModelInstanceByName("g1_drake_export")).FixValue(
        pc, np.concatenate([[tgt[n] for n in order], np.zeros(len(order))]))
    plant.get_actuation_input_port().FixValue(pc, np.zeros(plant.num_actuators()))
    sim.AdvanceTo(SETTLE_T)
    F = plant.get_net_actuation_output_port().Eval(pc)
    Fd = np.array([F[order.index(jn)] for jn in sp["driven"]])
    # per-JOINT holding torques. MuJoCo's protocol averages L/R per type, which is exact there
    # because its stiff settle is symmetric; Drake's settle is not perfectly symmetric (right
    # shoulder 4.4 vs left -0.7 N m), and the L/R mean then tips the released robot (hold test:
    # COM vz -0.15 m/s at 0.5 s averaged vs -0.06 per-joint). Recorded as a protocol deviation.
    q = plant.GetPositions(pc); v = plant.GetVelocities(pc)
    return dict(q=q, v=v, act0=Fd/sp["tau"], vmax=float(np.abs(v).max()))

def run(pkl, **kw):
    P = pickle.load(open(pkl, "rb")); sp = P["spec"]
    st = settle_drake(sp, **kw)
    diagram, plant = build(sp, **kw)
    sim = Simulator(diagram); ctx = sim.get_mutable_context(); pc = plant.GetMyMutableContextFromRoot(ctx)
    # start state: Drake's own settled rest (same plant layout, so positions/velocities map 1:1)
    plant.SetPositions(pc, st["q"]); plant.SetVelocities(pc, st["v"])
    # actuator ordering in Drake's actuation vector
    order = [plant.get_joint_actuator(i).joint().name() for i in plant.GetJointActuatorIndices()]
    drv = [order.index(jn) for jn in sp["driven"]]
    pd_names = [n for n in order if n in {h["joint"] for h in sp["held"]}]
    htarget = {h["joint"]: h["target"] for h in sp["held"]}
    # desired-state port covers every actuator in the instance (ignored for non-PD actuators)
    xd = np.concatenate([[htarget.get(n, 0.0) for n in order], np.zeros(len(order))])
    plant.get_desired_state_input_port(plant.GetModelInstanceByName("g1_drake_export")).FixValue(pc, xd)
    jd = [plant.GetJointByName(jn) for jn in sp["driven"]]
    kn = sp["kn"]; NK = kn.shape[1]; ti = sp["ti"]; sgn = sp["sgn"]; tau = sp["tau"]; om = sp["om"]
    act = st["act0"].copy(); alpha = sp["alpha"]; dt = sp["dt"]; kd_ = sp["knot_dt"]
    nst = int(sp["horizon"]/dt)
    M = plant.CalcTotalMass(pc)
    t_, vz_, fz_, q_ = [], [], [], []
    u = np.zeros(plant.num_actuators()); port = plant.get_actuation_input_port()
    vprev = plant.CalcCenterOfMassTranslationalVelocityInWorld(pc)[2]
    for s in range(nst):
        tt = s*dt; k = min(int(tt/kd_ + 1e-9), NK-1); k2 = min(k+1, NK-1)
        f = min(1.0, (tt - k*kd_)/kd_) if k < NK-1 else 0.0
        cmd = (kn[:, k] + (kn[:, k2]-kn[:, k])*f)[ti]*sgn
        q = np.array([j.get_angle(pc) for j in jd]); w = np.array([j.get_angular_rate(pc) for j in jd])
        i = min(int(round(tt/dt)), len(sp["QR"]) - 1)
        cmd = cmd + (sp["tr_kp"]*(sp["QR"][i] - q) + sp["tr_kd"]*(sp["QDR"][i] - w))/tau
        act += alpha*(cmd - act)
        frac = np.maximum(0.0, 1.0 - np.abs(w)/om)
        hi = np.where(w > 0, tau*frac, tau); lo = np.where(w < 0, tau*frac, tau)
        tq = np.clip(act*tau, -lo, hi)
        u[:] = 0.0; u[drv] = tq; port.FixValue(pc, u)
        sim.AdvanceTo((s+1)*dt)
        vz = plant.CalcCenterOfMassTranslationalVelocityInWorld(pc)[2]
        # floor normal force from momentum balance (gravity + contact are the only external forces)
        fz = M*((vz - vprev)/dt + G); vprev = vz
        cr = plant.get_contact_results_output_port().Eval(pc)
        ncon = cr.num_point_pair_contacts()
        t_.append(s*dt); vz_.append(vz); fz_.append(fz if ncon > 0 else 0.0); q_.append(q)
    t_, vz_, fz_ = map(np.array, (t_, vz_, fz_))
    v, tto, nf = score(t_, vz_, fz_)
    mj = P["mj"]["trace"]; vm, tm, nm_ = score(np.array(mj["t"]), np.array(mj["vz"]), np.array(mj["fz"]))
    pre = t_ <= (tto if tto is not None else t_[-1])
    return dict(key=os.path.basename(pkl)[:-4], **P["meta"], v_mujoco_rescored=vm, t_mujoco=tm,
                settle_vmax=st["vmax"], act0_drake=st["act0"].tolist(), act0_mujoco=sp["act0"].tolist(),
                v_drake=v, t_drake=tto, n_flights_drake=nf,
                peak_grf_bw_drake=float(fz_[pre].max()/(M*G)), M_drake=M,
                q_rms_dev=float(np.sqrt(np.mean((np.array(q_) - P["mj"]["q"][:len(q_)])**2))) if len(P["mj"]["q"]) else None,
                trace=dict(t=t_.tolist(), vz=vz_.tolist(), fz=fz_.tolist()))

def sweep_main():
    """All stored solutions x Drake contact stiffness (penetration allowance)."""
    from multiprocessing import Pool
    files = sorted(glob.glob(f"{XS}/*.pkl"))
    jobs = [(f, k) for f in files for k in (None, 1e4, 1e5, 1e6)]
    with Pool(12) as p: res = p.starmap(_one, jobs)
    json.dump(res, open(f"{XS}/drake_results.json", "w"))
    for r in res:
        print(f'{r["key"]:38s} k {str(r["contact_k"]):8s} mujoco {r["v_mujoco"]:.3f} drake {r["v_drake"]:.3f} '
              f'grf drake {r["peak_grf_bw_drake"]:.1f} mj {r["grf_bw"]:.1f}', flush=True)

def _one(f, k):
    r = run(f, contact_k=k); r["contact_k"] = k; return r


if __name__ == "__main__" and sys.argv[1:2] == ["sweep"]:
    sweep_main()
elif __name__ == "__main__":
    files = sorted(glob.glob(f"{XS}/*.pkl")) if len(sys.argv) < 2 else sys.argv[1:]
    out = []
    for fpath in files:
        t0 = time.time(); r = run(fpath)
        print(f'{r["key"]:40s} mujoco {r["v_mujoco"]:.3f} (rescored {r["v_mujoco_rescored"]:.3f}) drake {r["v_drake"]:.3f} '
              f'grf_bw drake {r["peak_grf_bw_drake"]:.1f} mj {r["grf_bw"]:.1f} qdev {r["q_rms_dev"]:.3f} {time.time()-t0:.0f}s', flush=True)
        out.append(r)
    if len(sys.argv) < 2:
        json.dump(out, open(f"{XS}/drake_results.json", "w"))
