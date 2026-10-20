"""S1-S10: control cases for every plant bug class found in this project. Run: python tests/test_plant.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, mujoco
import plant as P, branch_i as BI, contract_g1 as CG
from validity import TrajectoryValidityReport as TVR

MEN = os.environ.get("MENAGERIE_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "models", "mujoco_menagerie"))
RAW = f"{MEN}/unitree_g1/scene_mjx.xml"
S = BI.setup(); BI.settle(S); m = S["m"]


def test_S1_position_servo_masquerading_as_torque():
    mr = mujoco.MjModel.from_xml_path(RAW); dr = mujoco.MjData(mr)
    v = P.PlantContractVerifier(mr, dr, P.PlantContract(RAW, "torque"))
    assert not v.pctv1_actuator_semantics()


def test_S2_unactuated_locked_joints():
    # a model whose declared-held joints are left as torque motors at ctrl=0 must fail
    import copy
    rb = BI.R.RobotBody(BI.MJCF, BI.SPEC, capacity="peak"); mm = BI.to_torque_motors(rb)
    c = CG.contract(S)
    v = P.PlantContractVerifier(mm, mujoco.MjData(mm), c)
    assert not v.pctv1_actuator_semantics()


def test_S3_elevated_body_and_S4_penetration():
    d = mujoco.MjData(m); c = CG.contract(S)
    q = S["q_settled"].copy(); q[2] += 0.05
    assert not P.PlantContractVerifier(m, d, c).pctv3_ground_placement(q)
    q = S["q_settled"].copy(); q[2] -= 0.01
    assert not P.PlantContractVerifier(m, d, c).pctv3_ground_placement(q)
    assert P.PlantContractVerifier(m, d, c).pctv3_ground_placement(S["q_settled"])


def test_S5_unexplained_energy():
    r = TVR(start_qdot_max=0.0, energy_residual=5.0, energy_tol=1.0, actuator_work=50.0,
            contact_energy=0.1, ke_takeoff=30.0, nonfoot_contact=False, took_off=True).evaluate()
    assert "ENERGY_ACCOUNTING_VIOLATION" in r.reasons
    v = P.PlantContractVerifier(m, mujoco.MjData(m), CG.contract(S))
    assert not v.pctv8_bound_guard(7.43, 7.16)          # the Phase 7 failure, caught
    r = TVR(start_qdot_max=13.2, energy_residual=-1.0, energy_tol=1.0, actuator_work=50.0,
            contact_energy=0.1, ke_takeoff=30.0, nonfoot_contact=False, took_off=True).evaluate()
    assert "START_NOT_AT_REST" in r.reasons            # the Phase 7 unsettled start, caught


def test_S5b_energy_hidden_in_dissipation():
    # AD3: a 20 J injection hidden under 60 J of dissipation passes the one-sided ENERGY gate
    # (residual -40 <= tol) but must fail the two-sided closure gate.
    r = TVR(start_qdot_max=0.0, energy_residual=-40.0, energy_tol=2.0, actuator_work=100.0,
            contact_energy=0.1, ke_takeoff=30.0, nonfoot_contact=False, took_off=True,
            energy_closure=-40.0 - (-60.0), closure_tol=max(5.0, 0.02*(100.0 + 60.0))).evaluate()
    assert "ENERGY_ACCOUNTING_VIOLATION" not in r.reasons and "ENERGY_CLOSURE" in r.reasons
    # and a real corrected-plant rollout closes
    o = BI.rollout(np.zeros(len(S["types"])*BI.NK), set(S["types"]), S=S)
    assert abs(o["energy_audit"]["closure"]) <= o["validity"]["energy_tol"], o["energy_audit"]


def test_S6_true_torque_motor():
    v = P.PlantContractVerifier(m, mujoco.MjData(m), CG.contract(S))
    assert v.pctv1_actuator_semantics() and v.pctv6_known_torque(S["driven"][0])


def test_S6b_scaled_capacity_realised():
    v = P.PlantContractVerifier(m, mujoco.MjData(m), CG.contract(S))
    t = S["types"][0]; aid = S["tmap"][t][0][6]; tau = S["tmap"][t][0][3]
    BI._set_capacity_limits(m, S, {t: (2.2, 2.2)})
    assert v.pctv9_capacity_realised({aid: 2.2*tau})
    BI._set_capacity_limits(m, S, None)                  # stale-limit case must fail
    assert not v.pctv9_capacity_realised({aid: 2.2*tau})
    # joint-level clamp layer alone must also fail (the case PCTV-6 caught in production)
    BI._set_capacity_limits(m, S, {t: (2.2, 2.2)}); jid = int(m.actuator_trnid[aid][0])
    m.jnt_actfrcrange[jid][:] = [-tau, tau]
    assert not v.pctv9_capacity_realised({aid: 2.2*tau})
    BI._set_capacity_limits(m, S, None)


def test_S7_S8_certificate_feasible_and_torque_infeasible():
    d = mujoco.MjData(m); cg = P.PlantContractVerifier(m, d, CG.contract(S)).collision_geoms()
    c = P.static_certificate(m, d, S["q_settled"], coll_geoms=cg)
    assert c["status"] == "STATIC_FEASIBLE" and abs(c["total_normal_force"]-c["weight"]) < 0.01*c["weight"]
    c = P.static_certificate(m, d, S["q_settled"], coll_geoms=cg, tau_scale=np.full(m.nu, 0.001))
    assert c["status"] == "TORQUE_INFEASIBLE"


def test_S8b_servo_limits_read_from_forcerange():
    # held joints are position servos; their ctrlrange is an ANGLE. The certificate must use
    # forcerange: at the settled start every realised torque limit must be N*m-scale (>= 5)
    d = mujoco.MjData(m); cg = P.PlantContractVerifier(m, d, CG.contract(S)).collision_geoms()
    c = P.static_certificate(m, d, S["q_settled"], coll_geoms=cg)
    assert c["status"] == "STATIC_FEASIBLE"
    for (a, *_r) in S["held"]:
        assert m.actuator_forcerange[a][1] >= 5.0


def test_S9b_tilted_capsule_counts_only_grounded_end():
    # a capsule touching the ground at ONE end must contribute only that end (defect P11)
    xml = """<mujoco><worldbody><geom name='floor' type='plane' size='1 1 .1'/>
      <body pos='0 0 0.02'><freejoint/><geom type='capsule' fromto='-0.05 0 0  0.05 0 0.01' size='0.01' mass='1'/></body>
      </worldbody></mujoco>"""
    mm = mujoco.MjModel.from_xml_string(xml); dd = mujoco.MjData(mm)
    dd.qpos[2] = 0.0; mujoco.mj_forward(mm, dd)
    lo = P.geom_low(mm, dd, 1); dd.qpos[2] -= lo; mujoco.mj_forward(mm, dd)
    pts = P.potential_contacts(mm, dd, [1], tol=0.003)
    assert len(pts) == 1 and pts[0][0] < 0, pts


def test_S9c_contact_points_attach_to_generating_body():
    # a foot point directly under a heavier body's geom must still load the FOOT (defect P14):
    # a crouched knee needs knee torque in any valid static certificate
    d = mujoco.MjData(m); cg = P.PlantContractVerifier(m, d, CG.contract(S)).collision_geoms()
    c = P.static_certificate(m, d, S["q_settled"], coll_geoms=cg)
    kn = [a for a in range(m.nu) if "knee" in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0]))]
    assert max(abs(c["tau"][a]) for a in kn) > 1.0, [c["tau"][a] for a in kn]


def test_S9_friction_infeasible():
    d = mujoco.MjData(m); cg = P.PlantContractVerifier(m, d, CG.contract(S)).collision_geoms()
    g0 = m.opt.gravity.copy()
    m.opt.gravity[:] = [-1.0, 0.0, -9.755]           # ~6 degree tilt, COM stays inside support
    try:
        c = P.static_certificate(m, d, S["q_settled"], mu=0.05, coll_geoms=cg)
        assert c["status"] == "FRICTION_INFEASIBLE", c["status"]
    finally:
        m.opt.gravity[:] = g0


def test_S10_certified_pose_dynamically_held():
    v, c = CG.verify(S)
    assert v.results["PCTV-5"]["passed"], v.results["PCTV-5"]


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); n += 1; print("PASS", k)
    print(f"{n} plant tests passed")
