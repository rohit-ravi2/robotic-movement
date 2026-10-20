"""Phase 10 (defects P15, P16): the planar human's joint conventions are DECLARED (human_planar.QSIGN)
and every angle-dependent capacity is evaluated in the anatomical convention. Physical checks, not
string checks. Also: the Anderson 2007 surfaces reproduce their own defining points."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, mujoco, human_planar as HP
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spec", "body_sourced.yaml")

def _xrel(h, joint, spec_angle, body, ref):
    q0 = h.place(None); d = mujoco.MjData(h.m); d.qpos[:] = q0
    i = HP.JOINTS.index(joint); d.qpos[h.qadr[i]] = HP.QSIGN[joint]*spec_angle; mujoco.mj_forward(h.m, d)
    b = mujoco.mj_name2id(h.m, mujoco.mjtObj.mjOBJ_BODY, body); r = mujoco.mj_name2id(h.m, mujoco.mjtObj.mjOBJ_BODY, ref)
    return d.xipos[b] - d.xpos[r]

def test_flexion_directions():
    h = HP.Human()
    assert _xrel(h, "hip", 1.0, "shank", "trunk")[0] > 0.2, "hip flexion must swing the thigh FORWARD (+x)"
    assert _xrel(h, "knee", 1.0, "foot", "thigh")[0] < -0.2, "knee flexion must swing the shank BACKWARD"
    assert _xrel(h, "elbow", 1.0, "farm", "uarm")[0] > 0.1, "elbow flexion must swing the forearm FORWARD (P16)"
    assert _xrel(h, "ankle", -0.5, "foot", "foot")[2] > _xrel(h, "ankle", 0.5, "foot", "foot")[2], "negative spec ankle = dorsiflexion"

def test_torque_angle_peak_in_anatomical_convention():
    h = HP.Human()
    for i, n in enumerate(HP.JOINTS):
        th = np.zeros(5); th[i] = HP.QSIGN[n]*h.ta[i][0]          # qpos at the spec peak angle
        assert abs(h.f_angle(th)[i] - 1.0) < 1e-9, f"{n}: f_angle must peak at the spec peak (P15)"

def test_anderson_defining_points():
    h = HP.Human(spec=SRC)
    for i in h.AJ:
        P = h.AP[i]
        for key, sgn in (("ext", 1), ("flex", -1)):
            p = P[key]; thA = p["C3"]
            th = np.zeros(5); th[i] = thA/(P["sA"]*h.qsign[i])
            for wa, frac in ((0.0, 1.0), (p["C4"], 0.75), (p["C5"], 0.50)):
                w = np.zeros(5); w[i] = sgn*P["e"]*wa              # concentric for this action
                lo, hi = h.limits(th, w)
                T = (hi if sgn*P["e"] > 0 else lo)[i]
                assert abs(T/p["T0"] - frac) < 1e-6, (HP.JOINTS[i], key, wa, T/p["T0"], frac)

if __name__ == "__main__":
    test_flexion_directions(); print("PASS flexion directions")
    test_torque_angle_peak_in_anatomical_convention(); print("PASS torque-angle peaks (P15)")
    test_anderson_defining_points(); print("PASS Anderson 2007 defining points")

def test_vectorised_matches_scalar():
    h = HP.Human(spec=SRC); rng = np.random.default_rng(3)
    for _ in range(200):
        th = rng.uniform(-1.5, 2.0, 5); w = rng.uniform(-12, 12, 5); lo, hi = h.limits(th, w)
        for i in h.AJ:
            P = h.AP[i]; thA = P["sA"]*h.qsign[i]*th[i]; wE = P["e"]*w[i]
            Te, Tf = h._ander(P["ext"], thA, wE), h._ander(P["flex"], thA, -wE)
            assert np.isclose(hi[i], Te if P["e"] > 0 else Tf) and np.isclose(lo[i], Tf if P["e"] > 0 else Te)

if __name__ == "__main__":
    test_vectorised_matches_scalar(); print("PASS vectorised Anderson == scalar")
