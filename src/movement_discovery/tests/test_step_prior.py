"""Step-as-an-effect prior (construction/step.py, representation servo_knots+step_prior): registration, goal -> waypoints for every
goal kind of the stepping regime, the LIPM COM reference, SE(2) helpers, the strategy-space appends (MuJoCo-free), and ONE prior-only
step rollout (execution marker)."""
import numpy as np
import pytest
from movement_discovery import construction as C
from movement_discovery.construction import step as ST
from movement_discovery.core.contracts import GoalSpec

NAME = "servo_knots+step_prior"
REGIME = ("P01", "P02", "P03", "P05", "P06", "P07", "P08", "P20", "P70", "P81", "P82", "P83", "P84", "P87", "P88")


def test_registered():
    assert NAME in C.REPRESENTATIONS and C.get(NAME) is ST.StepPrior and C.get(NAME).label


def test_se2_roundtrip():
    a = np.array([0.3, -0.2, 0.7]); b = np.array([0.25, 0.1, -0.3])
    assert np.allclose(ST._relative(a, ST._compose(a, b)), b)


def test_goal_waypoints_per_kind():
    st = np.array([0.0, 0.0, 0.0])
    w = ST.goal_waypoints(dict(goal=GoalSpec(kind="com_displacement", params=dict(target_xy=[1.0, -2.0]))), st, 7.0)
    assert w["spec"]["kind"] == "point" and np.allclose(w["waypoints"][-1][:2], [1.0, -2.0]) and w["waypoints"][-1][2] is None
    w = ST.goal_waypoints(dict(goal=GoalSpec(kind="com_displacement", params=dict(x_min=1.2))), st, 6.0)
    assert w["spec"]["kind"] == "x_min" and w["waypoints"][-1][0] > 1.2
    w = ST.goal_waypoints(dict(goal=GoalSpec(kind="stability", params=dict(x_min=0.8))), st, 3.0)
    assert w["spec"]["kind"] == "x_min"
    w = ST.goal_waypoints(dict(goal=GoalSpec(kind="com_velocity", params=dict(v_xy=[0.5, 0.0]))), st, 4.0)
    assert w["speed"] == pytest.approx(0.5) and w["waypoints"][-1][0] == pytest.approx(2.0)
    w = ST.goal_waypoints(dict(goal=GoalSpec(kind="heading", params=dict(dpsi=0.8, speed=0.5))), st, 4.0)
    assert w["spec"]["kind"] == "heading" and w["waypoints"][-1][2] == pytest.approx(0.8)
    w = ST.goal_waypoints(dict(goal=GoalSpec(kind="path", params=dict(shape="arc", R=1.0, arc=np.pi/2, side=1, speed=0.4))), st, 8.0)
    end = w["waypoints"][-1]; assert end[0] == pytest.approx(1.0, abs=1e-6) and end[1] == pytest.approx(1.0, abs=1e-6) and end[2] == pytest.approx(np.pi/2)
    w = ST.goal_waypoints(dict(path=[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]), st, 5.0)
    assert len(w["waypoints"]) == 2 and w["spec"]["kind"] == "point"
    assert ST.goal_waypoints(dict(goal=GoalSpec(kind="stability", params=dict(disturbance="support_loss"))), st, 3.0)["waypoints"] == []


def test_lipm_com_rest_and_tracking():
    N = 400; dt = 0.01; p = np.zeros((N, 2)); p[100:300, 1] = 0.12          # ZMP over a stance foot 12 cm to the side, then back
    x = ST.lipm_com(p, dt, np.sqrt(9.81/0.67), np.zeros(2), np.zeros(2))
    assert np.allclose(x[0], 0, atol=1e-4) and np.allclose(x[-1], 0, atol=1e-3)
    v = np.gradient(x, dt, axis=0); assert np.abs(v[[0, -1]]).max() < 0.05
    assert 0.05 < x[200, 1] < 0.12                                            # the COM sways toward the stance foot, never past it


def test_stepping_regime_lists_step_prior(registry):
    for cls in registry.programs():
        p = cls(); pre = p.id.split("_")[0]
        if pre not in REGIME: continue
        ss = p.strategy_space(p.sample_scenario(0)); assert NAME in ss and ss[0] != NAME, (p.id, ss)


@pytest.mark.execution
@pytest.mark.mujoco
def test_one_step_prior_lands_and_holds():
    from movement_discovery.construction.base_effect import BaseEffectTask
    from engine import core
    t = BaseEffectTask("G1"); rep = C.get(NAME)(t)
    dem = dict(t.demand_default(), effector="pelvis", support="both", prior=None, footsteps=[dict(foot="right", dx=0.2)], horizon=[1.0, 4.0])
    S = rep.plant(dem, core.DT_OPT); r = rep.run(S, dem, rep.z0(S, dem))
    e = r["extra"]; assert r["valid"], r["reasons"]
    assert e["steps_landed"] == 1 and e["placement_err"][0] < 0.05 and e["v_end"] < 0.15 and not e["fell"]
