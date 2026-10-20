"""Representation registry and serialization (duck-typed task; no MuJoCo), strategy-space names, horizon binding."""
import pytest
from movement_discovery import construction as C


class _DummyTask:
    """engine.task.Task-shaped stand-in (non-native: no one_foot_hold), so delegation paths run without a plant."""
    name = "dummy"; robot = "G1"
    def demand_default(self): return dict(prior=True)
    def n_params(self, S, d): return 5
    def unpack(self, S, d, z): return dict(x=list(z[:4]), tscale=0.87 + 0.3*float(z[4]))
    def floor(self, d): return None
    def sigma0(self, d, w): return 0.2


def test_registry_names():
    assert {"torque_knots+tracked_prior", "torque_knots", "servo_knots", "contact_schedule+knots"} <= set(C.REPRESENTATIONS)   # prior families (brace, absorb, rise, step) register beside the core four
    for n in C.REPRESENTATIONS: assert C.get(n).name == n and C.get(n).label
    with pytest.raises(KeyError): C.get("hand_authored_phase_sequence")
    da = C.describe_all(); assert set(da) == set(C.REPRESENTATIONS) and all("label" in v and "doc" in v for v in da.values())


def test_strategy_space_names_registered(registry):
    for cls in registry.programs():
        p = cls(); ss = p.strategy_space(p.sample_scenario(0)); assert ss and all(s in C.REPRESENTATIONS for s in ss), (p.id, ss)


def test_describe_and_z0_delegated():
    import numpy as np
    rep = C.get("torque_knots+tracked_prior")(_DummyTask()); d = rep.describe()
    assert d["representation"] == "torque_knots+tracked_prior" and d["task"] == "dummy" and d["robot"] == "G1" and d["native"] is False and d["params"] == "delegated to the task"
    z = rep.z0(None, dict(prior=True)); assert len(z) == 5 and z[-1] == pytest.approx((1.0 - 0.87)/0.3) and np.all(z[:4] == 0)
    assert rep.n_params(None, dict(prior=True)) == 5 and C.get("torque_knots")(_DummyTask()).describe()["native"] is False


def test_horizon_binding_restores_module():
    import throw as T
    orig = T.rollout; rep = C.get("torque_knots")(_DummyTask(), horizon=1.6)
    assert rep.nk() == max(2, int(round(T.NK*1.6/T.HOR)))
    with C.bind_horizon(1.6): assert T.rollout is not orig and T.rollout.keywords["horizon"] == 1.6
    assert T.rollout is orig
    with C.bind_horizon(None): assert T.rollout is orig


def test_score_is_lexicographic():
    rep = C.get("torque_knots")(_DummyTask())
    bad = rep.score(dict(violation=0.5, valid=False, value=10.0), {}); assert bad >= 100.0
    class _Obj(_DummyTask):
        def objective(self, rec, d): return -rec["value"]
    assert C.get("torque_knots")(_Obj()).score(dict(violation=0.0, valid=True, value=3.0), {}) == -3.0 < bad
