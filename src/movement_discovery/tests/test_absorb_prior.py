"""ABSORB prior (construction/absorb.py): registration, the momentum / stroke arithmetic, the timing law on a synthetic retreat line,
the parameter mapping (z0 / unpack / warm round trip) and the interception strategy spaces: all MuJoCo-free. One execution test (a
prior-only P73 rollout puts the palm on the ball's path) runs only with --run-execution."""
import numpy as np
import pytest
from movement_discovery import construction as C
from movement_discovery.construction import absorb as A

REP = "torque_knots+absorb_prior"


def test_registered():
    cls = C.get(REP); assert cls is A.TorqueKnotsAbsorbPrior and cls.name == REP and cls.label
    assert issubclass(cls, C.TorqueKnotsTrackedPrior) and REP in C.describe_all()


def test_stroke_plan_momentum_and_room():
    caps = dict(torque=80.0, tipping=60.0, friction=500.0)
    p = A.stroke_plan(3.0, 0.2, 1.0, caps, 1, -0.5, 0.5)
    assert p["F_tau"] == pytest.approx(40.0) and p["a_acc"] == pytest.approx(40.0)
    assert p["v1"] == pytest.approx((0.2*3.0 + 1.0*p["v_c"])/1.2)                              # inelastic contact, momentum conserved
    assert p["s"] == pytest.approx((0.2 + 1.0)*p["v1"]**2/(2*p["F_use"]))                       # s = (m_obj + L) v'^2 / (2 F_use)
    assert p["d_pre"] == pytest.approx(p["v_c"]**2/(2*p["a_acc"])) and p["d_pre"] <= 0.5 + 1e-9 and p["s"] <= 0.5 + 1e-9
    tight = A.stroke_plan(3.0, 0.2, 1.0, caps, 1, -0.05, 0.05)                                   # little room: less matching, bounded stroke
    assert tight["v_c"] < p["v_c"] and tight["s"] <= 0.05 + 1e-9 and tight["rel_impact"] > p["rel_impact"]
    none = A.stroke_plan(3.0, 0.2, 1.0, caps, 1, 0.0, 0.0)
    assert none["v_c"] == 0.0 and none["s"] == 0.0 and none["rel_impact"] == pytest.approx(3.0)
    heavy = A.stroke_plan(3.0, 5.0, 1.0, caps, 1, -0.5, 2.0)                                     # 15 N s: the support caps bind, the stroke grows
    assert heavy["s"] > p["s"] and heavy["F_use"] <= heavy["F_tau"] + 1e-9


def _geometry(n_joint=3):
    """Synthetic retreat line: base fixed, joint j = u (one joint per unit of line coordinate), u in [-0.3, 0.3]."""
    us = np.round(np.arange(-0.3, 0.3001, 0.01), 4); Q = np.zeros((len(us), 7 + n_joint)); Q[:, 3] = 1.0; Q[:, 7:] = us[:, None]
    start = np.zeros(7 + n_joint); start[3] = 1.0; start[7:] = -0.5
    sp = A.stroke_plan(2.0, 0.1, 0.5, dict(torque=60.0, tipping=40.0, friction=400.0), 1, -0.3, 0.3)
    return dict(t_a=0.45, u=us, Qline=Q, u_min=-0.3, u_max=0.3, start=start, nqb=7 + n_joint, **{k: sp[k] for k in ("v_c", "v1", "s", "d_pre", "tau")})


def test_timing_law_phases():
    g = _geometry(); r = A.timing_law(g, horizon=0.8); Q = r["Q"]; fps = r["fps"]; q = lambda t: Q[int(round(t*fps)), 7]
    assert Q.shape == (81, 10) and np.allclose(Q[:5, 7:], -0.5)                                 # stance until 0.05 s
    assert q(r["t_prep_end"]) == pytest.approx(-g["d_pre"], abs=0.02)                            # ready pose (upstream by d_pre) at t_a - tau
    assert q(g["t_a"]) == pytest.approx(0.0, abs=0.03)                                           # on the contact point at arrival
    after = Q[int(round(g["t_a"]*fps)):, 7]; assert np.all(np.diff(after) >= -1e-9)               # the stroke only retreats along the velocity
    assert Q[-1, 7] == pytest.approx(g["s"], abs=0.01) and r["t_end"] == pytest.approx(g["t_a"] + 2*g["s"]/g["v1"])
    r2 = A.timing_law(g, scale=2.0, shift=0.05, horizon=0.8)
    assert r2["t_a"] == pytest.approx(0.50) and r2["stroke"] == pytest.approx(min(2*g["s"], 0.3)) and r2["t_end"] > r["t_end"]
    assert np.allclose(np.linalg.norm(Q[:, 3:7], axis=1), 1.0)


class _NativeTask:
    name = "catch"; robot = "G1"
    def one_foot_hold(self, *a): raise AssertionError("not called in this test")


def test_param_mapping_round_trip():
    import throw as T
    rep = C.get(REP)(_NativeTask()); m = object(); S = dict(types=["a", "b", "c"], m=m, q_settled=np.zeros(10))
    g = dict(_geometry(), key="absorb_prior"); rep._refs[("absorb", id(m))] = g; rep.task._refs = {}
    n = 3*rep.nk(); demand = dict(object=dict(pos=[1, 0, 1]), omega=20.0)
    assert rep.n_params(S, demand) == n + 3
    z = rep.z0(S, demand); p = rep.unpack(S, demand, z)
    assert p["scale"] == pytest.approx(1.0) and p["shift"] == pytest.approx(0.0) and p["tscale"] == 1.0 and 20.0 <= p["omega"] <= A.OMEGA_MAX
    assert T.release_time(z[n]) == pytest.approx(min(0.8, A.timing_law(g)["t_end"] + 0.02), abs=1e-6)
    z2 = z.copy(); z2[n + 1] = 0.6; z2[n + 2] = -0.5; p2 = rep.unpack(S, demand, z2)
    assert A.SCALE[0] <= p2["scale"] <= A.SCALE[1] and p2["shift"] == pytest.approx(-0.5*A.SHIFT)
    w = rep.warm(S, demand, dict(x=p2["x"], scale=p2["scale"], shift=p2["shift"])); assert np.allclose(w, z2)
    assert rep.describe()["prior"] == "absorb"


def test_interception_strategy_spaces(registry):
    for cls in registry.programs():
        p = cls()
        if not p.id.startswith(("P73", "P74", "P75", "P76", "P77", "P78", "P79", "P80")): continue
        ss = p.strategy_space(p.sample_scenario(0)); assert ss[0] == "torque_knots+tracked_prior", (p.id, ss)
        assert (REP in ss) == (not p.id.startswith("P75")), (p.id, ss)


@pytest.mark.execution
@pytest.mark.mujoco
def test_p73_prior_reaches_the_path():
    from movement_discovery import runner as R
    from engine import core
    asm = R.assemble("P73_catch_ballistic_object", 0, "G1", REP); d = asm["demand"]; rep = asm["rep"]; S = rep.plant(d, core.DT_OPT)
    rec = rep.run(S, d, rep.z0(S, d)); assert rec["extra"]["dmin_obj"] < 0.15 and "STANCE_LOST" not in rec["reasons"]
