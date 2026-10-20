"""ValidityReport semantics: reason -> flag mapping, family policies, phase prefixes."""
from movement_discovery.physics import validity as V


def test_reason_mapping():
    r = V.report(dict(reasons=["OVERSPEED", "STANCE_LOST", "OUT_OF_CONE", "ENERGY_CLOSURE", "NOT_HIT"], violation=2.0, extra={}), "stationary")
    assert not r.ok and r.violation == 2.0 and not r.actuator_speed and not r.envelope and not r.support and not r.direction and not r.numerical and not r.object_interaction
    assert r.joint_limits and r.contact


def test_policies_allow_expected_facts():
    rec = dict(reasons=["STANCE_LOST"], violation=1.0, extra={})
    assert not V.report(rec, "stationary").ok and V.report(rec, "aerial").ok and V.report(rec, "locomotion").ok
    assert V.report(rec, "aerial").violation == 0.0            # an allowed reason carries no violation
    rec = dict(reasons=["NONFOOT_CONTACT"], violation=0.5, extra={}); assert V.report(rec, "body_contact").ok and V.report(rec, "posture").ok and not V.report(rec, "quasi_static").ok


def test_phase_prefix_stripped():
    assert V.strip_phase("P2:STANCE_LOST") == "STANCE_LOST" and V.strip_phase("STANCE_LOST") == "STANCE_LOST"
    r = V.report(dict(reasons=["P1:STANCE_LOST", "P2:FELL"], violation=1.0, extra={}), "locomotion"); assert r.reasons == ["P2:FELL"] and not r.support


def test_require_hit_and_envelope_extras():
    rec = dict(reasons=[], violation=0.0, extra=dict(dmin_obj=0.2, ball_speed=0.0)); r = V.report(rec, "impact"); assert "NOT_HIT" in r.reasons and not r.object_interaction
    rec = dict(reasons=[], violation=0.0, extra=dict(peak_util=1.2)); r = V.report(rec, "stationary"); assert not r.actuator_torque and not r.envelope
    rec = dict(reasons=[], violation=0.0, extra=dict(overspeed=1.3)); assert V.report(rec, V.Policy("tight", overspeed_tol=1.1)).reasons == ["OVERSPEED"]


def test_policy_for_every_family():
    from movement_discovery.core.contracts import Family
    for f in Family: assert isinstance(V.policy_for(f.value), V.Policy)
    assert V.policy_for("aerial").name == "aerial" and V.policy_for("force_interaction").name == "sustained_force" and V.policy_for("compound").name == "stationary"
    assert all(k in V._FLAG for k in V.REASONS)
