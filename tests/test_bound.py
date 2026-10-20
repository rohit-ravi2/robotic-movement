"""Closed-form checks on the bound module. Each case has an answer computable by hand."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import bound as B

def test_flat_torque_constant_work():
    """floor=1 makes f_angle identically 1, so work = tau * |dtheta| exactly."""
    b = B.Body()
    b.j["knee"].update(floor=1.0, peak=0.0, width=10.0, tau=500.0)
    w = b.work_rom("knee", 0.0, 2.0)
    assert abs(w - 500.0*2.0) < 1e-6, w

def test_f_angle_endpoints():
    b = B.Body()
    p = b.j["hip"]
    assert abs(b.f_angle("hip", p["peak"]) - 1.0) < 1e-12
    assert abs(b.f_angle("hip", p["peak"]+p["width"]) - p["floor"]) < 1e-12
    assert abs(b.f_angle("hip", p["peak"]-p["width"]) - p["floor"]) < 1e-12

def test_f_angle_monotone_away_from_peak():
    b = B.Body(); p = b.j["knee"]
    xs = np.linspace(p["peak"], p["peak"]+p["width"], 50)
    f = b.f_angle("knee", xs)
    assert np.all(np.diff(f) <= 1e-12)

def test_power_bound_formula():
    b = B.Body()
    b.j["ankle"].update(floor=1.0, peak=0.0, width=10.0, tau=300.0, wmax=14.0)
    pp = b.peak_power("ankle", -0.5, 0.5)
    assert abs(pp - 300.0*14.0/4.0) < 1e-9, pp

def test_energy_ledger_closes():
    """KE must equal W_used - M g dz exactly."""
    b = B.Body()
    r = B.subset_bound(b, ("hip","knee","ankle"))
    assert abs(r["KE"] - max(0.0, r["W_used"] - r["PE"])) < 1e-9

def test_v_from_ke():
    b = B.Body(); r = B.subset_bound(b, ("knee",))
    assert abs(r["v_takeoff"] - np.sqrt(2*r["KE"]/b.total_mass)) < 1e-12

def test_monotone_in_subset():
    """adding a joint cannot lower the ROM-bound work"""
    b = B.Body()
    a = B.subset_bound(b, ("knee",), T_push=10.0)
    c = B.subset_bound(b, ("knee","hip"), T_push=10.0)
    assert c["W_rom"] >= a["W_rom"] - 1e-9

def test_mass_sums_to_body_mass():
    b = B.Body()
    assert abs(b.total_mass - b.M) < 1e-6, (b.total_mass, b.M)

def test_zero_torque_zero_work():
    b = B.Body(); b.j["elbow"]["tau"] = 0.0
    assert b.work_rom("elbow", 0.0, 2.0) == 0.0

def test_no_free_energy_when_no_joints_move():
    b = B.Body()
    q = B.crouch_pose(b)
    r = B.subset_bound(b, (), start=q)
    assert r["W_used"] == 0.0 and r["v_takeoff"] == 0.0


def test_conditional_dominates_power_branch():
    """Closed-form: with u = dtheta/(t*omega0), conditional/power = 4u(1-u) <= 1 for all u.
    So the conditional branch is never looser than the power branch, and the power branch
    can never bind strictly. Verified numerically across a wide grid."""
    b = B.Body()
    b.j["knee"].update(floor=1.0, peak=0.0, width=10.0, tau=500.0, wmax=18.0)
    for dth in (0.05, 0.5, 1.5, 2.5):
        for T in (0.05, 0.1, 0.25, 0.5, 2.0):
            wc = b.work_conditional("knee", 0.0, dth, T)
            wp = b.peak_power("knee", 0.0, dth)*T
            assert wc <= wp + 1e-9, (dth, T, wc, wp)

def test_long_push_off_recovers_rom_branch():
    """As t_contact grows the forced mean speed vanishes and the conditional branch
    converges to the isometric ROM branch."""
    b = B.Body()
    r = B.subset_bound(b, ("hip","knee","ankle"), T_push=1000.0)
    for v in r["per_joint_work"].values():
        assert abs(v["conditional"] - v["rom"]) < 1e-3*max(1.0, v["rom"]), v

def test_per_joint_min_is_tighter_than_aggregate_min():
    """sum of per-joint minima <= min of the sums, always."""
    b = B.Body()
    for T in (0.1, 0.25, 0.409, 1.0):
        r = B.subset_bound(b, tuple(B.JOINTS), T_push=T)
        assert r["W_used"] <= min(r["W_rom"], r["W_power"]) + 1e-9, (T, r["W_used"])

def test_power_bound_value_matches_formula():
    b = B.Body()
    b.j["knee"].update(floor=1.0, peak=0.0, width=10.0, tau=500.0, wmax=18.0)
    T = 0.2
    r = B.subset_bound(b, ("knee",), T_push=T)
    expect = 500.0*18.0/4.0*T
    assert abs(r["per_joint_work"]["knee"]["power"] - expect) < 1e-6

def test_com_x_zero_when_upright():
    b = B.Body()
    a = dict(shank=0.0, thigh=0.0, trunk=0.0, uarm=0.0, farm=0.0)
    assert abs(b.com_x_abs(a) - b.foot_mid_x*b.seg["foot"]["m"]/b.total_mass) < 1e-9

def test_support_polygon_spans_foot_length():
    b = B.Body()
    heel, toe = b.support_polygon()
    assert abs((toe-heel) - b.seg["foot"]["L"]) < 1e-12

if __name__ == "__main__":
    fails = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            try: f(); print(f"  PASS {n}")
            except Exception as e: fails += 1; print(f"  FAIL {n}: {e}")
    print(("all passed" if not fails else f"{fails} FAILED"))
    sys.exit(1 if fails else 0)

def test_frontier_monotone_synthetic():
    """Monotonicity is a consequence of set inclusion, so it must hold for any rows that
    come from a correct formulation. This guards the frontier permanently."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    import robot_bound as RB
    good = [dict(subset=("a",), v_takeoff=1.0), dict(subset=("b",), v_takeoff=2.0),
            dict(subset=("a", "b"), v_takeoff=2.5)]
    assert RB.check_monotone(good) == []
    bad = [dict(subset=("a",), v_takeoff=3.0), dict(subset=("a", "b"), v_takeoff=2.0)]
    v = RB.check_monotone(bad)
    assert len(v) == 1 and v[0][0] == "a" and v[0][2] == "a+b"

def test_required_depth_is_monotone_in_demand():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    import robot_bound as RB
    rows = [dict(subset=("a",), v_takeoff=1.0), dict(subset=("b",), v_takeoff=2.0),
            dict(subset=("a", "b"), v_takeoff=3.0)]
    rd = RB.required_depth(rows, [0.5, 1.5, 2.5, 3.5])
    depths = [r["depth"] for r in rd]
    assert depths[:3] == [1, 1, 2] and depths[3] is None, depths

def test_conditional_binds_when_sweep_is_fast():
    """A large sweep in a short time forces mean speed near omega_max, so the conditional
    branch collapses toward zero and must bind."""
    b = B.Body()
    b.j["knee"].update(floor=1.0, peak=0.0, width=10.0, tau=500.0, wmax=18.0)
    dth = 1.8; T = 0.12                     # wbar = 15 rad/s against wmax 18
    wc = b.work_conditional("knee", 0.0, dth, T)
    wr = b.work_rom("knee", 0.0, dth)
    expect = 500.0*(1.0 - (dth/T)/18.0)*dth
    assert abs(wc - expect) < 1e-6, (wc, expect)
    assert wc < wr

def test_conditional_does_not_bind_when_sweep_is_slow():
    b = B.Body()
    b.j["knee"].update(floor=1.0, peak=0.0, width=10.0, tau=500.0, wmax=18.0)
    wc = b.work_conditional("knee", 0.0, 0.2, 10.0)   # wbar = 0.02 rad/s
    wr = b.work_rom("knee", 0.0, 0.2)
    assert wc > 0.99*wr, (wc, wr)

def test_conditional_zero_when_sweep_exceeds_speed_limit():
    b = B.Body()
    b.j["ankle"].update(floor=1.0, peak=0.0, width=10.0, tau=300.0, wmax=14.0)
    assert b.work_conditional("ankle", 0.0, 3.0, 0.1) == 0.0    # wbar = 30 > wmax
