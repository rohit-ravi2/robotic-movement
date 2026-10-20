"""WALKING construction (M5, 2026-10-18): the LOW-SPEED REGIME of the sprint construction, generalised to double support.

Same pipeline as engine.sprint (body + objective + support contract -> scaffold -> search -> gates); what changes is the
support contract: a walk keeps at least one foot loaded at all times and BOTH feet loaded for a double-support interval each step
(per-leg duty factor (1 + delta)/2 > 0.5, no flight).

  stride   = SPEC-DERIVED walking ceiling (walk_ceiling): for each speed v, every (step length s, double-support fraction delta, stance
             hip height h0 = hr L) on a grid is checked on the body spec (thigh/shank lengths, masses, rated torque/speed of hip and
             knee under the rating table): reach (ankle within 0.97 L of the hip at every instant), stance hip/knee torque and speed
             inside the driving envelope (LIPM ground force through the leg), swing hip/knee torque and speed (planar inverse dynamics of
             the lumped leg on the scaffold's own swing path) within KNEE_U of the rating, and T_ds >= pi x TC_LAG (the weight transfer
             inside the project actuator-lag band; the sprint's contact-time rule). The PREFERRED stride at v is the feasible one with
             the least predicted positive mechanical work per metre (LIPM COM positive work + swing positive work); the predicted
             maximum walking speed is the largest v with any feasible stride.
  scaffold = one gait cycle (two steps, mirrored legs). COM: the periodic LIPM solution (height z_c from h0) with the ZMP on the
             stance ankle in single support and moving linearly ankle-to-ankle in double support (x and y). Feet: stance foot pinned;
             swing ankle a world-frame rest-to-rest quintic from the rear to the front foothold with the sole lifted CLEAR x
             64 s^3 (1-s)^3. Pelvis = COM - the posture's COM offset (one correction pass), upright, at hip height h0. Numerical IK on
             the MJCF (ankle position + level sole relative to the pelvis). ARMS: sprint's angular-momentum counter-swing (fundamental of
             the legs' yaw momentum). FEEDFORWARD: inverse dynamics of the swing/upper-body reference (stance legs zero).
  control  = phase clock synchronised to touchdowns (held at a scheduled touchdown until the foot loads, jumped forward on an early
             touchdown after 80 % of the swing); the reference is re-anchored at every touchdown on the foot that landed (position and
             heading), so drift is absorbed each step. FORCE-CONTROLLED SUPPORT: total GRF = M (LIPM acceleration + PD on COM about the
             anchored reference) + M (g + PD on height), attitude moment I_c (-om^2 att - 2 om rate), distributed over the stance feet by
             the reference load share (DS: linear transfer) x each foot's contact availability, residual moment clipped to each sole;
             tau = -J^T [F; M]. Swing: world-frame foot target (latched liftoff point -> capture-point-corrected foothold) mapped to joints
             through the scaffold's IK Jacobian about the reference, PD + ID; stance legs PD on the reference pelvis over the actual foot,
             gain x 0.1 under load. Feedback on the state predicted one actuator lag ahead (the sprint's / K2 stand's lag-aware rule);
             reference and feedforward advanced by the lag.
  search   = CMA-ES (seed >= 1) over step-length amp + time scale + pelvis lean + a 13-dim CONTROLLER-GAIN block (foothold gains, CoP
             DCM gain, attitude bandwidth, CoP margin, touchdown depth, leg tracker bandwidth, clock mode, stance-yaw gain); sprint's
             periodic torque knots exist but are frozen at 0 (WALK_KNOTS=1 frees them); worse-of-dt score; valid gaits are ranked by
             energy per metre (the searcher chooses stride and cadence at the commanded speed).
  gates    = actuator envelope + back-EMF + OVERSPEED 1.1, actuator lag (default 10 ms), energy closure, no non-foot contact, foot
             slip <= SLIP_MAX per stance, no fall, NO FLIGHT (no both-unloaded interval >= 20 ms after the transient), DOUBLE SUPPORT
             every step (both feet loaded >= 20 ms between consecutive touchdowns), >= 10 consecutive periodic strides (right
             touchdown states), speed within +-SPEED_TOL of the command, heading gate; all at 1 ms AND 0.5 ms.
usage (from src/):
  python engine/walk.py predict                         spec-only prediction for G1 and H1 -> results/movements/M5_walking/prediction.json
  python engine/walk.py prior G1 0.6 [turn_deg]         the prior (z = 0) at both dt
  python engine/walk.py search G1 0.6 iters pop workers seed [turn_deg] [warm_json]
  python engine/walk.py validate <gait json> [workers]  fresh-plant validation + perturbations
"""
import os, sys, json, time, itertools
os.environ.setdefault("SPRINT_LAG", "0.010")          # M5: 10 ms first-order actuator lag (the sprint module reads it at import)
os.environ.setdefault("SPRINT_RATINGS", "published")  # Unitree URDF effort AND the vendor's published joint maximum (G1 knee 120)
os.environ.setdefault("STOCK_CONTACT", "1")           # every robot keeps its MJCF contact (H1 solref 0.02)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
import branch_i as BI, plant as P
from engine import core, sprint as SP, sprint_ceiling as SC

G = 9.81
RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "movements", "M5_walking"))
LAG = SP.ACT_TAU
TDS_MIN = np.pi*LAG                # double support >= pi x the plant lag (31 ms at 10 ms): the weight transfer's half-sine inside the actuator lag's band (the sprint's t_c rule, applied to the plant lag the walk is built for)
CLEAR = 0.03                       # swing sole clearance (the sprint's / ceiling's 3 cm)
KNEE_U = SP.KNEE_U                 # swing joints may use 80 % of the rated torque and speed
SLIP_MAX = 0.02                    # m of foot travel while loaded, per stance (the standing support gate's 2 cm)
SPEED_TOL = 0.05                   # measured speed within +-5 % of the command
N_TRANS = 2                        # transient cycles
N_PER = 10                         # consecutive periodic strides required
PER_TOL = dict(SP.PER_TOL)         # sprint tolerances (z 2 cm, pitch/roll 0.05, q 0.08 RMS, qd 1.5 RMS, v 0.10)
FLIGHT_CONFIRM = BI.FLIGHT_CONFIRM # 20 ms
HEADING_TOL = 0.10                 # rad (straight: |yaw_end| <= 0.10; turn: |yaw_end - turn| <= 0.10)
ROBOT_SOLREF = {r: (None if os.environ.get("STOCK_CONTACT") == "1" else BI.ROBOTS[r].get("geom_solref")) for r in ("G1", "H1")}
SUPPLIED = dict(
    cycle="two-step walking cycle, mirror-symmetric legs; right touchdown at phase 0, double support [0, T_ds), right single support [T_ds, T), double support [T, T + T_ds), left single support [T + T_ds, 2T)",
    stride_choice="spec-derived (walk_ceiling): least predicted positive work per metre among the feasible (s, delta, hr) on the grid s 0.05..0.80 (0.025), delta 0.05..0.60 (0.05), hr 0.70..0.96 (6); T_ds >= pi x plant lag; single support >= 80 ms; utilisations <= 1 (stance: envelope; swing: KNEE_U of rating)",
    com="periodic LIPM, z_c = posture COM height, ZMP at the stance ankle in single support, linear ankle-to-ankle in double support (a sole-centre ZMP variant, WALK_ZMPSOLE=1, was tested on G1/H1 0.6 and not adopted)",
    reach="the scaffold lowers the predicted hip height (0.01 steps) until the stance ankle stays within 0.93 L of the hip (the predictor allows 0.97 L); recorded per stride (hr_predicted, reach_adjusted)",
    swing=f"world rest-to-rest quintic ankle path rear -> front foothold, sole bump {CLEAR} m x 64 s^3 (1-s)^3, level sole; numerical IK on the MJCF",
    arms="sprint angular-momentum counter-swing (cancels the fundamental of the legs' yaw momentum)",
    support="force control: DCM tracking through the CoP, p_cmd = p_ref + 2 (xi - xi_ref) (xi = c + v/omega), clipped to the loaded soles (share-weighted centres +- 70 % of the sole box); F_xy = M omega^2 (c - p_cmd) tilted by the attitude moment (M_x = h dF_y, M_y = -h dF_x); F_z = M (g + om_z^2 ez + 2 om_z dez); om_z = min(1/0.040, 0.5/lag); attitude moment I_c(-om_a^2 att - 2 om_a rate), om_a = 2 sqrt(g/z_c); commanded CoP inside 70 % of the sole; share by reference load x contact availability (load/0.15 Mg); per-sole CoP clip, |F_xy| <= 0.8 F_z, |M_z| <= 0.05 F_z",
    placement="foothold correction: d = (c - c_ref) + K_V (v - v_ref)/omega_LIPM (heading frame), K_V = (1.0 sagittal, 1.5 lateral), clipped +-0.15 m (x) / +-0.10 m (y), feet never closer than 0.5 x the nominal width, latched at 80 % swing",
    ds_contract="off (an opt-in hold at the end of double support until the new stance foot carries 30 % BW was tested on G1 and made it worse)",
    anchor="reference re-anchored at every touchdown on the landed foot (position, heading = turn reference at that event)",
    lag_comp="feedback on the state predicted one lag ahead (q + qd tau, c + v tau, attitude + rate tau); reference and feedforward advanced by the lag",
    tracker="PD per joint, omega 25 rad/s legs / 20 rad/s upper body (sprint's), zeta 1, locked-chain inertia; stance-leg PD x 0.1 under load",
    gains="SEARCHED controller-gain block g in [-1,1]^13 (g = 0 = the construction defaults): foothold velocity gains kvx = 1.0 x 2^g0, kvy = 1.5 x 2^g1; CoP DCM gain 2 x 2^g2; attitude bandwidth (2 x 2^g3) x omega_LIPM; CoP margin 0.7 + 0.2 g4; touchdown depth 0.02 x 2^g5 m; leg tracker bandwidth 25 x 2^(g6/2) rad/s; clock mode g7 > 0 = TIMED (fixed step timing, re-anchor at the scheduled event on the expected foot) else event-synchronised; g8 > 0 = stance hip-yaw PD at full gain; contact-share load 0.15 x 4^g9 of body weight (a stance foot takes its full reference share of the support force at this load); touchdown acceptance at 0.75 + 0.15 g10 of the swing (event clock); foothold-correction clip x 2^g11 (default +-0.15 m sagittal, +-0.10 m lateral); minimum foot separation (0.5 + 0.3 g12) x the nominal width. The defaults were set while debugging the G1 0.6 m/s prior (listed, not hidden); the search adapts them per robot and speed",
    knots="(frozen at 0 in the M5 searches unless WALK_KNOTS=1) sprint layout: 8 periodic torque knots per right-leg and right-arm joint (left mirrored half a cycle later), 4 per centre joint; amp 0.8..1.2 (step length), tscale 0.85..1.15, lean -0.25..0.25 rad (centred on the upright scaffold)",
    start="walking start at mid right single support, COM at the reference velocity, zero angular momentum about the COM",
    turn="heading reference ramps linearly by the turn angle over 5 strides starting after cycle 4; footholds rotate with the anchor heading; swing hip yaw lands the foot at the reference heading",
    lag=f"first-order actuator lag {LAG*1000:.0f} ms (plant), torque-speed envelope with droop, back-EMF brake, OVERSPEED gate 1.1",
    ratings=SP.SUPPLIED["ratings"],
    contact="stock MJCF contact on every geom (STOCK_CONTACT=1; G1 solref 0.02 floor; H1 0.02)" if os.environ.get("STOCK_CONTACT") == "1" else "branch_i override (H1 0.004)",
    friction="MuJoCo pyramidal cone, condim 3, mu 1.0 (model)",
    solver="Newton, 50 iterations (project apply_dt; a DEPARTURE from the MJCFs' own settings, G1 5 and H1 100 iterations); control every physics step (1 ms and 0.5 ms)",
    tc_lag="the project lag TC_LAG = 40 ms (not the 10 ms plant lag) sets the support-height bandwidth cap om_z = min(1/0.040, 0.5/lag) and the scaffold feedforward's velocity smoothing window (0.5 x 40 ms)",
    robust_score="repair round 2 (WALK_ROBUST): candidates scored at 1 ms on 3 mass/inertia draws (seeds 21-23), 2 lateral shoves at seed-drawn phases (+y 5.31, +y 3.64), lag_fric (real mu 0.7, lag +25 %), shove_back, start_noise, shove_lat; lexicographic classes (nominal gate failure > n failing draws > off band > valid, 0.5 ms nominal decides among valid); labels from the fresh held-out F1-F4 only",
    friction_override="perturbation friction (mu) is applied to the floor AND every foot geom (repair round 2; before it, the floor-only override was a no-op because MuJoCo takes the max of the pair)",
    nominal="the plant's settled stance (branch_i settle); if that settle fell (H1 at stock contact), the MJCF keyframe with its lowest foot point on the floor (recorded per result)",
    held_joints="joints outside the driven set (G1: shoulder yaw, wrists; H1: shoulder yaw) are position servos kp = I_eff/dt^2 (project convention, dt-dependent)",
    gates=f"slip <= {SLIP_MAX} m/stance, no flight >= 20 ms, double support >= 20 ms every step, {N_PER} consecutive periodic strides (tolerances {PER_TOL}), speed within +-{SPEED_TOL*100:.0f}%, heading within {HEADING_TOL} rad, no fall, no non-foot contact, overspeed <= 1.1, energy closure; both dt",
)


# ====================================================================== LIPM
def lipm_periodic(zmp, res, om, shift, sub=4):
    """Periodic LIPM solution x'' = om^2 (x - p) over one cycle (p sampled at res, held over each sample), x(T) = x(0) + shift,
    x'(T) = x'(0). Returns x, xd, xdd sampled at res."""
    h = res/sub; ch, sh = np.cosh(om*h), np.sinh(om*h)
    def run(x0, v0):
        x, v = float(x0), float(v0); X = np.zeros(len(zmp)); V = np.zeros(len(zmp))
        for k, p in enumerate(zmp):
            X[k], V[k] = x, v
            for _ in range(sub): e = x - p; x, v = p + e*ch + v*sh/om, e*om*sh + v*ch
        return X, V, x, v
    _, _, bx, bv = run(0.0, 0.0); _, _, x1, v1 = run(1.0, 0.0); _, _, x2, v2 = run(0.0, 1.0)
    A = np.array([[x1 - bx, x2 - bx], [v1 - bv, v2 - bv]]); b = np.array([bx, bv])
    x0, v0 = np.linalg.solve(A - np.eye(2), np.array([shift, 0.0]) - b)
    X, V, _, _ = run(x0, v0); return X, V, om*om*(X - np.asarray(zmp))


def zmp_cycle(N, res, T, Tds, s, w):
    """ZMP over one cycle: x from the rear ankle to the front one in double support, on the stance ankle otherwise."""
    t = np.arange(N)*res; px = np.zeros(N); py = np.zeros(N)
    for k, tk in enumerate(t):
        if tk < Tds: f = tk/Tds; px[k] = -s + s*f; py[k] = w - 2*w*f
        elif tk < T: px[k] = 0.0; py[k] = -w
        elif tk < T + Tds: f = (tk - T)/Tds; px[k] = s*f; py[k] = -w + 2*w*f
        else: px[k] = s; py[k] = w
    return px, py


# ====================================================================== spec-derived walking ceiling / prediction
def walk_body(robot):
    """Body quantities for the ceiling: sprint body spec (rating table applied) + posture quantities from the model."""
    B = SP.body(robot); S = SP.plant(robot, core.DT_OPT); nominal_pose(S); m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d)
    c = SC.CHAINS[robot]; hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, c["hipb"]); ab = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, c["ankleb"])
    B = dict(B); B["M"] = float(sum(m.body_mass[b] for b in S["robot_bodies"]))
    B["dcx"] = float(d.subtree_com[1][0] - d.xpos[hb][0]); B["dcz"] = float(d.subtree_com[1][2] - d.xpos[hb][2])
    B["w"] = float(abs(d.xpos[ab][1] - d.subtree_com[1][1]))
    L1, L2 = B["L1"], B["L2"]; B["I_hip"] = B["m_thigh"]*(L1*L1/3) + B["m_shank"]*((L1 + 0.4*L2)**2 + L2*L2/12)
    B["I_knee"] = B["m_shank"]*((0.4*L2)**2 + L2*L2/12); B["mr_hip"] = B["m_thigh"]*L1/2 + B["m_shank"]*(L1 + 0.4*L2)
    rt = SP.rating_table(robot) if SP.RATINGS != "peak" else {}
    B["ratings_used"] = {"hip_pitch": B["hip"]["tau"], "knee": B["knee"]["tau"]}
    fb_ = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, c["ankleb"]); zc_ = [d.contact[i].pos[2] for i in range(d.ncon) if d.contact[i].geom1 in S["feet"][0] or d.contact[i].geom2 in S["feet"][0]]
    if zc_: B["hf"] = float(d.xpos[fb_][2] - np.mean(zc_))          # ankle above the sole's floor contacts in the nominal pose (the ceiling's value read from a fallen settle for H1)
    return B


def _planar_leg(xa, za, L1, L2):
    th, ka = SC.ik(xa, za, L1, L2); return np.asarray(th), np.asarray(ka)


def _util(tau, w, tmax, om, u=1.0):
    """utilisation of the torque-speed envelope: driving (tau w > 0) |tau| <= u tmax (1 - |w|/om), absorbing |tau| <= u tmax;
    speed |w| <= 1.1 om (stance, the OVERSPEED gate) or u om (swing)."""
    drive = tau*w > 0; cap = u*tmax*np.where(drive, np.clip(1.0 - np.abs(w)/om, 1e-3, 1.0), 1.0)
    return float(max(np.max(np.abs(tau)/cap), np.max(np.abs(w))/(1.1*om) if u == 1.0 else np.max(np.abs(w))/(u*om)))


def gait_check(B, v, s, delta, hr, res=0.002, full=False, strict=True):
    """Feasibility and predicted work per metre of one walking stride on the body spec. Returns dict or None (infeasible)."""
    L1, L2 = B["L1"], B["L2"]; L = L1 + L2; h0 = hr*L; T = s/v; Tds = delta*T
    if Tds < TDS_MIN - 1e-9 or T - Tds < 0.08: return None
    zc = h0 + B["hf"] + B["dcz"]; om = np.sqrt(G/zc); Tc = 2*T; N = int(round(Tc/res))
    if om*Tc > 25 or T > 1.5: return None                                    # steps longer than 1.5 s are not a walk at this speed (numerics)
    px, py = zmp_cycle(N, res, T, Tds, s, B["w"]); X, VX, AX = lipm_periodic(px, res, om, 2*s, sub=2); Y, VY, AY = lipm_periodic(py, res, om, 0.0, sub=2)
    t = np.arange(N)*res; M = B["M"]
    # STANCE (right leg: [0, T + Tds)): hip over the foot; GRF share 1 in single support, linear in double support
    st = t < T + Tds; share = np.where(t < Tds, t/Tds, np.where(t < T, 1.0, np.where(t < T + Tds, 1.0 - (t - T)/Tds, 0.0)))
    xh = X - B["dcx"]; xa = -(xh[st]); za = -h0*np.ones(st.sum())
    if np.any(np.hypot(xa, za) > 0.97*L): return None
    th, ka = _planar_leg(xa, za, L1, L2)
    if np.any(ka < B["knee"]["rng"][0]) or np.any(ka > B["knee"]["rng"][1]): return None
    Fx = M*AX[st]*share[st]; Fz = M*G*share[st]
    # planar Jacobian of the ankle relative to the hip w.r.t. (th, ka); torques that hold the ground force: tau = J^T F
    j11 = L1*np.cos(th) + L2*np.cos(th - ka); j12 = -L2*np.cos(th - ka); j21 = L1*np.sin(th) + L2*np.sin(th - ka); j22 = -L2*np.sin(th - ka)
    t_h = j11*Fx + j21*Fz; t_k = j12*Fx + j22*Fz; wth = np.gradient(th, res); wka = np.gradient(ka, res)
    u_st = max(_util(t_h, wth, B["hip"]["tau"], B["hip"]["omega"]), _util(t_k, wka, B["knee"]["tau"], B["knee"]["omega"]))
    if u_st > 1.0 and strict: return None
    # SWING (right leg: [T + Tds, 2T)): world rest-to-rest quintic of the ankle, rear foothold (0) -> front (2s), sole bump CLEAR
    sw = ~st; tsw = T - Tds; u = (t[sw] - (T + Tds))/tsw; f = 10*u**3 - 15*u**4 + 6*u**5
    xa_w = 2*s*f; xa = xa_w - xh[sw]; za = -h0 + CLEAR*64*u**3*(1 - u)**3
    if np.any(np.hypot(xa, za) > 0.985*L): return None
    th, ka = _planar_leg(xa, za, L1, L2); wth = np.gradient(th, res); wka = np.gradient(ka, res); ath = np.gradient(wth, res); aka = np.gradient(wka, res)
    tau_h = B["I_hip"]*ath + G*B["mr_hip"]*np.sin(th); tau_k = B["I_knee"]*aka
    u_sw = max(_util(tau_h, wth, B["hip"]["tau"], B["hip"]["omega"], KNEE_U), _util(tau_k, wka, B["knee"]["tau"], B["knee"]["omega"], KNEE_U))
    if u_sw > 1.0 and strict: return None
    # PREDICTED POSITIVE WORK per cycle: COM (x, y; LIPM, flat) + swing hip/knee (two swings per cycle)
    Pc = M*(AX*VX + AY*VY); E_com = float(np.sum(np.maximum(Pc, 0.0))*res)
    E_sw = 2*float((np.sum(np.maximum(tau_h*wth, 0.0)) + np.sum(np.maximum(tau_k*wka, 0.0)))*res)
    J_m = (E_com + E_sw)/(2*s)
    out = dict(v=v, s=s, delta=delta, hr=hr, h0=h0, T=T, Tds=Tds, cadence_spm=60.0/T, step_hz=1.0/T, stride=2*s, duty=(1 + delta)/2, zc=zc, om=om,
               u_stance=u_st, u_swing=u_sw, E_com=E_com, E_swing=E_sw, J_per_m=J_m, cot=J_m/(M*G))
    if full: out.update(X=X, Y=Y, px=px, py=py)
    return out


S_GRID = np.round(np.arange(0.05, 0.801, 0.025), 4); D_GRID = np.round(np.arange(0.05, 0.601, 0.05), 3); HR_GRID = np.round(np.linspace(0.70, 0.96, 6), 3)
def preferred(B, v):
    best = None; n_ok = 0
    for s, dl, hr in itertools.product(S_GRID, D_GRID, HR_GRID):
        r = gait_check(B, v, s, dl, hr)
        if r is None: continue
        n_ok += 1
        if best is None or r["J_per_m"] < best["J_per_m"]: best = r
    if best is not None: best["n_feasible"] = n_ok
    return best


def least_utilised(B, v):
    """BEYOND THE SPEC CEILING (flagged): the geometrically valid stride with the smallest max(stance, swing) utilisation."""
    best = None
    for s, dl, hr in itertools.product(S_GRID, D_GRID, HR_GRID):
        r = gait_check(B, v, s, dl, hr, strict=False)
        if r is None: continue
        r["u_max"] = max(r["u_stance"], r["u_swing"])
        if best is None or r["u_max"] < best["u_max"]: best = r
    if best is not None: best["beyond_ceiling"] = True
    return best


def predict(robots=("G1", "H1"), speeds=(0.3, 0.6, 1.0, 1.5), vmax_grid=np.round(np.arange(0.6, 3.01, 0.05), 2)):
    out = dict(rule=SUPPLIED["stride_choice"], tds_min=TDS_MIN, lag=LAG, ratings=SUPPLIED["ratings"], robots={})
    for r in robots:
        B = walk_body(r); pr = {}
        for v in speeds:
            b = preferred(B, v) or least_utilised(B, v); pr[str(v)] = None if b is None else {k: (round(x, 4) if isinstance(x, float) else x) for k, x in b.items()}
            print(r, v, None if b is None else (round(b["s"], 3), b["delta"], round(b["hr"], 3), round(b["cadence_spm"], 1), round(b["cot"], 3), b.get("beyond_ceiling", False), round(b["u_swing"], 2)), flush=True)
        vmax = None; vmax_rec = None
        for v in vmax_grid:
            b = preferred(B, float(v))
            if b is not None: vmax, vmax_rec = float(v), b
            else: break
        # what binds just above vmax: the least-utilised infeasible reason is reported by relaxing each rule
        out["robots"][r] = dict(body={k: (round(x, 4) if isinstance(x, float) else x) for k, x in B.items() if not isinstance(x, dict)} | dict(hip=B["hip"], knee=B["knee"]),
                                preferred=pr, v_max=vmax, v_max_stride=None if vmax_rec is None else {k: (round(x, 4) if isinstance(x, float) else x) for k, x in vmax_rec.items()},
                                froude_walk_run=float(np.sqrt(G*(B["L1"] + B["L2"]))), froude_half=float(np.sqrt(0.5*G*(B["L1"] + B["L2"]))))
        print(r, "v_max", vmax, flush=True)
    os.makedirs(RES, exist_ok=True); json.dump(out, open(f"{RES}/{os.environ.get('WALK_PRED_OUT', 'prediction.json')}", "w"), indent=1, default=float); return out


def nominal_pose(S):
    """The plant's settled stance is the nominal pose; if the plant's own settle FELL (H1 at stock contact solref 0.02: the settled base
    lies on the floor), the MJCF keyframe placed with its lowest foot point on the floor is used instead (flag returned)."""
    q = np.array(S["q_settled"]); w = abs(q[3]); q0 = np.array(S["q0"])
    if w > np.cos(0.1) and q[2] > 0.8*q0[2]: return "settled"
    m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = q0; mujoco.mj_kinematics(m, d)
    low = min(P.geom_low(m, d, g) for g in S["feet"][0] | S["feet"][1]); q0[2] -= low + 0.0005
    S["q_settled"] = q0; S.setdefault("v_settled", np.zeros(m.nv)); return "keyframe_on_floor (the plant's settle fell)"


ZMP_AT_SOLE = float(os.environ.get("WALK_ZMPSOLE", 0.0))   # 0: ZMP reference at the ankle (default); 1: at the sole centre (tested, not adopted)
REACH_SCAFFOLD = float(os.environ.get("WALK_REACH", 0.93))             # the SCAFFOLD keeps the stance ankle within 0.93 L of the hip (knee-bend margin for tracking errors)
def reach_margin(B, st):
    """Lower the stance hip height (hr, 0.01 steps) until the stance leg's longest extension over the cycle is <= REACH_SCAFFOLD x L
    (the predictor allowed 0.97 L); every other stride quantity kept. Flagged in the stride record."""
    if st is None: return None
    L = B["L1"] + B["L2"]; hr = st["hr"]; hr0 = hr
    while hr > 0.6:
        r = gait_check(B, st["v"], st["s"], st["delta"], hr, strict=False, full=True)
        if r is not None:
            xh = r["X"] - B["dcx"]; t = np.arange(len(xh))*0.002; stn = t < r["T"] + r["Tds"]
            if np.max(np.hypot(xh[stn], r["h0"])) <= REACH_SCAFFOLD*L: break
        hr -= 0.01
    r = {k: v for k, v in r.items() if k not in ("X", "Y", "px", "py")}
    r.update(hr_predicted=hr0, reach_adjusted=bool(hr < hr0 - 1e-9), beyond_ceiling=st.get("beyond_ceiling", False), n_feasible=st.get("n_feasible")); return r


# ====================================================================== scaffold
class PLeg(SP.Leg):
    """Sprint's numerical leg IK, with the ankle position expressed relative to the PELVIS (base at the origin, upright)."""
    def fk(self, qj):
        d = self.d; d.qpos[:] = self.q_nom
        for a, v in zip(self.qa, qj): d.qpos[a] = v
        mujoco.mj_kinematics(self.m, d); p = d.xpos[self.ankb] - d.xpos[1]; R = d.xmat[self.ankb].reshape(3, 3)
        return np.array([p[0], p[1], p[2], np.arctan2(-R[2, 0], R[2, 2])] + ([np.arctan2(R[2, 1], R[2, 2])] if len(self.qa) == 5 else []))
    def target(self, x, y, z):
        return np.array([x, y, z, self.ori0[0]] + ([self.ori0[1]] if len(self.qa) == 5 else []))


def scaffold(S, B, st, res=0.001):
    m = S["m"]; types = S["types"]; nT = len(types); idx = {t: i for i, t in enumerate(types)}; robot = S["robot"]
    v, s, T, Tds, h0 = st["v"], st["s"], st["T"], st["Tds"], st["h0"]; Tc = 2*T; N = int(round(Tc/res)); t = np.arange(N)*res
    zc = h0 + B["hf"] + B["dcz"]; om = np.sqrt(G/zc); w = B["w"]
    # SOLE POLYGON from the settled stance's own floor contacts (right foot, ankle frame): heel/toe = contact x range; the lateral
    # CoP range is half the contact y half-range (the heel of both robots is a single central contact: a triangle-like support)
    ab = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, SC.CHAINS[robot]["ankleb"].replace("left", "right")); d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d); Ra = d.xmat[ab].reshape(3, 3); pts = []
    for i_ in range(d.ncon):
        cn = d.contact[i_]
        if cn.geom1 in S["feet"][1] or cn.geom2 in S["feet"][1]: pts.append(Ra.T @ (cn.pos - d.xpos[ab]))
    pts = np.array(pts); sole = dict(heel=float(pts[:, 0].min()), toe=float(pts[:, 0].max()), half_w=float(max(0.25*np.ptp(pts[:, 1]), 0.005)), hf=float(-pts[:, 2].mean()), n_contacts=len(pts))
    # the LIPM's ZMP reference sits at the SOLE CENTRE (ankle + the sole's x-centre), not the ankle (H1's ankle is at its heel)
    px, py = zmp_cycle(N, res, T, Tds, s, w); px = px + ZMP_AT_SOLE*0.5*(sole["heel"] + sole["toe"]); CX, VX, AX = lipm_periodic(px, res, om, 2*s); CY, VY, AY = lipm_periodic(py, res, om, 0.0)
    # foot world positions (ankle body origin) over the cycle: right [0]: at 0 in [0, T+Tds), swing to 2s; left [1]: at -s until T+... -> s
    hf = B["hf"]; FT = np.zeros((N, 2, 3)); SWP = np.zeros((N, 2)); SH = np.zeros((N, 2))
    tsw = T - Tds
    for k, tk in enumerate(t):
        # right leg
        if tk < T + Tds: FT[k, 0] = [0.0, -w, hf]; SH[k, 0] = tk/Tds if tk < Tds else (1.0 if tk < T else 1.0 - (tk - T)/Tds)
        else: u = (tk - T - Tds)/tsw; FT[k, 0] = [2*s*SP.mj_(u), -w, hf + CLEAR*64*u**3*(1 - u)**3]; SWP[k, 0] = u
        # left leg (stance [T, 2T + Tds) mod Tc, i.e. [0, Tds) of this cycle on the rear foothold -s, swing [Tds, T) to s)
        if tk < Tds: FT[k, 1] = [-s, w, hf]; SH[k, 1] = 1.0 - tk/Tds
        elif tk < T: u = (tk - Tds)/tsw; FT[k, 1] = [-s + 2*s*SP.mj_(u), w, hf + CLEAR*64*u**3*(1 - u)**3]; SWP[k, 1] = u
        else: FT[k, 1] = [s, w, hf]; SH[k, 1] = (tk - T)/Tds if tk < T + Tds else 1.0
    legs = {"right": PLeg(S, "right"), "left": PLeg(S, "left")}
    d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; d.qpos[:3] = 0; d.qpos[3:7] = [1, 0, 0, 0]; mujoco.mj_kinematics(m, d)
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, SC.CHAINS[robot]["hipb"].replace("left", "right"))
    pel_hip_z = float(d.xpos[1][2] - d.xpos[hb][2])
    q_up = np.array([S["q_settled"][S["jarr"]["qad"][i]] for i in range(nT)])
    dcom = np.zeros((N, 3)); dcom[:, 0] = float(d.subtree_com[1][0] - d.xpos[1][0]); dcom[:, 1] = float(d.subtree_com[1][1] - d.xpos[1][1])
    PZ = hf + h0 + pel_hip_z
    def solve_all(dcom):
        Q = np.repeat(q_up[None], N, 0); JI = [np.zeros((N, len(legs["right"].qa), len(legs["right"].qa))), np.zeros((N, len(legs["left"].qa), len(legs["left"].qa)))]
        PEL = np.zeros((N, 3)); PEL[:, 0] = CX - dcom[:, 0]; PEL[:, 1] = CY - dcom[:, 1]; PEL[:, 2] = PZ; err = 0.0; REL = np.zeros((N, 2, 3))
        for li, side in enumerate(("right", "left")):
            lg = legs[side]; qj = np.array([S["q_settled"][a] for a in lg.qa]); cols = [idx[n] for n in lg.names]
            for k in range(N):
                rel = FT[k, li] - PEL[k]; REL[k, li] = rel; tgt = lg.target(*rel)
                qs, e, Jm = lg.solve(tgt, qj)
                if e > 1e-4: qs2, e2, J2 = lg.solve(tgt, np.array([S["q_settled"][a] for a in lg.qa]))
                if e > 1e-4 and e2 < e: qs, e, Jm = qs2, e2, J2
                qj = qs; err = max(err, e); Q[k, cols] = qs; JI[li][k] = np.linalg.pinv(Jm)
        return Q, JI, PEL, REL, err
    Q, JI, PEL, REL, err = solve_all(dcom)
    # one correction pass: the posture's actual COM offset from the pelvis per sample
    dd = mujoco.MjData(m)
    for k in range(N):
        dd.qpos[:] = S["q_settled"]; dd.qpos[:3] = PEL[k]; dd.qpos[3:7] = [1, 0, 0, 0]; dd.qpos[S["jarr"]["qad"]] = Q[k]; mujoco.mj_kinematics(m, dd); mujoco.mj_comPos(m, dd)
        dcom[k] = dd.subtree_com[1] - dd.xpos[1]
    Q, JI, PEL, REL, err = solve_all(dcom)
    QD = (np.roll(Q, -1, 0) - np.roll(Q, 1, 0))/(2*res)
    # hip-yaw sign per leg (foot yaw per joint radian)
    ysign = {}
    for side in ("right", "left"):
        jn = SP.LEG[robot]["hy"].format(s=side); a = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]); lg = legs[side]
        for dv in (0.0, 0.1):
            dd.qpos[:] = lg.q_nom; dd.qpos[a] += dv; mujoco.mj_kinematics(m, dd); R = dd.xmat[lg.ankb].reshape(3, 3); yy = np.arctan2(R[1, 0], R[0, 0])
            if dv == 0.0: y0 = yy
        ysign[side] = float(np.sign(yy - y0)); ysign[side + "_i"] = idx[jn]
    # ARMS: sprint's angular-momentum counter-swing
    sp = [idx[n] for n in types if "shoulder_pitch" in n]; sgn = np.array([1.0 if types[i].startswith("right") else -1.0 for i in sp]); arm_info = None
    if sp:
        def angmom(k, extra=None):
            dd.qpos[:] = S["q_settled"]; dd.qpos[:3] = 0; dd.qpos[3:7] = [1, 0, 0, 0]; dd.qpos[S["jarr"]["qad"]] = Q[k]
            vv = np.zeros(m.nv); vv[0] = VX[k]; vv[1] = VY[k]; vv[S["jarr"]["dof"]] = QD[k]
            if extra is not None: vv[S["jarr"]["dof"][sp]] += extra
            dd.qvel[:] = vv; mujoco.mj_forward(m, dd); mujoco.mj_subtreeVel(m, dd); return dd.subtree_angmom[1].copy()
        Lz = np.array([angmom(k)[2] for k in range(N)])
        b = np.array([angmom(k, sgn)[2] - Lz[k] for k in range(0, N, max(1, N//20))]).mean()
        F_ = np.fft.rfft(Lz); F_[0] = 0; F_[2:] = 0; Lz1 = np.fft.irfft(F_, N); w_sp = -Lz1/b; ang = np.cumsum(w_sp)*res; ang -= ang.mean()
        for j, i in enumerate(sp): Q[:, i] = q_up[i] + sgn[j]*ang
        QD = (np.roll(Q, -1, 0) - np.roll(Q, 1, 0))/(2*res); arm_info = dict(Lz_ptp=float(np.ptp(Lz)), sp_ptp=float(np.ptp(ang)))
    # INVERSE-DYNAMICS FEEDFORWARD (base on the reference pelvis path, no contact), stance legs zero
    wq = max(1, int(round(0.5*SP.TC_LAG/res))); kq = np.ones(wq)/wq
    QDs = np.stack([np.convolve(np.concatenate([QD[-wq:, i], QD[:, i], QD[:wq, i]]), kq, "same")[wq:-wq] for i in range(nT)], 1)
    QDD = (np.roll(QDs, -1, 0) - np.roll(QDs, 1, 0))/(2*res); ID = np.zeros((N, nT)); qa_, dof_ = S["jarr"]["qad"], S["jarr"]["dof"]
    legcols = [[i for i, tn in enumerate(types) if tn.startswith(sd) and any(x in tn for x in ("hip", "knee", "ankle"))] for sd in ("right", "left")]
    for k in range(N):
        dd.qpos[:] = S["q_settled"]; dd.qpos[:3] = [0, 0, 5.0]; dd.qpos[3:7] = [1, 0, 0, 0]; dd.qpos[qa_] = Q[k]
        dd.qvel[:] = 0; dd.qvel[0] = VX[k]; dd.qvel[1] = VY[k]; dd.qvel[dof_] = QD[k]; dd.qacc[:] = 0; dd.qacc[0] = AX[k]; dd.qacc[1] = AY[k]; dd.qacc[dof_] = QDD[k]
        mujoco.mj_inverse(m, dd); ID[k] = dd.qfrc_inverse[dof_]
        for li in (0, 1):
            if SH[k, li] > 0 or SWP[k, li] == 0: ID[k, legcols[li]] = 0.0
    # centroidal quantities and sole polygon
    d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d); cc = d.subtree_com[1].copy(); I_c = np.zeros(3)
    for bb in S["robot_bodies"]:
        if bb == 0: continue
        r = d.xipos[bb] - cc; Rb = d.ximat[bb].reshape(3, 3); Ic = Rb @ np.diag(m.body_inertia[bb]) @ Rb.T
        for ax in range(3): I_c[ax] += float(Ic[ax, ax] + m.body_mass[bb]*(r @ r - r[ax]**2))
    ieff = np.array([1.0/S["Minv_key"][dof, dof] for dof in S["jarr"]["dof"]]) if False else None
    Mf = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, Mf, d.qM); ieff = np.array([Mf[dof, dof] for dof in S["jarr"]["dof"]])
    CR = np.stack([CX, CY, np.full(N, zc)], 1); VR = np.stack([VX, VY, np.zeros(N)], 1); AR = np.stack([AX, AY, np.zeros(N)], 1)
    # COM reference height: the posture's (after the correction pass) above the floor
    dd.qpos[:] = S["q_settled"]; dd.qpos[:3] = PEL[0]; dd.qpos[3:7] = [1, 0, 0, 0]; dd.qpos[qa_] = Q[0]; mujoco.mj_kinematics(m, dd); mujoco.mj_comPos(m, dd)
    low = min(P.geom_low(m, dd, g) for g in S["feet"][1]); CR[:, 2] = float(dd.subtree_com[1][2] - low)
    ikcols = [[idx[n] for n in legs[sd].names] for sd in ("right", "left")]
    return dict(Q=Q, QD=QD, ID=ID, JI=JI, REL=REL, PEL=PEL, FT=FT, PX=px, PY=py, ikcols=ikcols, SH=SH, SWP=SWP, CR=CR, VR=VR, AR=AR, N=N, res=res, T=T, Tds=Tds, Tc=Tc, s=s, v=v, w=w, h0=h0, zc=zc, om=om,
                I_c=I_c, ieff=ieff, sole=sole, ysign=ysign, legcols=legcols, ik_err=err, arm=arm_info, M=float(sum(m.body_mass[b] for b in S["robot_bodies"])), q_up=q_up,
                pel_z=PZ, low0=low)


def cached_scaffold(S, B, st):
    """The scaffold is deterministic in (robot, stride); cached on disk (pickle) so every worker / fresh process reuses it."""
    import pickle, hashlib
    import inspect
    src = hashlib.sha1((inspect.getsource(scaffold) + inspect.getsource(lipm_periodic) + inspect.getsource(zmp_cycle)).encode()).hexdigest()[:8]   # repair round 2: the key includes the scaffold code (stale pickles made outcomes cache-dependent)
    key = hashlib.sha1(json.dumps([S["robot"], round(st["v"], 4), round(st["s"], 4), round(st["delta"], 4), round(st["hr"], 4), CLEAR, LAG, src] + ([ZMP_AT_SOLE] if ZMP_AT_SOLE != 0.0 else []), default=float).encode()).hexdigest()[:12]
    pth = f"{RES}/cache/scaffold_{S['robot']}_{key}.pkl"
    if os.path.exists(pth):
        try: return pickle.load(open(pth, "rb"))
        except Exception: pass
    R = scaffold(S, B, st); os.makedirs(os.path.dirname(pth), exist_ok=True); tmp = pth + f".{os.getpid()}"
    pickle.dump(R, open(tmp, "wb")); os.replace(tmp, pth); return R


# ====================================================================== rollout
def rpy_mat(R):
    return np.arctan2(R[2, 1], R[2, 2]), np.arcsin(np.clip(-R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0])


def rotz(a): c, s_ = np.cos(a), np.sin(a); return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])


def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi


def cross(a, b): return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])


def initial_state(S, R, p, noise=None):
    """Walking start at mid right single support (phase T_ds + T_ss/2): reference posture, right sole on the floor, COM moving at
    the reference velocity, zero angular momentum about the COM."""
    m = S["m"]; J = S["jarr"]; res = R["res"]; ph0 = R["Tds"] + 0.5*(R["T"] - R["Tds"]); k = int(round(ph0/res)) % R["N"]
    q = np.array(S["q_settled"]); q[:3] = R["PEL"][k]; q[0] *= p["amp"]; q[3:7] = SP.rpy_quat(0.0, p["lean"], 0.0); q[J["qad"]] = R["Q"][k]
    if noise is not None: q[J["qad"]] += noise.get("q", 0.0)
    d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_kinematics(m, d)
    low = min(P.geom_low(m, d, g) for g in S["feet"][1]); q[2] += -low + 0.0005
    v = np.zeros(m.nv); v[J["dof"]] = R["QD"][k]/p["tscale"]
    tgt = np.array([R["VR"][k][0]*p["amp"]/p["tscale"] + (noise.get("vx", 0.0) if noise else 0.0), R["VR"][k][1]/p["tscale"], 0.0, 0.0, 0.0, 0.0])
    def mom(vv):
        d.qpos[:] = q; d.qvel[:] = vv; mujoco.mj_forward(m, d); mujoco.mj_subtreeVel(m, d); return np.concatenate([d.subtree_linvel[1], d.subtree_angmom[1]])
    f0 = mom(v); A = np.zeros((6, 6))
    for i in range(6):
        vp = v.copy(); vp[i] += 1.0; A[:, i] = mom(vp) - f0
    v[:6] += np.linalg.solve(A, tgt - f0)
    return q, v, ph0


def _chord(bs):
    """NET progress: sum of COM displacements between consecutive same-phase stride boundaries (right touchdowns). Lateral sway
    cancels at equal gait phase (the earlier path-length speed counted it as progress, +1-14 %); on a turn it follows the arc."""
    return float(sum(np.linalg.norm(np.asarray(b["comx"])[:2] - np.asarray(a["comx"])[:2]) for a, b in zip(bs[:-1], bs[1:])))


def rollout(S, R, p, n_cyc=None, turn=0.0, record=False, shove=None, noise=None, lag_plant=None, friction=None, mass_noise=None):
    """mass_noise = (seed, sd): every robot body's mass and inertia x N(1, sd) clipped to +-2 sd (PARAMETER noise); restored after
    the run. CACHED nominal quantities the controller keeps: the scaffold (Q, ID feedforward, PEL/COM offsets), total mass M, I_c,
    the joint tracker inertias ieff. MEASURED from the (perturbed) plant: the COM position/velocity (subtree_com, i.e. the controller
    sees the true COM: an optimistic state estimate), foot loads, attitude."""
    if mass_noise is not None:
        m = S["m"]; bm0 = m.body_mass.copy(); bi0 = m.body_inertia.copy(); rng = np.random.default_rng(mass_noise[0])
        f = np.clip(rng.normal(1.0, mass_noise[1], m.nbody), 1 - 2*mass_noise[1], 1 + 2*mass_noise[1])
        for b in S["robot_bodies"]: m.body_mass[b] *= f[b]; m.body_inertia[b] *= f[b]
        mujoco.mj_setConst(m, mujoco.MjData(m))
        try: return _rollout(S, R, p, n_cyc, turn, record, shove, noise, lag_plant, friction)
        finally:
            m.body_mass[:] = bm0; m.body_inertia[:] = bi0; mujoco.mj_setConst(m, mujoco.MjData(m))
    return _rollout(S, R, p, n_cyc, turn, record, shove, noise, lag_plant, friction)


def _rollout(S, R, p, n_cyc=None, turn=0.0, record=False, shove=None, noise=None, lag_plant=None, friction=None):
    """One walk on plant S. turn: total heading change (rad) over 5 strides after cycle 4. shove: (fx, fy, t0, dur) on the pelvis.
    lag_plant: the plant's actuator lag (the controller always assumes LAG). friction: floor sliding friction override."""
    m = S["m"]; d = mujoco.MjData(m); J = S["jarr"]; types = S["types"]; nT = len(types); dt = m.opt.timestep; ar = np.arange(nT)
    if n_cyc is None: n_cyc = N_TRANS + N_PER + 2
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    fl = S["floor"]; fric_g = [fl] + sorted(S["foot_geoms"]); fric0 = m.geom_friction[fric_g].copy()
    if friction is not None: m.geom_friction[fric_g, 0] = friction   # repair round 2: floor AND foot geoms (MuJoCo takes the max of a pair's mu; the floor alone was a no-op)
    tau_v = J["tau"]; om_v = J["om"]
    for i in range(nT):
        aid = J["aid"][i]; m.actuator_ctrlrange[aid][:] = [-tau_v[i], tau_v[i]]; m.actuator_forcerange[aid][:] = m.actuator_ctrlrange[aid]
        m.jnt_actfrcrange[S["tmap"][types[i]][0][0]][:] = m.actuator_ctrlrange[aid]
    q0, v0, ph0 = initial_state(S, R, p, noise); d.qpos[:] = q0; d.qvel[:] = v0; mujoco.mj_forward(m, d)
    for (a_, dof, qad, tau, kp_, kd_) in S["held"]: d.ctrl[a_] = q0[qad]
    N, res, ts, amp, lean = R["N"], R["res"], p["tscale"], p["amp"], p["lean"]; T, Tds, Tc = R["T"], R["Tds"], R["Tc"]; tsw = T - Tds
    Q, QD, ID, JI, REL, FT, SH, SWP, CR, VR, AR, PX, PY = (R[k] for k in ("Q", "QD", "ID", "JI", "REL", "FT", "SH", "SWP", "CR", "VR", "AR", "PX", "PY"))
    KN = SP.knot_table(S, p["x"], N); nadv = int(round(LAG/(res*ts)))
    gn = p.get("g", {}); KV = np.array([gn.get("kvx", K_V[0]), gn.get("kvy", K_V[1])]); OMAF = gn.get("oma", OMA_F); COPM = gn.get("copm", COP_MARGIN); TDD = gn.get("tdd", TD_DEPTH)
    OML = gn.get("oml", SP.OM_LEG); timed = gn.get("timed", TIMED); yawfull = gn.get("yawfull", YAW_FULL); AVL = gn.get("avail", AVAIL_LOAD); TDA = gn.get("td_accept", TD_ACCEPT)
    CLIPF = gn.get("clipf", 1.0); MINW = gn.get("minw", 0.5)
    ieff = R["ieff"]; omj = np.array([OML if any(k_ in t for k_ in ("hip", "knee", "ankle")) else SP.OM_UP for t in types]); kp = omj**2*ieff; kd = 2*omj*ieff
    legcols = [np.array(c) for c in R["legcols"]]; Mtot = R["M"]; Mg = Mtot*G; om = R["om"]
    K_XI = gn.get("kxi", K_XI_); om_z = min(1.0/SP.TC_LAG, 0.5/LAG); om_att = OMAF*om; Ka, Da = om_att**2, 2*om_att; Kz, Dz = om_z**2, 2*om_z
    lagp = LAG if lag_plant is None else lag_plant; alpha = 1.0 - np.exp(-dt/lagp); act = ID[int(ph0/res) % N]/tau_v
    ankb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in BI.ROBOTS[S["robot"]]["foot_bodies"]]   # [left, right]
    fb = [ankb[1], ankb[0]]                                                                                     # per leg li: 0 right, 1 left
    feet = S["feet"]; fgeo = [feet[1], feet[0]]
    cg = [g for g in range(m.ngeom) if m.geom_bodyid[g] in S["robot_bodies"] and g not in S["foot_geoms"] and m.geom_group[g] == 3] if S["robot"] == "G1" else []
    E0 = float(d.energy[0] + d.energy[1]); W_act = W_con = W_pas = 0.0; work = np.zeros(nT); overj = np.zeros(nT)
    T_end = n_cyc*Tc*ts; nst = int(round(T_end/dt))
    # clock and anchor
    phi = ph0; ev = 1                          # next touchdown event index: event k at phase k T (even = right, odd = left)
    def ev_leg(k): return 0 if k % 2 == 0 else 1
    def ref_foot(k): return np.array([k*amp*R["s"], -R["w"] if k % 2 == 0 else R["w"]])       # the touching foot's reference xy at event k (unwrapped cycle frame)
    anc = dict(f_act=d.xpos[fb[0]][:2].copy(), f_ref=ref_foot(0), psi=0.0)
    turn0 = 4*Tc; turn_T = 5*Tc
    def psi_ref(ph): return turn*np.clip((ph - turn0)/turn_T, 0.0, 1.0) if turn else 0.0
    def to_world(pxy, ph_unw):                                                                   # cycle-frame xy (unwrapped) -> world
        return anc["f_act"] + rotz(anc["psi"])[:2, :2] @ (pxy - anc["f_ref"])
    fell = False; t_fall = None; nonfoot = False; loaded = [True, False]; t_air = [0.0, 1.0]; slip0 = [d.xpos[fb[0]][:2].copy(), None]; slip_hold = [None, None]; slip_max = 0.0; slip_ankle = 0.0; slide = [np.zeros(2), np.zeros(2)]; slide_max = 0.0
    lo_pt = [None, np.array(list(to_world(np.array([-amp*R["s"], R["w"]]), 0.0)) + [R["sole"]["hf"]])]; lo_yaw = [0.0, 0.0]; plc = [np.zeros(2), np.zeros(2)]; tds = [[], []]; n_early = 0; n_late_hold = 0.0
    both_air = 0.0; flights = []; ds_t = 0.0; ds_steps = []; td_all = []; bounds = []; z0 = float(q0[2])
    trace = dict(t=[], qpos=[], com=[], vcom=[], load=[], cref=[], phi=[], tq=[], feet=[], tgt=[]) if record else None; tgt_dbg = [np.zeros(3), np.zeros(3)]
    shv = None; ff_ = [Mg, 0.0]; slip_log = []; n_ds_hold = 0.0; td_log = []
    if shove is not None: shv = (np.array([shove[0], shove[1], 0.0])*Mg, shove[2]*Tc*ts, shove[3])
    path_len = 0.0; c_prev = d.subtree_com[1][:2].copy()
    for st_ in range(nst):
        t = st_*dt
        # ---------------- clock
        ph_td = ev*T; ph_dse = (ev - 1)*T + Tds                      # next touchdown; end of the current double support
        if timed:
            phi += dt/ts
            if phi >= ph_td:                                       # TIMED EVENT: the scheduled touchdown re-anchors on the expected foot wherever it is
                li_e = ev_leg(ev); pxy_e = d.xpos[fb[li_e]][:2].copy()
                if record: mujoco.mj_subtreeVel(m, d); td_log.append(dict(t=round(t, 3), leg=li_e, load=round(ff_[li_e]/Mg, 2), foot_minus_com=np.round(pxy_e - d.subtree_com[1][:2], 3).tolist(), v=np.round(d.subtree_linvel[1][:2], 3).tolist(), plc=np.round(plc[li_e], 3).tolist()))
                anc = dict(f_act=pxy_e, f_ref=ref_foot(ev), psi=psi_ref(ev*T)); tds[li_e].append(t); td_all.append(t)
                if len(td_all) >= 2: ds_steps.append(ds_t)
                ds_t = 0.0; lo_pt[li_e] = None; ev += 1
                if li_e == 0:
                    mujoco.mj_subtreeVel(m, d); rr, pp, yy = rpy_mat(d.xmat[1].reshape(3, 3)); vh = rotz(yy).T @ d.subtree_linvel[1]
                    bounds.append(dict(t=t, z=float(d.qpos[2]), roll=float(rr), pitch=float(pp), yaw=float(yy), q=d.qpos[J["qad"]].copy(), qd=d.qvel[J["dof"]].copy(), vx=float(vh[0]), path=path_len, comx=d.subtree_com[1].copy()))
        elif phi < ph_td:
            phn = min(phi + dt/ts, ph_td)
            if phi <= ph_dse < phn and ff_[ev_leg(ev - 1)] < DS_LOAD*Mg: phn = ph_dse; n_ds_hold += dt     # SUPPORT CONTRACT: the rear foot is not unloaded before the new stance foot carries DS_LOAD of the weight
            phi = phn
        else: n_late_hold += dt
        holding = phi >= ph_td
        kc = int(np.floor(phi/Tc)); u = phi - kc*Tc; k0 = int(u/res) % N; kk = (k0 + (0 if holding else nadv)) % N
        kca = int(np.floor((phi + (0 if holding else nadv*res))/Tc))          # cycle index of the advanced reference sample
        mujoco.mj_subtreeVel(m, d); c = d.subtree_com[1].copy(); vc = d.subtree_linvel[1].copy()
        Rp = d.xmat[1].reshape(3, 3); r_, p_, y_ = rpy_mat(Rp); wW = Rp @ d.qvel[3:6]
        qn = d.qpos[J["qad"]].copy(); wv = d.qvel[J["dof"]].copy()
        # lag-aware: feedback on the state predicted one lag ahead
        qn_p = qn + wv*LAG; c_p = c + vc*LAG; att_p = np.array([r_, p_, y_]) + wW*LAG
        psi = psi_ref(phi); Rz = rotz(anc["psi"])
        cyc_off = np.array([kca*2*amp*R["s"], 0.0])
        cr_xy = to_world(np.array([CR[kk, 0]*amp, CR[kk, 1]]) + cyc_off, phi); vr = Rz[:2, :2] @ np.array([VR[kk, 0]*amp/ts, VR[kk, 1]/ts])
        zr = CR[kk, 2]; pr_xy = to_world(np.array([PX[kk]*amp, PY[kk]]) + cyc_off, phi)
        # ---------------- support force: DCM tracking through the CoP (p_cmd = p_ref + K_XI (xi - xi_ref)), LIPM force M om^2 (c - p_cmd)
        xi_e = (c_p[:2] + vc[:2]/om) - (cr_xy + vr/om); p_cmd = pr_xy + K_XI*xi_e
        att_err = np.array([att_p[0], att_p[1] - lean, wrap(att_p[2] - psi)]); Mc = R["I_c"]*(-Ka*att_err) - Da*R["I_c"]*wW
        mode = []; shares = np.zeros(2)
        for li in (0, 1):
            sh = SH[k0, li]; mode.append("stance" if (sh > 0 or SWP[k0, li] == 0) else "swing")
            if holding and li == ev_leg(ev): mode[li] = "swing"
            if timed and DS_LOAD > 0 and mode[li] == "swing" and ff_[li] > 0.05*Mg and ff_[1 - li] < DS_LOAD*Mg and SWP[k0, li] < 0.3:
                mode[li] = "stance"; sh = 1.0                          # SUPPORT CONTRACT (timed clock, opt-in): the rear foot stays a support until the front foot carries DS_LOAD
            shares[li] = sh if mode[li] == "stance" else 0.0
        tst = np.zeros(nT); gsc = np.ones(nT); avail = [min(1.0, ff_[li]/(AVL*Mg)) for li in (0, 1)]
        sa = shares*np.array(avail); tot = sa.sum()
        if tot > 1e-6: sa = sa/tot
        # CoP FEASIBILITY: the commanded CoP is clipped to what the loaded soles can produce (share-weighted sole centres +- the sole box,
        # heading frame); the support force is then the LIPM force from that CoP, M om^2 (c - p)
        sole = R["sole"]; xc_ = 0.5*(sole["toe"] + sole["heel"]); hx_ = COPM*0.5*(sole["toe"] - sole["heel"]); hy_ = COPM*sole["half_w"]
        pf_l = [d.xpos[fb[li]] - np.array([0, 0, sole["hf"]]) for li in (0, 1)]; yf_l = [rpy_mat(d.xmat[fb[li]].reshape(3, 3))[2] for li in (0, 1)]
        p_base = np.zeros(2); off = np.zeros(2)
        if tot > 1e-6:
            for li in (0, 1): p_base += sa[li]*(pf_l[li][:2] + rotz(yf_l[li])[:2, :2] @ np.array([xc_, 0.0]))
            off = Rz[:2, :2].T @ (p_cmd - p_base); off = np.array([min(max(off[0], -hx_), hx_), min(max(off[1], -hy_), hy_)]); p_cmd = p_base + Rz[:2, :2] @ off
        Fz_ = max(0.0, Mtot*(G + Kz*(zr - c_p[2]) + Dz*(0.0 - vc[2]))); hz = max(c_p[2], 0.2)
        # radial LIPM force from the CoP, TILTED for the attitude moment (hip strategy, as the sprint's stance): M_x = h dF_y, M_y = -h dF_x
        F = np.array([Mtot*om*om*(c_p[0] - p_cmd[0]) - Mc[1]/hz, Mtot*om*om*(c_p[1] - p_cmd[1]) + Mc[0]/hz, Fz_])
        fxy = np.hypot(F[0], F[1])
        if fxy > 0.8*F[2]: F[:2] *= 0.8*F[2]/max(fxy, 1e-9)
        if DBG2 and st_ % int(0.05/dt) == 0: print(f"t {t:.2f} phi {phi:.2f} vx {vc[0]:.3f} vref {vr[0]:.3f} c-cref {c_p[0]-cr_xy[0]:+.3f} xi_e {xi_e[0]:+.3f} Flipm {Mtot*om*om*(c_p[0]-p_cmd[0]):+.1f} Ftilt {-Mc[1]/hz:+.1f} pitch {att_p[1]:+.3f}")
        for li in (0, 1):
            if sa[li] <= 1e-6: continue
            pf = pf_l[li]; Fi = sa[li]*F; yf = yf_l[li]; Rf = rotz(yf)
            cop_i = pf + np.concatenate([Rf[:2, :2] @ np.array([xc_, 0.0]) + Rz[:2, :2] @ off, [0.0]])
            m_f = cross(cop_i - pf, Fi)                                       # the force acts at this foot's (feasible) CoP: exact
            # plus this foot's share of the attitude moment, inside what the sole has left: CoP shift (x, y) and TORSION <= TORSION_MU Fz
            ma = Rf.T @ (sa[li]*Mc); Fzl = max(Fi[2], 0.0); ma[0] = ma[1] = 0.0; ma[2] = min(max(ma[2], -TORSION_MU*Fzl), TORSION_MU*Fzl)   # pitch/roll via the tilt; yaw by torsion
            mi = m_f + Rf @ ma
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); mujoco.mj_jac(m, d, jp, jr, pf, fb[li]); cols = legcols[li]
            tst[cols] += -(jp[:, J["dof"][cols]].T @ Fi + jr[:, J["dof"][cols]].T @ mi); gsc[cols] = 1.0 - 0.9*min(1.0, avail[li])
            if yawfull: gsc[R["ysign"][("right" if li == 0 else "left") + "_i"]] = 1.0
            if HIP_FULL:
                for i_ in R["ikcols"][li][:2]: gsc[i_] = max(gsc[i_], HIP_FULL)
        # ---------------- joint reference: scaffold + task-space corrections through the scaffold's IK Jacobian
        qt = Q[kk].copy(); qdt = QD[kk]/ts
        pel_ref_xy = cr_xy - Rz[:2, :2] @ (np.array([CR[kk, 0], CR[kk, 1]]) - R["PEL"][kk][:2])        # pelvis = COM ref - the posture's COM offset
        pel_ref = np.array([pel_ref_xy[0], pel_ref_xy[1], R["PEL"][kk][2]])
        pel_act = d.xpos[1] + d.qvel[0:3]*LAG
        Ract = Rp
        dxy = (c_p[:2] - cr_xy) + Rz[:2, :2] @ (KV*(Rz[:2, :2].T @ (vc[:2] - vr)))/om                                                  # foothold correction: COM offset + K_XI x the velocity part of the DCM error
        dxy_h = Rz[:2, :2].T @ dxy; dxy_h = np.array([np.clip(dxy_h[0], -0.15*CLIPF, 0.15*CLIPF), np.clip(dxy_h[1], -0.10*CLIPF, 0.10*CLIPF)])
        for li in (0, 1):
            sd = "right" if li == 0 else "left"; JIk = JI[li][kk]; nq = JIk.shape[0]; cols = R["ikcols"][li]
            yi = R["ysign"][sd + "_i"]
            if mode[li] == "stance":
                fpos = d.xpos[fb[li]].copy()
                if timed and ff_[li] < 0.05*Mg: fpos[2] = R["sole"]["hf"] - TDD          # a scheduled stance foot not yet loaded reaches down
                rel_des = Rz.T @ (fpos - pel_ref)
                dori = [-lean, 0.0]
                yaw_f = rpy_mat(d.xmat[fb[li]].reshape(3, 3))[2]; qt[yi] += R["ysign"][sd]*wrap(yaw_f - psi)
            else:
                if lo_pt[li] is None: lo_pt[li] = d.xpos[fb[li]].copy()
                k_ev = ev if ev_leg(ev) == li else ev + 1                                          # this swing ends at event k_ev
                swp = float(np.clip((phi + (0 if holding else nadv*res) - ((k_ev - 1)*T + Tds))/tsw, 0.0, 1.0))
                if swp < 0.8:
                    plc[li] = dxy_h.copy()
                    if PROJ:
                        t_rem = max(0.0, (1.0 - swp)*tsw*ts); xe = Rz[:2, :2].T @ xi_e*np.exp(om*t_rem); plc[li] = np.array([min(max(xe[0], -0.15), 0.15), min(max(xe[1], -0.10), 0.10)])
                land_c = ref_foot(k_ev).astype(float); land_c[0] += 0.0
                land = to_world(land_c, phi) + Rz[:2, :2] @ plc[li]
                other = d.xpos[fb[1 - li]][:2]; lat = Rz[:2, :2].T @ (land - other); sgn_l = -1.0 if li == 0 else 1.0
                if sgn_l*lat[1] < MINW*2*R["w"]: lat[1] = sgn_l*MINW*2*R["w"]; land = other + Rz[:2, :2] @ lat
                f_ = SP.mj_(swp); tgt = np.array([lo_pt[li][0] + (land[0] - lo_pt[li][0])*f_, lo_pt[li][1] + (land[1] - lo_pt[li][1])*f_,
                                                  R["sole"]["hf"] + CLEAR*64*swp**3*(1 - swp)**3 - TDD*SP.mj_((swp - TD_FROM)/(1.0 - TD_FROM))])
                if holding: tgt[2] = R["sole"]["hf"] - TDD - 0.02*min(1.0, n_late_hold/0.1)          # waiting for touchdown: reach down
                rel_des = Ract.T @ (tgt - pel_act); tgt_dbg[li] = tgt
                dori = [-att_p[1], -att_p[0]]
                qt[yi] += R["ysign"][sd]*wrap(psi_ref(k_ev*T) - att_p[2])
            e = np.concatenate([rel_des - REL[kk, li]*np.array([1.0, 1.0, 1.0]), dori[:nq - 3]])
            e[:3] = np.clip(e[:3], -0.25, 0.25)
            dq = JIk @ e; qt[cols] += dq
            if DBG and li == 1 and st_ % int(0.01/dt) == 0 and 0.10 < t < 0.30: print(f"t {t:.3f} {mode[li]} e {np.round(e, 3)} dq {np.round(dq, 3)} sa {sa[li]:.2f} qt-q {np.round((qt - qn)[legcols[li]], 3)} tst {np.round(tst[legcols[li]], 1)} act {np.round((act*tau_v)[legcols[li]], 1)} F {np.round(F, 1)} Mc {np.round(Mc, 1)} wz {wW[2]:.2f}")
        idr = ID[kk].copy()
        if holding: idr[legcols[ev_leg(ev)]] = 0.0
        cmd = (idr + tst + gsc*(kp*(qt - qn_p) + kd*(qdt - wv)))/tau_v + KN[kk, ar]
        act += alpha*(cmd - act)
        frac = np.maximum(0.0, 1.0 - np.abs(wv)/om_v); hi = np.where(wv > 0, tau_v*frac, tau_v); lo = np.where(wv < 0, tau_v*frac, tau_v)
        tq = np.clip(act*tau_v, -lo, hi); overj = np.maximum(overj, np.abs(wv)/om_v)
        d.ctrl[J["aid"]] = tq; BI.backemf_brake(m, d, S)
        if shv is not None:
            d.xfrc_applied[1, :3] = shv[0] if shv[1] <= t < shv[1] + shv[2] else 0.0
        qv0 = d.qvel.copy(); mujoco.mj_step(m, d); qm = 0.5*(qv0 + d.qvel)
        W_act += float(np.dot(d.actuator_force, qm[S["act_dof"]]))*dt; W_con += float(np.dot(d.qfrc_constraint, qm))*dt
        W_pas += (float(np.dot(d.qfrc_passive, qm)) + float(np.dot(d.qfrc_applied, qm)))*dt
        pw = tq*d.qvel[J["dof"]]; work += np.where(pw > 0, pw, 0.0)*dt
        ff = [0.0, 0.0]; cop = [np.zeros(2), np.zeros(2)]   # per leg li
        for ci in range(d.ncon):
            cn = d.contact[ci]
            if fl not in (cn.geom1, cn.geom2): continue
            fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv); o = cn.geom2 if cn.geom1 == fl else cn.geom1
            if o in fgeo[0]: ff[0] += fv[0]; cop[0] += fv[0]*cn.pos[:2]
            elif o in fgeo[1]: ff[1] += fv[0]; cop[1] += fv[0]*cn.pos[:2]
            elif o in S["robot_geoms"]: nonfoot = True
        ff_ = ff
        if DBG and st_ % int(0.005/dt) == 0 and 0.27 < t < 0.20:
            Fw = np.zeros(3); Tw = np.zeros(3); nc = 0
            for ci in range(d.ncon):
                cn = d.contact[ci]; o = cn.geom2 if cn.geom1 == fl else cn.geom1
                if fl in (cn.geom1, cn.geom2) and o in fgeo[1]:
                    fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv); fr = cn.frame.reshape(3, 3).T @ fv[:3]
                    if cn.geom1 != fl: fr = -fr
                    Fw += fr; Tw += cross(cn.pos - d.xpos[fb[1]], fr); nc += 1
            print(f"   CONTACT L t {t:.3f} n {nc} F {np.round(Fw, 1)} T@ankle {np.round(Tw, 1)} cmdF {np.round(F*sa[1], 1)} foot_v {np.round(d.cvel[fb[1]][3:] if False else np.zeros(3), 2)} wz_foot {d.cvel[fb[1]][2]:.2f}")
        if cg and st_ % 10 == 0 and min(P.geom_low(m, d, g_) for g_ in cg) < -0.002: nonfoot = True
        c_now = d.subtree_com[1][:2]; path_len += float(np.linalg.norm(c_now - c_prev)); c_prev = c_now.copy()
        for li in (0, 1):
            on = ff[li] > (0.02 if li == ev_leg(ev) else 0.05)*Mg; pxy = d.xpos[fb[li]][:2].copy()      # the expected foot's touchdown threshold is 2 % of body weight
            if not on: t_air[li] += dt
            if on and not loaded[li]:
                fresh = t_air[li] >= FLIGHT_CONFIRM - 1e-12
                if fresh:
                    swp_now = (phi - (ev - 1)*T - Tds)/tsw if True else 0.0
                    slide[li] = np.zeros(2)                                                             # a fresh (>= 20 ms airborne) touchdown starts a new stance
                    if timed: slip0[li] = pxy.copy()
                    elif li == ev_leg(ev) and (holding or swp_now >= TDA):
                        # TOUCHDOWN EVENT: phase jumps to the event, reference re-anchored on this foot
                        if record: mujoco.mj_subtreeVel(m, d); td_log.append(dict(t=round(t, 3), leg=li, phi_before=round(phi, 3), late=round(n_late_hold, 3), foot_minus_com=np.round(pxy - d.subtree_com[1][:2], 3).tolist(), v=np.round(d.subtree_linvel[1][:2], 3).tolist(), plc=np.round(plc[li], 3).tolist()))
                        phi = ev*T; anc = dict(f_act=pxy.copy(), f_ref=ref_foot(ev), psi=psi_ref(ev*T)); tds[li].append(t); td_all.append(t)
                        if len(td_all) >= 2: ds_steps.append(ds_t)
                        ds_t = 0.0; slip0[li] = pxy.copy(); slide[li] = np.zeros(2); lo_pt[li] = None; ev += 1; n_late_hold = 0.0
                        if li == 0:
                            mujoco.mj_subtreeVel(m, d); rr, pp, yy = rpy_mat(d.xmat[1].reshape(3, 3)); vh = rotz(yy).T @ d.subtree_linvel[1]
                            bounds.append(dict(t=t, z=float(d.qpos[2]), roll=float(rr), pitch=float(pp), yaw=float(yy), q=d.qpos[J["qad"]].copy(), qd=d.qvel[J["dof"]].copy(), vx=float(vh[0]), path=path_len, comx=d.subtree_com[1].copy()))
                    else: n_early += 1; slip0[li] = pxy.copy()
                elif slip_hold[li] is not None: slip0[li] = slip_hold[li]                         # load blip < 20 ms: the same stance, the anchor is kept
                t_air[li] = 0.0
            if on:
                pc = cop[li]/max(ff[li], 1e-9); ov = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, fb[li], ov, 0)
                vp = ov[3:5] + cross(ov[:3], np.array([pc[0], pc[1], 0.0]) - d.xpos[fb[li]])[:2]; slide[li] += vp*dt
                slide_max = max(slide_max, float(np.linalg.norm(slide[li])))
            if on and slip0[li] is not None and mode[li] == "stance":
                sl_ = float(np.linalg.norm(pxy - slip0[li])); slip_ankle = max(slip_ankle, sl_)
                if record and sl_ > SLIP_MAX and not slip_log or (record and sl_ > SLIP_MAX and slip_log[-1][0] != len(td_all)): slip_log.append((len(td_all), li, round(t, 3), round(sl_, 3), round(phi, 3)))
            if not on and loaded[li]: lo_pt[li] = d.xpos[fb[li]].copy(); slip_hold[li] = slip0[li]; slip0[li] = None
            loaded[li] = on
        if ff[0] > 0.02*Mg and ff[1] > 0.02*Mg: ds_t += dt
        if ff[0] < 1.0 and ff[1] < 1.0: both_air += dt
        elif both_air > 0: flights.append((t, both_air)); both_air = 0.0
        if record and st_ % 4 == 0:
            trace["t"].append(t); trace["qpos"].append(d.qpos.copy()); trace["com"].append(d.subtree_com[1].copy()); trace["vcom"].append(vc.copy()); trace["load"].append(np.array(ff)/Mg)
            trace["cref"].append(np.array([cr_xy[0], cr_xy[1], zr])); trace["phi"].append(phi); trace["tq"].append(tq.copy())
            trace["feet"].append(np.concatenate([d.xpos[fb[0]], d.xpos[fb[1]]])); trace["tgt"].append(np.concatenate(tgt_dbg))
        rr, pp, _ = rpy_mat(d.xmat[1].reshape(3, 3))
        if d.qpos[2] < 0.6*z0 or abs(rr) > 0.7 or abs(pp - lean) > 0.7 or not np.isfinite(d.qpos[2]):
            fell = True; t_fall = t; break
    if friction is not None: m.geom_friction[fric_g] = fric0
    slip_max = slide_max                                                    # SLIP GATE: sliding of the sole point under the CoP (rolling is not slip)
    E1 = float(d.energy[0] + d.energy[1]); closure = (E1 - E0) - W_act - W_con - W_pas; ctol = max(5.0, 0.02*(abs(W_act) + abs(W_con)))
    if shv is not None: ctol = np.inf                                                       # the external shove's work is not tracked
    t_run = t_fall if fell else nst*dt
    nb = len(bounds); per = None; per_n = 0; v_avg = 0.0; v_path = 0.0; errs = []
    if nb >= N_TRANS + 2:
        for a, b in zip(bounds[N_TRANS:-1], bounds[N_TRANS + 1:]):
            errs.append(dict(z=abs(b["z"] - a["z"])/PER_TOL["z"], pitch=abs(b["pitch"] - a["pitch"])/PER_TOL["pitch"], roll=abs(b["roll"] - a["roll"])/PER_TOL["roll"],
                             q=float(np.sqrt(np.mean((b["q"] - a["q"])**2)))/PER_TOL["q"], qd=float(np.sqrt(np.mean((b["qd"] - a["qd"])**2)))/PER_TOL["qd"], vx=abs(b["vx"] - a["vx"])/PER_TOL["vx"]))
        run_ = 0
        for e in errs:
            run_ = run_ + 1 if max(e.values()) <= 1.0 else 0; per_n = max(per_n, run_)
        per = max(max(e.values()) for e in errs)
        v_avg = _chord(bounds[N_TRANS:])/(bounds[-1]["t"] - bounds[N_TRANS]["t"])
        v_path = (bounds[-1]["path"] - bounds[N_TRANS]["path"])/(bounds[-1]["t"] - bounds[N_TRANS]["t"])
    elif nb >= 2:
        v_avg = _chord(bounds)/max(bounds[-1]["t"] - bounds[0]["t"], 1e-9); v_path = (bounds[-1]["path"] - bounds[0]["path"])/max(bounds[-1]["t"] - bounds[0]["t"], 1e-9)
    t_tr = bounds[N_TRANS]["t"] if nb > N_TRANS else None
    fl_after = [f for (te, f) in flights if t_tr is not None and te > t_tr]
    flight_max = float(max(fl_after)) if fl_after else 0.0
    ds_after = [x for x, te in zip(ds_steps, td_all[1:]) if t_tr is not None and te > t_tr]
    ds_min = float(min(ds_after)) if ds_after else 0.0
    overspeed = float(overj.max()); reasons = []; v_des = R["v"]
    if fell: reasons.append("FELL")
    if overspeed > BI.OVERSPEED_TOL: reasons.append("OVERSPEED")
    if nonfoot: reasons.append("NONFOOT_CONTACT")
    if abs(closure) > ctol: reasons.append("ENERGY_CLOSURE")
    if slip_max > SLIP_MAX: reasons.append("SLIP")
    if not fell and flight_max >= FLIGHT_CONFIRM: reasons.append("FLIGHT")
    if not fell and ds_after and ds_min < FLIGHT_CONFIRM: reasons.append("NO_DOUBLE_SUPPORT")
    yaw_end = bounds[-1]["yaw"] if bounds else 0.0
    if not fell and abs(wrap(yaw_end - turn)) > HEADING_TOL: reasons.append("HEADING")
    periodic = per_n >= N_PER
    speed_ok = abs(v_avg - v_des) <= SPEED_TOL*v_des
    viol = 0.0
    if fell: viol += 5.0 + 10.0*(1 - t_run/T_end)
    if "OVERSPEED" in reasons: viol += 1.0 + 5.0*(overspeed - BI.OVERSPEED_TOL)
    if nonfoot: viol += 3.0
    if "ENERGY_CLOSURE" in reasons: viol += 1.0 + min(abs(closure)/ctol, 5.0)
    if "SLIP" in reasons: viol += 1.0 + 20.0*(slip_max - SLIP_MAX)
    if "FLIGHT" in reasons: viol += 1.0 + 10.0*flight_max
    if "NO_DOUBLE_SUPPORT" in reasons: viol += 1.0 + 10.0*max(0.0, FLIGHT_CONFIRM - ds_min)/FLIGHT_CONFIRM
    if "HEADING" in reasons: viol += 1.0 + 5.0*(abs(wrap(yaw_end - turn)) - HEADING_TOL)
    dist = _chord(bounds) if nb > 1 else 0.0                                   # net distance (stride-sampled COM chords), not path length
    steps_T = np.diff(td_all[2*N_TRANS:]) if len(td_all) > 2*N_TRANS + 2 else np.diff(td_all) if len(td_all) > 2 else np.array([])
    step_T = float(np.median(steps_T)) if len(steps_T) else None
    out = dict(value=float(v_avg), valid=(not reasons) and periodic and speed_ok, reasons=reasons, violation=viol, periodic=bool(periodic), per_err=None if per is None else float(per), per_n=int(per_n),
               speed_ok=bool(speed_ok), t_run=float(t_run), fell=bool(fell), overspeed=overspeed, closure=float(closure), slip_max=float(slip_max), yaw_end=float(yaw_end),
               n_td=[len(tds[0]), len(tds[1])], n_early=n_early, n_cycles=nb, step_T=step_T, cadence_spm=(60.0/step_T if step_T else None), step_len=(float(v_avg*step_T) if step_T else None),
               v_path=float(v_path), slip_ankle=float(slip_ankle), ds_min=ds_min, ds_hold=float(n_ds_hold), td_hold=float(n_late_hold), ds_med=(float(np.median(ds_after)) if ds_after else 0.0), ds_frac=(float(np.median(ds_after))/step_T if (ds_after and step_T) else None), flight_max=flight_max,
               duty=(0.5*(1 + float(np.median(ds_after))/step_T) if (ds_after and step_T) else None),
               W_pos=float(work.sum()), J_per_m=(float(work.sum())/max(dist, 1e-3) if nb > 1 else None), cot=(float(work.sum())/(Mg*max(dist, 1e-3)) if nb > 1 else None), dist=float(dist), dt=dt)
    if record:
        out["td_log"] = td_log; out["slip_log"] = slip_log; out["td_all"] = td_all; out["trace"] = {k: np.asarray(v).tolist() for k, v in trace.items()}; out["bounds"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in b.items()} for b in bounds]
    return out


N_GAIN = 13
def gains(g):
    """controller-gain vector g in [-1, 1]^N_GAIN -> the walking controller's constants (g = 0: the construction defaults). Shorter
    vectors (earlier searches) are padded with zeros (= defaults)."""
    g = np.concatenate([np.asarray(g, float), np.zeros(max(0, N_GAIN - len(g)))])
    return dict(kvx=1.0*2**g[0], kvy=1.5*2**g[1], kxi=2.0*2**g[2], oma=2.0*2**g[3], copm=0.7 + 0.2*g[4], tdd=0.02*2**g[5], oml=SP.OM_LEG*2**(0.5*g[6]), timed=bool(g[7] > 0), yawfull=bool(g[8] > 0),
                avail=0.15*2**(2*g[9]), td_accept=0.75 + 0.15*g[10], clipf=2**g[11], minw=0.5 + 0.3*g[12])


def unpack(S, z):
    """sprint's parameter vector (+ the controller-gain block); the walk's lean is centred on 0 (upright scaffold): lean = 0.25 z."""
    z = np.asarray(z, float); zs = z[:SP.n_params(S)]; p = SP.unpack(S, zs); p["lean"] = 0.25*float(zs[-1])
    if len(z) > len(zs): p["g"] = gains(z[len(zs):])
    return p


DBG = os.environ.get("WALK_DBG") == "1"
COP_MARGIN = float(os.environ.get("WALK_COPM", 0.7))                  # the commanded CoP stays inside 70 % of the sole's extent about its centre (no rocking onto an edge)
K_V = np.array([float(os.environ.get("WALK_KVX", 1.0)), float(os.environ.get("WALK_KVY", 1.5))])   # foothold velocity gains (x (v - v_ref)/omega), sagittal / lateral (heading frame)
K_XI_ = float(os.environ.get("WALK_KXI", 2.0))        # CoP DCM gain
OMA_F = float(os.environ.get("WALK_OMA", 2.0))        # attitude bandwidth = OMA_F x omega_LIPM
TIMED = os.environ.get("WALK_TIMED", "0") == "1"      # default clock mode (the gain block can switch it per search)
YAW_FULL = os.environ.get("WALK_YAWFULL", "0") == "1"  # default: stance hip-yaw PD scaled like the rest of the stance leg
PROJ = os.environ.get("WALK_PROJ", "0") == "1"        # opt-in: foothold = DCM error projected to touchdown (tested on G1, not adopted)
HIP_FULL = float(os.environ.get("WALK_HIPFULL", 0.0)) # opt-in: stance hip pitch/roll PD floor (tested on G1, not adopted)
DBG2 = os.environ.get("WALK_DBG2") == "1"
AVAIL_LOAD = float(os.environ.get("WALK_AVAIL", 0.15))   # a stance foot takes its full reference share of the support force once it carries this fraction of the weight
SPEED_TRIM = os.environ.get("WALK_TRIM", "1") == "1"   # repair round 1: net-speed trim of the warm start before CMA-ES (see search)
ROBUST_SEARCH = [x for x in os.environ.get("WALK_ROBUST", "").split(",") if x]   # perturbations scored inside the search (opt-in)
TORSION_MU = 0.02                  # attitude yaw moment through a foot <= 0.02 m x its normal force (the soles' narrow contact patterns; G1 heel is one central point)
TD_DEPTH = float(os.environ.get("WALK_TDD", 0.02))
TD_FROM = float(os.environ.get("WALK_TDF", 0.7))      # the touchdown descent starts at this swing fraction
TD_DEPTH_OLD = 0.01                    # the swing target ends 1 cm below the floor (from 80 % of the swing) so the foot lands with a pressing force
DS_LOAD = float(os.environ.get("WALK_DSLOAD", 0.0))    # opt-in (tested at 0.3, not adopted): the double support ends only once the new stance foot carries DS_LOAD of the body weight
TD_ACCEPT = float(os.environ.get("WALK_TDA", 0.75))                   # an expected foot landing after 80 % of its swing is a (slightly early) touchdown; earlier = scuff


def score(r, v_des):
    if r["reasons"]: return 100.0 + r["violation"] - 0.01*min(r["t_run"], 100.0)
    short = abs(r["value"] - v_des)/v_des
    if not r["periodic"] or short > SPEED_TOL:
        return 10.0 + min(r["per_err"] if r["per_err"] is not None else 50.0, 50.0)/10.0 + 20.0*max(0.0, short - SPEED_TOL) + 0.5*(1 - r["per_n"]/N_PER)
    c = max(r["cot"] or 0.0, 0.0); return -1.0 + c/(1.0 + c)         # valid: lower energy per metre is better (in (-1, 0))


# ====================================================================== problem object
class Walk:
    def __init__(self, robot, v, turn=0.0, stride=None):
        self.robot, self.v, self.turn = robot, float(v), float(turn)
        self.B = walk_body(robot)
        self.st = stride or reach_margin(self.B, preferred(self.B, self.v) or least_utilised(self.B, self.v))
        if self.st is None: raise ValueError(f"no feasible walking stride at {v} m/s")
        self.S = {dt: SP.plant(robot, dt) for dt in (core.DT_OPT, core.DT_CHECK)}
        self.nominal = {dt: nominal_pose(S_) for dt, S_ in self.S.items()}
        self.R = cached_scaffold(self.S[core.DT_OPT], self.B, self.st)
    def unpack(self, z): return unpack(self.S[core.DT_OPT], z)
    def run(self, z, dt=core.DT_OPT, **kw): return rollout(self.S[dt], self.R, self.unpack(z), turn=self.turn, **kw)
    def robust(self, z):
        r1 = self.run(z, core.DT_OPT)
        if not ROBUST_SEARCH:
            if r1["reasons"]: return score(r1, self.v), r1, r1          # the 1 ms score already fails a gate: the 0.5 ms run is skipped (search only)
            r2 = self.run(z, core.DT_CHECK); return max(score(r1, self.v), score(r2, self.v)), r1, r2
        # ROBUSTNESS IN THE SCORE (repair round 2, amendment C): lexicographic classes, every draw run once the nominal is gate-clean
        #   nominal fails a gate                          -> 300 + its violation            (worst)
        #   nominal clean, n draws fail a gate            -> 100 + 20 n + mean violation of the failing draws (fewer failures rank higher)
        #   every run gate-clean, some off band/periodic  -> 10..50 (worst draw's score) + 1
        #   every run valid at 1 ms                       -> the 0.5 ms nominal decides, max(s1, s2) (< 0 when valid)
        # (round 1 / amendment B returned at the first failing draw, so a nominal gate failure ranked ABOVE a nominal-valid gait that
        # failed one draw; the search parked on nominal double-support failures)
        if r1["reasons"]: return 300.0 + r1["violation"], r1, r1
        worst = score(r1, self.v); wr = r1; fails = []
        for nm in ROBUST_SEARCH:
            assert nm not in HELD_OUT and nm not in FRESH, "held-out perturbations are never scored in a search"
            kw = _pert_kw(nm, len(self.S[core.DT_OPT]["types"]), 1)
            rp = self.run(z, core.DT_OPT, **kw); sp = score(rp, self.v)
            if rp["reasons"]: fails.append(rp); continue
            if sp > worst: worst, wr = sp, rp
        if fails: return 100.0 + 20.0*len(fails) + float(np.mean([f["violation"] for f in fails])), r1, max(fails, key=lambda f: f["violation"])
        if worst >= 0: return worst + 1.0, r1, wr
        r2 = self.run(z, core.DT_CHECK); return max(score(r1, self.v), score(r2, self.v)), r1, r2
    def n(self): return SP.n_params(self.S[core.DT_OPT]) + N_GAIN


KEYS = ("value", "v_path", "valid", "reasons", "periodic", "per_err", "per_n", "speed_ok", "t_run", "overspeed", "closure", "slip_max", "slip_ankle", "yaw_end", "n_td", "n_early", "cadence_spm", "step_len", "ds_min", "ds_frac", "duty", "flight_max", "J_per_m", "cot", "dt")
def summarize(r): return {k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in KEYS}


def stride_rec(st): return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items() if not isinstance(v, np.ndarray)}


_W = {}
def _init(robot, v, turn, stride): _W["P"] = Walk(robot, v, turn, stride)
def _eval(z): s, r1, r2 = _W["P"].robust(z); return s, summarize(r1), summarize(r2)


def search(robot, v, iters, pop, workers, seed, turn=0.0, z0=None, stride=None, sigma=0.1, tag=""):
    import cma, multiprocessing as mp
    assert seed >= 1, "pycma seed 0 is random"
    Pb = Walk(robot, v, turn, stride); st = Pb.st; z0 = np.zeros(Pb.n()) if z0 is None else np.asarray(z0, float)
    if len(z0) < Pb.n(): z0 = np.concatenate([z0, np.zeros(Pb.n() - len(z0))])
    out_dir = f"{RES}/{robot}"; os.makedirs(out_dir, exist_ok=True); name = f"{out_dir}/search_v{v}{'_turn%d' % round(np.degrees(turn)) if turn else ''}{tag}"
    lg = open(name + ".log", "a"); ctx = mp.get_context("fork"); pool = ctx.Pool(workers, initializer=_init, initargs=(robot, v, turn, st))
    t0 = time.time(); s0, r1, r2 = Pb.robust(z0); best = (s0, z0.copy(), summarize(r1), summarize(r2)); hist = [(0, float(s0))]; last = 0
    lg.write(f"prior score {s0:.4f} r1 {summarize(r1)} r2 {summarize(r2)}\n"); lg.flush()
    trim = []
    if SPEED_TRIM:
        # SPEED TRIM (repair round 1, general outer loop): a gait that is periodic and gate-clean but off the NET-speed band has its
        # step-length amplitude rescaled by v_cmd / v_net (fixed point, <= 3 passes, kept only if the score improves); then CMA-ES.
        nS_ = SP.n_params(Pb.S[core.DT_OPT]); zt = z0.copy()
        for _ in range(3):
            rt = Pb.run(zt, core.DT_OPT)
            if rt["reasons"] or not rt["periodic"] or rt["value"] <= 0 or abs(rt["value"] - v)/v <= 0.5*SPEED_TOL: break
            amp = 1.0 + 0.2*zt[nS_ - 3]; zt = zt.copy(); zt[nS_ - 3] = float(np.clip((amp*v/rt["value"] - 1.0)/0.2, -1, 1))
            st_, a1, a2 = Pb.robust(zt); trim.append(dict(amp=round(float(1.0 + 0.2*zt[nS_ - 3]), 4), v_before=round(rt["value"], 4), score=round(float(st_), 4)))
            lg.write(f"trim amp {1.0 + 0.2*zt[nS_ - 3]:.4f} (v_net was {rt['value']:.4f}) score {st_:.4f} v {a1['value']:.3f}/{a2['value']:.3f} {a1['reasons']}/{a2['reasons']}\n"); lg.flush()
            if st_ < best[0]: best = (st_, zt.copy(), summarize(a1), summarize(a2)); z0 = zt.copy()
    nS = SP.n_params(Pb.S[core.DT_OPT]); active = (list(range(nS)) if os.environ.get("WALK_KNOTS", "0") == "1" else list(range(nS - 3, nS))) + list(range(nS, nS + N_GAIN))
    def full(x): zz = z0.copy(); zz[active] = x; return zz
    es = cma.CMAEvolutionStrategy(z0[active], float(os.environ.get("WALK_SIGMA", sigma)), {"bounds": [-1, 1], "seed": seed, "verbose": -9, "maxiter": iters, "popsize": pop})
    try:
        while not es.stop():
            X = es.ask(); Z = [full(x) for x in X]; res_ = pool.map(_eval, Z); Fv = [o[0] for o in res_]; es.tell(X, Fv); k = int(np.argmin(Fv))
            if Fv[k] < best[0]: best = (Fv[k], np.array(Z[k]), res_[k][1], res_[k][2]); last = es.countiter
            hist.append((es.countiter, float(best[0])))
            lg.write(f"it {es.countiter} best {best[0]:.4f} v {best[2]['value']:.3f}/{best[3]['value']:.3f} {best[2]['reasons']}/{best[3]['reasons']} per {best[2]['per_n']}/{best[3]['per_n']} cot {best[2]['cot']} | {time.time() - t0:.0f}s\n"); lg.flush()
            json.dump(dict(robot=robot, v=v, turn=turn, seed=seed, score=float(best[0]), z=best[1].tolist(), r1=best[2], r2=best[3], stride=stride_rec(st), hist=hist, secs=time.time() - t0, evals=int(es.countevals), done=False, supplied=SUPPLIED),
                      open(name + ".json", "w"), indent=1, default=float)
            if best[0] < 0 and es.countiter - last >= 10: break
    finally:
        pool.close(); pool.join()
    import subprocess
    res = dict(git=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip(), robot=robot, v=v, turn=turn, seed=seed, score=float(best[0]), z=best[1].tolist(), r1=best[2], r2=best[3], stride=stride_rec(st), hist=hist, trim=trim, secs=time.time() - t0, evals=int(es.countevals), done=True,
               params={k: (x if isinstance(x, dict) else float(x)) for k, x in Pb.unpack(best[1]).items() if k != "x"}, supplied=SUPPLIED)
    json.dump(res, open(name + ".json", "w"), indent=1, default=float); return res


# ====================================================================== validation (fresh plants, perturbations)
PERTURB = {
    "shove_lat": dict(shove=(0.0, 0.1, 4.5, 0.2), n_cyc=N_TRANS + N_PER + 8),     # 0.1 body weight lateral, 0.2 s, mid cycle 4 (mid-walk); then >= 10 periodic strides needed
    "shove_back": dict(shove=(-0.1, 0.0, 4.5, 0.2), n_cyc=N_TRANS + N_PER + 8),   # 0.1 body weight backward
    "start_noise": dict(noise="start"),                     # joint angles N(0, 0.02 rad), start vx +0.05 m/s (pre-set; 10 x the project's Q3 convention)
    "start_noise_2mrad": dict(noise="start", q_sd=0.002),   # the project's Q3 convention: joint angles N(0, 2 mrad), start vx +0.05 m/s
    "lag_fric": dict(lag_plant=1.25*LAG, friction=0.7),     # plant lag +25 % (controller assumes LAG), floor mu 0.7
    # search-only parameter-noise draw (repair round 1): masses/inertias x N(1, 0.05) seed 7, plant lag -20 %; seed differs from the held-out N3
    "param_noise": dict(mass_noise=(7, 0.05), lag_plant=0.8*LAG, friction=0.8),
    "shove_diag": dict(shove=(-0.07, 0.07, 6.25, 0.2), n_cyc=N_TRANS + N_PER + 8),   # search-only diagonal shove (mirror of the held-out N4 in direction and time)
    # HELD-OUT set (the physics verifier's N1-N4, never scored in any search): the robustness label uses this set
    "N1": dict(noise="start", seed=101, vx=-0.05),
    "N2": dict(noise="start", seed=202, vx=0.05, lag_plant=1.2*LAG, friction=0.8),
    "N3": dict(mass_noise=(303, 0.05), lag_plant=0.8*LAG, friction=0.8),
    "N4": dict(shove=(0.07, -0.07, 5.25, 0.2), n_cyc=N_TRANS + N_PER + 8),
}
HELD_OUT = ["N1", "N2", "N3", "N4"]
# REPAIR ROUND 2 search draws (general recipe, every gait alike; seeds/phases drawn once from rng seed 31, see PREDICTION_R2.md):
# three mass/inertia draws at nominal lag and two lateral shoves at drawn phases/directions
PERTURB.update({
    "mass_a": dict(mass_noise=(21, 0.05)), "mass_b": dict(mass_noise=(22, 0.05)), "mass_c": dict(mass_noise=(23, 0.05)),
    "lat_a": dict(shove=(0.0, 0.1, 5.31, 0.2), n_cyc=N_TRANS + N_PER + 8), "lat_b": dict(shove=(0.0, 0.1, 3.64, 0.2), n_cyc=N_TRANS + N_PER + 8),
    # FRESH HELD-OUT set F1-F4 (registered in PREDICTION_R2.md before any repair-2 run; run ONCE on the final gaits; never scored or
    # consulted between searches). Friction is now a real override (floor + feet).
    "F1": dict(noise="start", seed=9001, vx=0.04, lag_plant=1.15*LAG),
    "F2": dict(mass_noise=(9002, 0.05)),
    "F3": dict(noise="start", seed=9003, vx=-0.04, friction=0.7),
    "F4": dict(shove=(0.0, 0.1, 4.90, 0.2), n_cyc=N_TRANS + N_PER + 8),
    # verifier round-1 perturbations N5-N8 (reported, not a label set any more: consulted)
    "N5": dict(noise="start", seed=505, vx=0.03), "N6": dict(noise="start", seed=606, vx=-0.03, lag_plant=1.1*LAG, friction=0.85),
    "N7": dict(mass_noise=(707, 0.05)), "N8": dict(shove=(0.0, -0.1, 3.75, 0.2), n_cyc=N_TRANS + N_PER + 8),
})
FRESH = ["F1", "F2", "F3", "F4"]
SEARCH_ONLY = ("param_noise", "shove_diag", "mass_a", "mass_b", "mass_c", "lat_a", "lat_b")


SWEEP = ["SY-3.75", "SY-4.0", "SY-4.25", "SY-4.5", "SY-4.75", "SY-5.0", "SY+3.75", "SY+4.25", "SY+4.5"]   # verifier's lateral-shove phase sweep (diagnostic, 1 ms)
def _pert_kw(name, nT, seed):
    if name.startswith("SY"): return dict(shove=(0.0, 0.1 if name[2] == "+" else -0.1, float(name[3:]), 0.2), n_cyc=N_TRANS + N_PER + 8)
    kw = dict(PERTURB.get(name, {})) if name != "nominal" else {}
    if kw.get("noise") == "start":
        rng = np.random.default_rng(kw.pop("seed", seed)); kw["noise"] = dict(q=rng.normal(0, kw.pop("q_sd", 0.02), nT), vx=kw.pop("vx", 0.05))
    return kw


def _val_job(a):
    robot, v, turn, stride, z, name, dt, seed = a
    os.environ.setdefault("SPRINT_LAG", str(LAG))
    P_ = Walk(robot, v, turn, stride); kw = _pert_kw(name, len(P_.S[dt]["types"]), seed)
    t0 = time.time(); r = P_.run(z, dt, **kw); return name, dt, summarize(r) | dict(secs=round(time.time() - t0, 1))


def validate(path, workers=5):
    import multiprocessing as mp
    g = json.load(open(path)); robot, v, turn, z, stride = g["robot"], g["v"], g.get("turn", 0.0), g["z"], g["stride"]
    Walk(robot, v, turn, stride)                                     # builds the (source-keyed) scaffold once before the workers start
    jobs = [(robot, v, turn, stride, z, nm, dt, 11) for nm in ["nominal"] + [k for k in PERTURB if k not in SEARCH_ONLY] for dt in (core.DT_OPT, core.DT_CHECK)]
    if os.environ.get("WALK_SWEEP", "1") == "1": jobs += [(robot, v, turn, stride, z, nm, core.DT_OPT, 11) for nm in SWEEP]
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool: res = pool.map(_val_job, jobs, chunksize=1)
    out = {}
    for nm, dt, r in res: out.setdefault(nm, {})[str(dt)] = r
    ok = {nm: all(out[nm][k]["valid"] for k in out[nm]) for nm in out}
    n_f = sum(ok[nm] for nm in FRESH); n_n58 = sum(ok[nm] for nm in ("N5", "N6", "N7", "N8")); n_sw = sum(ok[nm] for nm in SWEEP if nm in ok)
    set_a = ["shove_lat", "shove_back", "start_noise", "lag_fric"]; set_b = ["shove_lat", "shove_back", "start_noise_2mrad", "lag_fric"]
    n_pert = sum(ok[nm] for nm in set_a); n_b = sum(ok[nm] for nm in set_b); n_h = sum(ok[nm] for nm in HELD_OUT)
    val = dict(robot=robot, v=v, turn=turn, source=path, metric="speed = net (stride-sampled COM chords); slip anchor kept through < 20 ms load blips (repair round 1)",
               v_net={dt: out["nominal"][dt]["value"] for dt in out["nominal"]}, v_path={dt: out["nominal"][dt]["v_path"] for dt in out["nominal"]},
               nominal_valid_both_dt=ok["nominal"], heldout_pass=n_h, robust_N1N4=bool(ok["nominal"] and n_h >= 3),
               fresh_pass=n_f, robust=bool(ok["nominal"] and n_f >= 3), n5n8_pass=n_n58, lat_sweep_pass=n_sw, lat_sweep_n=len(SWEEP),
               label_rule="repair round 2: robust = nominal valid at 1 ms and 0.5 ms AND >= 3/4 of the FRESH set F1-F4 at both dt (registered in PREDICTION_R2.md); N1-N8 reported (consulted earlier, not a label set)",
               perturb_pass_builder_set=n_pert, perturb_n=4, perturb_pass_2mrad_set=n_b, robust_builder_set=bool(ok["nominal"] and n_pert >= 3),
               sets=dict(heldout=HELD_OUT, builder_pre_set=set_a, project_convention=set_b), seed_start_noise=11, ok=ok, runs=out, perturbations={k: str(x) for k, x in PERTURB.items()},
                                                      note="fresh processes (spawn) build fresh plants; shoves are judged on the walking gates with the energy-closure gate waived (external work)")
    json.dump(val, open(path.replace(".json", os.environ.get("WALK_VAL_SUFFIX", "_validation") + ".json"), "w"), indent=1, default=float); return val


def _shove_ok(r): return r["valid"]


def save_prior(robot, v, turn_deg=0.0):
    """The prior (z = 0) as a gait record (same schema as a search result) so it can be validated and stored in the library."""
    Pb = Walk(robot, v, np.radians(turn_deg)); z = np.zeros(Pb.n()); d_ = f"{RES}/{robot}"; os.makedirs(d_, exist_ok=True)
    pth = f"{d_}/prior_v{v}{'_turn%d' % round(turn_deg) if turn_deg else ''}.json"
    json.dump(dict(robot=robot, v=v, turn=float(np.radians(turn_deg)), seed=None, score=None, z=z.tolist(), stride=stride_rec(Pb.st), nominal=Pb.nominal[core.DT_OPT], kind="prior (z = 0, no search)",
                   params={k: (x if isinstance(x, dict) else float(x)) for k, x in Pb.unpack(z).items() if k != "x"}, supplied=SUPPLIED), open(pth, "w"), indent=1, default=float)
    return pth


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "predict": predict(robots=tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("G1", "H1"))
    elif cmd == "prior":
        robot, v = sys.argv[2], float(sys.argv[3]); turn = np.radians(float(sys.argv[4])) if len(sys.argv) > 4 else 0.0
        t0 = time.time(); Pb = Walk(robot, v, turn); print("scaffold", round(time.time() - t0, 1), "s ik_err", Pb.R["ik_err"], stride_rec(Pb.st), flush=True)
        z = np.zeros(Pb.n())
        for dt in (core.DT_OPT, core.DT_CHECK):
            t0 = time.time(); r = Pb.run(z, dt); print(dt, round(time.time() - t0, 1), "s", summarize(r), flush=True)
    elif cmd == "search":
        robot, v, it, pop, wk, seed = sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
        turn = np.radians(float(sys.argv[8])) if len(sys.argv) > 8 else 0.0; z0 = None; stride = None
        if len(sys.argv) > 9:
            w_ = json.load(open(sys.argv[9])); z0 = w_["z"]
        r = search(robot, v, it, pop, wk, seed, turn=turn, z0=z0, stride=stride, tag=os.environ.get("WALK_TAG", ""))
        msg = f"walk {robot} v{v} turn{round(np.degrees(turn))}: score {r['score']:.3f} v {r['r1']['value']:.2f}/{r['r2']['value']:.2f} {r['r1']['reasons']}{r['r2']['reasons']} {r['secs']/60:.0f}min"
        print(msg, flush=True); os.system(f'~/bin/notify "{msg[:150]}" >/dev/null 2>&1')
    elif cmd == "saveprior": print(save_prior(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 0.0))
    elif cmd == "validate":
        r = validate(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5); print(json.dumps({k: r[k] for k in ("robot", "v", "turn", "v_net", "nominal_valid_both_dt", "heldout_pass", "n5n8_pass", "fresh_pass", "lat_sweep_pass", "perturb_pass_builder_set", "robust", "ok")}))
