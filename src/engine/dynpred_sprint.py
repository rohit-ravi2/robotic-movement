"""DYNAMIC STRIDE PREDICTOR for the sprint (track T2, 2026-10-17). Replaces the static ceiling (engine.sprint_ceiling) as the
predictor of v_max. Spec only: every body number is read from the stock MJCF and the vendor ratings in throw.setup (rb); NO constant
is fitted to a found speed. A counterfactual "what-if body" (ankle_scale) multiplies the ankle pitch tau AND omega, as track C did,
and is a dose-response study only, never a robot capability.

For a candidate speed v the model asks whether ANY running stride (contact t_c, flight t_f, stance hip height h0) satisfies:
  VERTICAL DYNAMICS   half-sine vertical GRF with impulse m g (t_c + t_f) per step; ballistic flight t_f; stance compression from
                      spring-mass consistency dz = g t_f t_c / (2 pi) (touchdown descent rate = g t_f / 2).
  STANCE              ground force along the CoP -> hip line (spring-mass, virtual pivot at the CoP). FLAT phase: CoP under the ankle
                      (the sole height hf still loads the ankle with hf F_x). HEEL-OFF (foot lever): the foot rotates about the toe
                      once the ankle can hold the CoP at the toe, i.e. from the first instant after mid-stance at which
                      |x_toe F_z + hf F_x| fits the ankle's torque-speed envelope for the rest of the stance (scanned later until the
                      whole push-off fits); the ankle then extends the leg (hip-ankle distance held at its heel-off value).
                      Hip, knee and ankle torques by J^T on the planar chain; every joint inside its torque-speed envelope
                      tau_max (1 - |w|/omega) when driving, tau_max when absorbing, |w| <= 1.1 omega (the OVERSPEED gate).
  ACTUATOR LAG        a first-order lag tau_a attenuates the stance force's fundamental (pi/t_c) by 1/sqrt(1 + (pi tau_a/t_c)^2): the
                      stance torque requirement is multiplied by that factor (physics of the lag; the construction advances the
                      reference by tau_a but does not amplify it).
  FRICTION            |F_x| <= mu F_z throughout the stance, mu = the model's foot-floor friction (MuJoCo takes the max of the pair).
  SWING               the other leg recovers within 2 (t_c + t_f) - t_c: hip sweep and knee fold, minimum-time moves of the MODEL's
                      composite inertias (qM diagonal at the settled pose) under the torque-speed envelope (sprint_ceiling.min_time);
                      the knee must also unfold inside one flight.
  YAW MOMENTUM        the legs' yaw angular momentum, H_legs = m_leg y_leg (v_back + v_fwd) (stance leg COM moving back at
                      (r_leg/L) v, swing leg COM forward at (r_leg/L) Lc/t_sw, opposite sides), is cancelled by the arms swinging
                      at the stride frequency up to the shoulder's torque-speed envelope (rate w_a, torque I_arm w_a 2 pi f +
                      m_arm g r_arm sin A); the residual must be reversed by the stance foot's friction yaw moment within one contact,
                      2 H_res / t_c <= mu Fz_mean (x_toe - x_heel)/2.
  TASK / CONSTRUCTION FLOORS (listed, not fitted): flight t_f >= 0.05 s (the project's running definition, TF_MIN); contact
                      t_c >= 0.85 x pi x 0.040 s (the scaffold's stance-force band floor x the search's fastest time scale 0.85);
                      t_c >= pi tau_a (a force pulse cannot be shorter than the lag's band).
Not modelled: trunk pitch angular-momentum balance (swing-hip reaction), impact losses, lateral balance, periodicity, the soft-contact
creep that the 3 cm SLIP gate catches. Output: v_max (largest feasible v on a 0.1 m/s grid) with the binding constraint (most frequent failure) at the next grid speed.
"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from engine import sprint_ceiling as SC

G = 9.81
TF_MIN = 0.05
TC_FLOOR = 0.85*np.pi*0.040
SHOULDER = dict(G1="left_shoulder_pitch_joint", H1="left_shoulder_pitch", Apollo="left_shoulder_pitch_joint", T1="left_shoulder_pitch_joint")


def spec(robot):
    """Body spec: sprint_ceiling.foot_spec plus model inertias (qM diagonal), arm and leg masses / lateral offsets, shoulder rating, mu."""
    import throw as T
    B = SC.foot_spec(robot); S = T.setup(robot=robot); m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, M, d.qM)
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n); c = SC.CHAINS[robot]
    jh, jk, js = jid(c["hip"]), jid(c["knee"]), jid(SHOULDER[robot])
    pel = 1; cy = float(d.subtree_com[pel][1])
    # leg = subtree of the first joint of the left leg chain (hip yaw/roll/pitch order differs): walk up from the hip-pitch body
    b = int(m.jnt_bodyid[jh])
    while int(m.body_parentid[b]) != pel and int(m.body_parentid[b]) != 0: b = int(m.body_parentid[b])
    bs = int(m.jnt_bodyid[js]); rs = d.subtree_com[bs] - d.xanchor[js]
    sh = S["rb"].J[SHOULDER[robot]]
    mu = max(float(m.geom_friction[g][0]) for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and int(m.geom_bodyid[g]) != 0) if any(
        (m.geom_contype[g] or m.geom_conaffinity[g]) and int(m.geom_bodyid[g]) != 0 for g in range(m.ngeom)) else 1.0
    mu = max(mu, float(m.geom_friction[0][0]))
    B.update(I_hip=float(M[m.jnt_dofadr[jh], m.jnt_dofadr[jh]]), I_knee=float(M[m.jnt_dofadr[jk], m.jnt_dofadr[jk]]),
             m_leg=float(m.body_subtreemass[b]), y_leg=float(abs(d.subtree_com[b][1] - cy)), r_leg=float(np.linalg.norm(d.subtree_com[b] - d.xanchor[jh])),
             m_arm=float(m.body_subtreemass[bs]), r_arm=float(np.linalg.norm(rs[[0, 2]])), y_arm=float(abs(d.subtree_com[bs][1] - cy)),
             I_arm=float(M[m.jnt_dofadr[js], m.jnt_dofadr[js]]), shoulder=dict(tau=float(sh["tau"]), omega=float(sh["omega"])), mu=mu)
    return B


def _env(tau, w, tmax, om):
    drive = tau*w > 0; lim = np.where(drive, tmax*np.clip(1 - np.abs(w)/om, 0, None), tmax)
    return float(np.max(np.abs(tau)/np.maximum(lim, 1e-9))), float(np.max(np.abs(w))/om)


def stance(B, v, tc, tf, h0, lag=0.002, ankle_scale=1.0, n=60):
    L1, L2, hf, xt = B["L1"], B["L2"], B["hf"], B["x_toe"]; Lc = v*tc; Tst = tc + tf; dz = G*tf*tc/(2*np.pi)
    kL = np.sqrt(1 + (np.pi*lag/tc)**2)
    t = np.linspace(0, tc, n); xh = -Lc/2 + v*t; hh = h0 - dz*np.sin(np.pi*t/tc)       # hip relative to the flat-foot ankle
    Fz = np.pi*B["mass"]*G*Tst/(2*tc)*np.sin(np.pi*t/tc)
    ta, oa = B["ankle"]["tau"]*ankle_scale, B["ankle"]["omega"]*ankle_scale
    toe = np.array([xt, -hf]); o0 = np.array([-xt, hf]); rot = lambda ps: (o0[0]*np.cos(ps) - o0[1]*np.sin(ps), o0[0]*np.sin(ps) + o0[1]*np.cos(ps))
    best = None
    # heel-off candidates: from mid-stance on (indices), earliest first; i0 = n means no heel-off (flat foot throughout)
    for i0 in [n] + list(range(n//2, n, 3)):
        ax = np.zeros(n); az = np.zeros(n); psi = np.zeros(n); cop = np.zeros(n)        # CoP x relative to the flat-foot ankle
        if i0 < n:
            r_off = np.hypot(xh[i0], hh[i0]); X, Z = xh[i0:] - toe[0], hh[i0:] - toe[1]
            f = lambda ps: np.hypot(X - rot(ps)[0], Z - rot(ps)[1]) - r_off
            if np.any(f(np.full(n - i0, -1.5)) > 0): continue
            lo, hi = np.full(n - i0, -1.5), np.zeros(n - i0)
            for _ in range(40):
                mid = 0.5*(lo + hi); pos = f(mid) > 0; hi = np.where(pos, mid, hi); lo = np.where(pos, lo, mid)
            psi[i0:] = hi; ax[i0:], az[i0:] = toe[0] + rot(hi)[0], toe[1] + rot(hi)[1]; cop[i0:] = xt
        if np.any(np.hypot(xh - ax, hh - az) > L1 + L2 - 1e-3): continue
        Fxg = Fz*(xh - cop)/hh                                                         # ground force on the foot, along the CoP -> hip line
        # (engine.sprint_ceiling uses Fz (-xh)/hh: the horizontal component MIRRORED about the leg line -- propulsive at touchdown and a
        #  spurious hip moment ~ 2 x_h F_z; corrected here)
        if np.any(np.abs(Fxg) > B["mu"]*Fz + 1e-9): continue
        th, ka = SC.ik(ax - xh, az - hh, L1, L2)
        if np.any(ka < B["knee"]["rng"][0]) or np.any(ka > B["knee"]["rng"][1]): continue
        ox = np.where(np.arange(n) < i0, 0.0, toe[0] - ax); oz = np.where(np.arange(n) < i0, -hf, toe[1] - az)
        dax_dth = L1*np.cos(th) + L2*np.cos(th - ka); daz_dth = L1*np.sin(th) + L2*np.sin(th - ka); dax_dka = -L2*np.cos(th - ka); daz_dka = -L2*np.sin(th - ka)
        tau_a = -(-oz*Fxg + ox*Fz); tau_h = -(dax_dth*Fxg + daz_dth*Fz) + tau_a; tau_k = -(dax_dka*Fxg + daz_dka*Fz) - tau_a
        phi = psi - (th - ka); w_h = np.gradient(th, t); w_k = np.gradient(ka, t); w_a = np.gradient(phi, t)
        res = {}; ok = True; why = ""
        for nm, tau, w, tm, om in (("hip", tau_h, w_h, B["hip"]["tau"], B["hip"]["omega"]), ("knee", tau_k, w_k, B["knee"]["tau"], B["knee"]["omega"]),
                                   ("ankle", tau_a, w_a, ta, oa)):
            u, s = _env(kL*tau, w, tm, om); res[nm] = (u, s)
            if u > 1.0 and ok: ok, why = False, f"{nm}_torque"
            if s > 1.1 and ok: ok, why = False, f"{nm}_speed"
        if ok:
            return True, "", dict(res, sweep=float(th[0] - th[-1]), ka_start=float(ka[0]), ka_end=float(ka[-1]), heel_off_frac=float(1 - i0/n), Lc=Lc, dz=dz, lag_factor=kL)
        best = best or (why, res)
    return False, (best[0] if best else "reach"), (best[1] if best else None)


def swing(B, tc, tf, h0, st):
    tsw = 2*(tc + tf) - tc; L1, L2, hf = B["L1"], B["L2"], B["hf"]
    reach = h0 - hf - 0.03
    c = np.clip((L1*L1 + L2*L2 - reach*reach)/(2*L1*L2), -1, 1); k_clear = max(np.pi - np.arccos(c), 0.0)
    th_h = SC.min_time(st["sweep"], B["I_hip"], B["hip"]["tau"], B["hip"]["omega"])
    tk1 = SC.min_time(k_clear - st["ka_end"], B["I_knee"], B["knee"]["tau"], B["knee"]["omega"]); tk2 = SC.min_time(k_clear - st["ka_start"], B["I_knee"], B["knee"]["tau"], B["knee"]["omega"])
    if th_h > tsw: return False, "hip_swing", None
    if tk1 + tk2 > tsw: return False, "knee_swing", None
    if tk2 > tf + 1e-9: return False, "knee_unfold_in_flight", None
    return True, "", dict(t_hip=th_h, t_knee=tk1 + tk2, t_sw=tsw, k_clear=k_clear)


def arms(B, v, tc, tf, L):
    """Yaw angular momentum: the legs' H_legs is cancelled by the arms up to their torque-speed envelope (antiphase swing at the
    stride frequency); any residual must be absorbed by the stance foot's friction yaw moment, M_z <= mu Fz_mean (x_toe - x_heel)/2
    (multi-point sole contact), reversing 2 H_res within one contact time."""
    tsw = 2*(tc + tf) - tc; f = 1.0/(2*(tc + tf)); k = B["r_leg"]/L; W = 2*np.pi*f
    H = B["m_leg"]*B["y_leg"]*(k*v + k*v*tc/tsw)
    tm, om = B["shoulder"]["tau"], B["shoulder"]["omega"]; ca = 2*B["m_arm"]*B["r_arm"]*B["y_arm"]
    def u_of(w_a):
        A = w_a/W; tau = B["I_arm"]*w_a*W + B["m_arm"]*G*B["r_arm"]*np.sin(min(A, np.pi/2))
        return tau/max(tm*max(0.0, 1 - w_a/om), 1e-9)
    w_need = H/ca; lo, hi = 0.0, min(w_need, om)
    if u_of(hi) <= 1.0: w_max = hi
    else:
        for _ in range(40):
            mid = 0.5*(lo + hi); lo, hi = (mid, hi) if u_of(mid) <= 1.0 else (lo, mid)
        w_max = lo
    H_res = max(0.0, H - ca*w_max)
    Mz_av = B["mu"]*B["mass"]*G*(tc + tf)/tc*(B["x_toe"] - B["x_heel"])/2; Mz_req = 2*H_res/tc
    info = dict(H_legs=H, w_arm=w_max, arm_share=float(ca*w_max/max(H, 1e-9)), Mz_req=Mz_req, Mz_av=Mz_av)
    if Mz_req > Mz_av: return False, "yaw_momentum", info
    return True, "", info


def feasible(B, v, lag=0.002, ankle_scale=1.0):
    L = B["L1"] + B["L2"]; fails = {}; tcmin = max(TC_FLOOR, np.pi*lag)
    for tc, tf, hr in itertools.product(np.arange(np.ceil(tcmin*100)/100, 0.401, 0.01), np.arange(TF_MIN, 0.201, 0.01), np.arange(0.70, 0.981, 0.04)):
        ok, why, ar = arms(B, v, tc, tf, L)
        if not ok: fails[why] = fails.get(why, 0) + 1; continue
        ok, why, st = stance(B, v, tc, tf, hr*L, lag, ankle_scale)
        if not ok: fails[why] = fails.get(why, 0) + 1; continue
        ok, why, sw = swing(B, tc, tf, hr*L, st)
        if not ok: fails[why] = fails.get(why, 0) + 1; continue
        return True, dict(tc=float(tc), tf=float(tf), h0=float(hr*L), step_hz=float(1/(tc + tf)), stance=st, swing=sw, arms=ar)
    return False, fails


def v_max(B, lag=0.002, ankle_scale=1.0, vs=np.arange(0.5, 8.001, 0.1)):
    best = None; first_fail = None
    for v in vs:
        ok, info = feasible(B, float(v), lag, ankle_scale)
        if ok: best = dict(v=round(float(v), 3), stride=info); first_fail = None
        elif best is not None and first_fail is None: first_fail = dict(v=round(float(v), 3), fails=info)
        if not ok and best is not None and v > best["v"] + 0.5: break
    if best is None: return dict(v_max=0.0, binding=max(info, key=info.get) if info else None, fails=info)
    ff = first_fail["fails"] if first_fail else {}
    return dict(v_max=best["v"], at=best["stride"], binding=(max(ff, key=ff.get) if ff else None), fails_above=ff)
