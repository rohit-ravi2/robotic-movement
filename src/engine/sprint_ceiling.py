"""Sprint speed CEILING from the body spec, before any search (2026-10-14).

Planar running model per robot, every number read from the MJCF / plant (throw.setup): thigh and shank lengths (hip-pitch ->
knee -> ankle body origins), ankle-to-sole height, leg masses, joint ranges, rated torque tau and rated speed omega per joint.
For a candidate speed v the model asks whether ANY stride (contact time t_c, flight time t_f, stance hip height h0, stance
compression d) is feasible for the actuators:
  STANCE  foot pinned under the ankle (point contact: optimistic for the ankle, which carries no torque here), hip moves from
          -Lc/2 to +Lc/2 over the foot, Lc = v t_c, hip height h0 - d sin(pi t/t_c). Vertical force half-sine with impulse
          m g (t_c + t_f) per step (one leg carries the body); force directed along the hip-ankle line (spring-mass), so it brakes
          then propels. Joint torques tau = J^T F (leg mass neglected in stance), joint speeds from the kinematics. Driving
          (tau w > 0): |tau| <= tau_max (1 - |w|/omega); absorbing: |tau| <= tau_max; always |w| <= 1.1 omega (OVERSPEED gate).
  SWING   the other leg must recover within 2T - t_c (T = t_c + t_f): hip pitch through the stance sweep angle, knee fold to the
          angle that clears the floor by 3 cm under the hip and unfold to touchdown, each a minimum-time move of a lumped
          segment inertia under the same torque-speed envelope (driving accel, rated-torque braking).
Not modelled (each makes the ceiling OPTIMISTIC): trunk pitch / angular-momentum balance, arm swing cost, ankle push-off limits,
foot slip (mu), impact losses at touchdown, lateral dynamics. The output is an upper bound with the binding joint named, to be
compared with what the construction achieves in the simulator.
Usage: python engine/sprint_ceiling.py   -> results/sprint/ceiling.json"""
import os, sys, json, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import throw as T
import mujoco

G = 9.81
CHAINS = dict(G1=dict(hip="left_hip_pitch_joint", knee="left_knee_joint", hipb="left_hip_pitch_link", kneeb="left_knee_link", ankleb="left_ankle_roll_link",
                      thigh=("left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link"), shank=("left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link")),
              H1=dict(hip="left_hip_pitch", knee="left_knee", hipb="left_hip_pitch_link", kneeb="left_knee_link", ankleb="left_ankle_link",
                      thigh=("left_hip_pitch_link",), shank=("left_knee_link", "left_ankle_link")),
              # HELD-OUT (2026-10-16): Apollo's hip order is yaw -> roll -> pitch (thigh = the pitch link, as H1); T1's is G1's
              Apollo=dict(hip="left_hip_pitch_joint", knee="left_knee_joint", hipb="left_hip_fe_link", kneeb="left_knee_fe_link", ankleb="left_foot_link",
                          thigh=("left_hip_fe_link",), shank=("left_knee_fe_link", "left_ankle_ie_link", "left_foot_link")),
              T1=dict(hip="left_hip_pitch_joint", knee="left_knee_joint", hipb="left_hip_pitch_link", kneeb="left_knee_link", ankleb="left_ankle_roll_link",
                      thigh=("left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link"), shank=("left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link")))


def body_spec(robot):
    S = T.setup(robot=robot); m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d); c = CHAINS[robot]
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n); jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    ph, pk, pa = d.xpos[bid(c["hipb"])], d.xpos[bid(c["kneeb"])], d.xpos[bid(c["ankleb"])]
    L1 = float(np.linalg.norm((pk - ph)[[0, 2]])); L2 = float(np.linalg.norm((pa - pk)[[0, 2]]))
    sole = min(float(d.geom_xpos[g][2] - (m.geom_size[g][2] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else m.geom_size[g][0])) for g in range(m.ngeom) if m.geom_bodyid[g] == bid(c["ankleb"])) if any(m.geom_bodyid[g] == bid(c["ankleb"]) for g in range(m.ngeom)) else 0.0
    hf = float(pa[2] - max(sole, 0.0))
    rb = S["rb"].J
    def rating(j):
        for k in rb:
            if k == j or k == j.replace("_joint", ""): return float(rb[k]["tau"]), float(rb[k]["omega"])
        raise KeyError(j)
    mt = float(sum(m.body_mass[bid(b)] for b in c["thigh"])); ms = float(sum(m.body_mass[bid(b)] for b in c["shank"]))
    return dict(robot=robot, mass=float(m.body_mass.sum()), L1=L1, L2=L2, hf=hf, m_thigh=mt, m_shank=ms,
                hip=dict(zip(("tau", "omega"), rating(c["hip"])), rng=m.jnt_range[jid(c["hip"])].tolist()),
                knee=dict(zip(("tau", "omega"), rating(c["knee"])), rng=m.jnt_range[jid(c["knee"])].tolist()))


def min_time(angle, I, tau, om, cap=1.1):
    """Minimum time to rotate a lumped inertia I through `angle` from rest to rest: accelerate on the driving envelope
    tau (1 - w/om) up to the switching point, brake at rated torque; speed capped at cap*om."""
    angle = abs(angle)
    if angle < 1e-6: return 0.0
    dt = 2e-4; w = th = t = 0.0; wmax = cap*om
    while True:
        a_brake = tau/I; stop = w*w/(2*a_brake)
        if th + stop >= angle: return t + w/a_brake
        w = min(w + dt*tau*max(0.0, 1 - w/om)/I, wmax); th += w*dt; t += dt
        if t > 5: return np.inf


def ik(x, z, L1, L2):
    """Hip at origin, ankle at (x, z) (z < 0): thigh angle from vertical (positive = foot ahead = flexion) and knee flexion."""
    r = np.hypot(x, z); c = np.clip((L1*L1 + L2*L2 - r*r)/(2*L1*L2), -1, 1); kappa = np.pi - np.arccos(c)
    alpha = np.arctan2(x, -z); beta = np.arccos(np.clip((L1*L1 + r*r - L2*L2)/(2*L1*r), -1, 1)); return alpha + beta, kappa   # knee bends backward: thigh ahead of the hip-ankle line


def stance_ok(B, v, tc, tf, h0, dz, n=60, scale=1.0):
    L1, L2 = B["L1"], B["L2"]; Lc = v*tc; Tst = tc + tf; t = np.linspace(0, tc, n); xh = -Lc/2 + v*t; hh = h0 - dz*np.sin(np.pi*t/tc)
    if np.any(np.hypot(xh, hh) > L1 + L2 - 1e-3): return False, "reach", None
    th, ka = ik(-xh, -hh, L1, L2)                                          # ankle relative to the hip
    if np.any(ka < B["knee"]["rng"][0]) or np.any(ka > B["knee"]["rng"][1]): return False, "knee_range", None
    Fz = np.pi*B["mass"]*G*Tst/(2*tc)*np.sin(np.pi*t/tc); Fx = Fz*(-xh)/hh   # along hip->ankle line: ground pushes the hip
    # Jacobian of ankle position (relative to hip) w.r.t. (thigh angle th, knee flexion ka): ankle = L1 (sin th, -cos th) + L2 (sin(th - ka), -cos(th - ka))
    dax_dth = L1*np.cos(th) + L2*np.cos(th - ka); daz_dth = L1*np.sin(th) + L2*np.sin(th - ka); dax_dka = -L2*np.cos(th - ka); daz_dka = -L2*np.sin(th - ka)
    tau_h = -(dax_dth*Fx + daz_dth*Fz); tau_k = -(dax_dka*Fx + daz_dka*Fz)   # torque the joints must supply against the ground reaction on the foot
    w_h = np.gradient(th, t); w_k = np.gradient(ka, t); worst = {}
    for nm, tau, w in (("hip", tau_h, w_h), ("knee", tau_k, w_k)):
        R = B[nm]; tmax = R["tau"]*scale; om = R["omega"]*scale
        drive = tau*w > 0; lim = np.where(drive, tmax*np.clip(1 - np.abs(w)/om, 0, None), tmax)
        util = np.max(np.abs(tau)/np.maximum(lim, 1e-9)); spd = np.max(np.abs(w))/om; worst[nm] = (float(util), float(spd))
        if util > 1.0: return False, f"{nm}_torque", worst
        if spd > 1.1: return False, f"{nm}_speed", worst
    return True, "", dict(worst, sweep=float(th[0] - th[-1]), ka_start=float(ka[0]), ka_end=float(ka[-1]))


def swing_ok(B, tc, tf, h0, st, scale=1.0):
    tsw = 2*(tc + tf) - tc; L1, L2, hf = B["L1"], B["L2"], B["hf"]
    I_hip = B["m_thigh"]*(0.45*L1)**2 + B["m_shank"]*(L1 + 0.35*L2)**2*0.7 + (B["m_thigh"]*L1**2 + B["m_shank"]*L2**2)/12   # knee partly folded in swing
    I_knee = B["m_shank"]*(0.4*L2)**2 + B["m_shank"]*L2**2/12
    reach = h0 - hf - 0.03                                                            # hip-to-ankle length that clears the floor by 3 cm
    c = np.clip((L1*L1 + L2*L2 - reach*reach)/(2*L1*L2), -1, 1); k_clear = max(np.pi - np.arccos(c), 0.0)
    th_h = min_time(st["sweep"], I_hip, B["hip"]["tau"]*scale, B["hip"]["omega"]*scale)
    th_k = min_time(k_clear - st["ka_end"], I_knee, B["knee"]["tau"]*scale, B["knee"]["omega"]*scale) + min_time(k_clear - st["ka_start"], I_knee, B["knee"]["tau"]*scale, B["knee"]["omega"]*scale)
    if th_h > tsw: return False, "hip_swing", dict(t_hip=th_h, t_knee=th_k, t_sw=tsw)
    if th_k > tsw: return False, "knee_swing", dict(t_hip=th_h, t_knee=th_k, t_sw=tsw)
    return True, "", dict(t_hip=th_h, t_knee=th_k, t_sw=tsw, k_clear=k_clear)


def feasible(B, v, scale=1.0):
    L = B["L1"] + B["L2"]; fails = {}
    for tc, tf, hr, dr in itertools.product(np.linspace(0.06, 0.40, 18), np.linspace(0.0, 0.20, 9), np.linspace(0.70, 0.98, 8), (0.0, 0.03, 0.06, 0.10)):
        ok, why, st = stance_ok(B, v, tc, tf, hr*L, dr*L, scale=scale)
        if not ok: fails[why] = fails.get(why, 0) + 1; continue
        ok, why, sw = swing_ok(B, tc, tf, hr*L, st, scale=scale)
        if not ok: fails[why] = fails.get(why, 0) + 1; continue
        return True, dict(tc=float(tc), tf=float(tf), h0=float(hr*L), dz=float(dr*L), stride=float(v*(tc + tf)), step_hz=float(1/(tc + tf)), stance=st, swing=sw)
    return False, fails


def ceiling(B, scale=1.0, vs=np.arange(0.5, 16.01, 0.25)):
    best = None; last_fail = None
    for v in vs:
        ok, info = feasible(B, float(v), scale)
        if ok: best = dict(v=float(v), stride=info)
        else: last_fail = dict(v=float(v), fails=info); break
    return best, last_fail


# ---- FOOT LEVER model (2026-10-16, opt-in: CEIL_FOOT=1 or foot=True) ------------------------------------------------------
# The point-foot stance above loads no ankle. Here the foot is a rigid lever read from the sole geoms (heel and toe x relative
# to the ankle, sole hf below it). Flat phase [0, (1-pf) t_c]: the centre of pressure moves linearly heel -> toe, the ankle stays
# where the point-foot model pins it. Push-off phase [(1-pf) t_c, t_c]: CoP at the toe, the heel lifts and the foot rotates about
# the toe so the hip-ankle length stays at its heel-off value (the ankle, not the knee, extends the leg). The ground force is
# the point-foot one (same magnitude, same direction); joint torques are -(d p_contact / d q)^T F, so the ankle carries
# tau_a = -(o x F) (o = CoP - ankle) and the same moment passes up to knee and hip. Ankle angular speed = d/dt (foot angle -
# shank angle). The ankle pitch joint has its own driving/absorbing envelope and 1.1 omega OVERSPEED gate.
ANKLE = dict(G1="left_ankle_pitch_joint", H1="left_ankle", Apollo="left_ankle_pitch_joint", T1="left_ankle_pitch_joint")


def foot_spec(robot, B=None):
    """B (body_spec) plus foot geometry: x_heel, x_toe (m, relative to the ankle, + forward) and the ankle pitch rating."""
    B = dict(B or body_spec(robot)); S = T.setup(robot=robot); m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d)
    c = CHAINS[robot]; b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, c["ankleb"]); pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH: continue
        R = d.geom_xmat[g].reshape(3, 3); p = d.geom_xpos[g]; s = m.geom_size[g]; ty = m.geom_type[g]
        if ty == mujoco.mjtGeom.mjGEOM_BOX: loc = [np.array([i*s[0], j*s[1], k*s[2]]) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
        elif ty == mujoco.mjtGeom.mjGEOM_CAPSULE: loc = [np.array([0, 0, k*s[1]]) + e for k in (-1, 1) for e in (np.array([s[0], 0, 0]), -np.array([s[0], 0, 0]), np.array([0, 0, s[0]]), -np.array([0, 0, s[0]]))]
        elif ty == mujoco.mjtGeom.mjGEOM_SPHERE: loc = [np.array([e*s[0], 0, 0]) for e in (-1, 1)] + [np.array([0, 0, -s[0]])]
        else: continue
        pts += [p + R @ l for l in loc]
    pts = np.array(pts); low = pts[pts[:, 2] < pts[:, 2].min() + 0.03]; ax = d.xpos[b][0]   # points within 3 cm of the sole
    tau, om = None, None
    for k, r in S["rb"].J.items():
        if k == ANKLE[robot]: tau, om = float(r["tau"]), float(r["omega"])
    B.update(x_heel=float(pts[:, 0].min() - ax) if len(low) == 0 else float(low[:, 0].min() - ax), x_toe=float(low[:, 0].max() - ax),
             ankle=dict(tau=tau, omega=om))
    return B


def _envelope(tau, w, tmax, om):
    drive = tau*w > 0; lim = np.where(drive, tmax*np.clip(1 - np.abs(w)/om, 0, None), tmax)
    return float(np.max(np.abs(tau)/np.maximum(lim, 1e-9))), float(np.max(np.abs(w))/om)


def stance_ok_foot(B, v, tc, tf, h0, dz, n=60, scale=1.0, ankle_scale=1.0, pf=0.4):
    L1, L2, hf, xh0, xt = B["L1"], B["L2"], B["hf"], B["x_heel"], B["x_toe"]; Lc = v*tc; Tst = tc + tf
    t = np.linspace(0, tc, n); xh = -Lc/2 + v*t; hh = h0 - dz*np.sin(np.pi*t/tc)
    if np.any(np.hypot(xh, hh) > L1 + L2 - 1e-3): return False, "reach", None
    Fz = np.pi*B["mass"]*G*Tst/(2*tc)*np.sin(np.pi*t/tc); Fx = Fz*(-xh)/hh       # the point-foot ground force, unchanged
    toff = (1 - pf)*tc; flat = t <= toff
    ax = np.zeros(n); az = np.zeros(n); psi = np.zeros(n)                          # ankle position (origin = flat-foot ankle) and foot pitch
    i0 = np.searchsorted(t, toff); r_off = np.hypot(np.interp(toff, t, xh), np.interp(toff, t, hh))
    toe = np.array([xt, -hf]); o0 = np.array([-xt, hf])                           # ankle relative to the toe, foot flat
    rot = lambda ps: (o0[0]*np.cos(ps) - o0[1]*np.sin(ps), o0[0]*np.sin(ps) + o0[1]*np.cos(ps))
    if i0 < n:                                                                     # heel rise: bisect foot angle psi in [-1.5, 0] (vectorised)
        X, Z = xh[i0:] - toe[0], hh[i0:] - toe[1]; f = lambda ps: np.hypot(X - rot(ps)[0], Z - rot(ps)[1]) - r_off
        if np.any(f(np.full(n - i0, -1.5)) > 0): return False, "reach", None
        lo, hi = np.full(n - i0, -1.5), np.zeros(n - i0)
        for _ in range(40):
            mid = 0.5*(lo + hi); pos = f(mid) > 0; hi = np.where(pos, mid, hi); lo = np.where(pos, lo, mid)
        psi[i0:] = hi; ax[i0:], az[i0:] = toe[0] + rot(hi)[0], toe[1] + rot(hi)[1]
    th, ka = ik(ax - xh, az - hh, L1, L2)
    if np.any(ka < B["knee"]["rng"][0]) or np.any(ka > B["knee"]["rng"][1]): return False, "knee_range", None
    cop = np.where(flat, xh0 + (xt - xh0)*np.minimum(t/max(toff, 1e-9), 1.0), xt)
    ox = np.where(flat, cop, toe[0] - ax); oz = np.where(flat, -hf, toe[1] - az)   # CoP relative to the ankle (world)
    dax_dth = L1*np.cos(th) + L2*np.cos(th - ka); daz_dth = L1*np.sin(th) + L2*np.sin(th - ka); dax_dka = -L2*np.cos(th - ka); daz_dka = -L2*np.sin(th - ka)
    tau_a = -(-oz*Fx + ox*Fz)                                                      # d o / d psi = (-oz, ox)
    tau_h = -(dax_dth*Fx + daz_dth*Fz) + tau_a; tau_k = -(dax_dka*Fx + daz_dka*Fz) - tau_a   # psi = th - ka + phi
    phi = psi - (th - ka); w_h = np.gradient(th, t); w_k = np.gradient(ka, t); w_a = np.gradient(phi, t); worst = {}
    for nm, tau, w, sc in (("hip", tau_h, w_h, scale), ("knee", tau_k, w_k, scale), ("ankle", tau_a, w_a, ankle_scale)):
        R = B[nm]; util, spd = _envelope(tau, w, R["tau"]*sc, R["omega"]*sc); worst[nm] = (util, spd)
        if util > 1.0: return False, f"{nm}_torque", worst
        if spd > 1.1: return False, f"{nm}_speed", worst
    return True, "", dict(worst, sweep=float(th[0] - th[-1]), ka_start=float(ka[0]), ka_end=float(ka[-1]), tau_a_peak=float(np.max(np.abs(tau_a))), w_a_peak=float(np.max(np.abs(w_a))))


def feasible_ext(B, v, scale=1.0, foot=False, max_hz=None, ankle_scale=1.0, pf=0.4):
    """feasible() with an optional cadence cap (step_hz <= max_hz) and the foot-lever stance."""
    L = B["L1"] + B["L2"]; fails = {}
    for tc, tf, hr, dr in itertools.product(np.linspace(0.06, 0.40, 18), np.linspace(0.0, 0.20, 9), np.linspace(0.70, 0.98, 8), (0.0, 0.03, 0.06, 0.10)):
        if max_hz and 1/(tc + tf) > max_hz + 1e-9: continue
        ok, why, st = (stance_ok_foot(B, v, tc, tf, hr*L, dr*L, scale=scale, ankle_scale=ankle_scale, pf=pf) if foot else stance_ok(B, v, tc, tf, hr*L, dr*L, scale=scale))
        if not ok: fails[why] = fails.get(why, 0) + 1; continue
        ok, why, sw = swing_ok(B, tc, tf, hr*L, st, scale=scale)
        if not ok: fails[why] = fails.get(why, 0) + 1; continue
        return True, dict(tc=float(tc), tf=float(tf), h0=float(hr*L), dz=float(dr*L), stride=float(v*(tc + tf)), step_hz=float(1/(tc + tf)), stance=st, swing=sw)
    return False, fails


def ceiling_ext(B, vs=np.arange(0.5, 16.01, 0.25), **kw):
    best = None; last_fail = None
    for v in vs:
        ok, info = feasible_ext(B, float(v), **kw)
        if ok: best = dict(v=float(v), stride=info)
        else: last_fail = dict(v=float(v), fails=info); break
    return best, last_fail


def footlever_main(out_path="../results/sprint/ceiling_footlever.json"):
    out = {}
    for robot in ("G1", "H1"):
        B = foot_spec(robot); res = dict(foot=dict(x_heel=B["x_heel"], x_toe=B["x_toe"], hf=B["hf"], ankle=B["ankle"]), cases={})
        print(f"{robot}: heel {B['x_heel']:+.3f} toe {B['x_toe']:+.3f} hf {B['hf']:.3f} ankle {B['ankle']}", flush=True)
        for cap in (None, 6.0, 4.5):
            row = {}
            for foot in (False, True):
                best, fail = ceiling_ext(B, foot=foot, max_hz=cap)
                row["foot_lever" if foot else "point_foot"] = dict(v_max=best["v"] if best else 0.0, binding=max(fail["fails"], key=fail["fails"].get) if fail and fail["fails"] else None,
                                                                   fails=fail["fails"] if fail else None, at_vmax={k: w for k, w in (best["stride"].items() if best else []) if k not in ("swing",)})
            vfree = ceiling_ext(B, foot=True, max_hz=cap, ankle_scale=50.0)[0]; vfree = vfree["v"] if vfree else 0.0
            k_free = None
            if row["foot_lever"]["v_max"] < vfree:                                     # smallest ankle scale reaching the ankle-unlimited ceiling
                lo, hi = 1.0, 50.0
                for _ in range(9):
                    mid = 0.5*(lo + hi); b = ceiling_ext(B, foot=True, max_hz=cap, ankle_scale=mid, vs=np.arange(row["foot_lever"]["v_max"], vfree + 0.01, 0.25))[0]
                    if b and b["v"] >= vfree: hi = mid
                    else: lo = mid
                k_free = round(hi, 2)
            else: k_free = 1.0
            row["ankle_unlimited_v_max"] = vfree; row["ankle_scale_not_limiting"] = k_free
            res["cases"]["any" if cap is None else f"le_{cap}Hz"] = row
            print(f"  cap {cap}: point {row['point_foot']['v_max']} ({row['point_foot']['binding']}) | lever {row['foot_lever']['v_max']} ({row['foot_lever']['binding']}) | ankle-free {vfree} at ankle x{k_free}", flush=True)
        out[robot] = res
    os.makedirs(os.path.dirname(out_path), exist_ok=True); json.dump(out, open(out_path, "w"), indent=1, default=float); return out


if __name__ == "__main__":
    if os.environ.get("CEIL_FOOT") == "1": footlever_main(); sys.exit(0)
    out = {}
    for robot in ("G1", "H1"):
        B = body_spec(robot); best, fail = ceiling(B)
        binding = max(fail["fails"], key=fail["fails"].get) if fail else None
        res = dict(body=B, v_max=best["v"] if best else 0.0, at_vmax=best, first_infeasible=fail, binding=binding, scaling={})
        print(f"{robot}: L1 {B['L1']:.3f} L2 {B['L2']:.3f} hf {B['hf']:.3f} m {B['mass']:.1f} | v_max {res['v_max']:.2f} m/s ({res['v_max']*2.237:.1f} mph) | binding at {fail['v'] if fail else '-'}: {fail['fails'] if fail else '-'}", flush=True)
        if best: print("   stride at v_max:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best["stride"].items() if k not in ("stance", "swing")}, "stance", best["stride"]["stance"], "swing", {k: round(v, 3) for k, v in best["stride"]["swing"].items()})
        for target in (3.3, 10.0, 12.66):                                       # hip+knee rating scale (tau and omega together) needed for a target
            lo, hi = 0.5, 6.0
            if ceiling(B, hi, vs=[target])[0] is None: res["scaling"][str(target)] = None; continue
            for _ in range(10):
                mid = 0.5*(lo + hi)
                if ceiling(B, mid, vs=[target])[0]: hi = mid
                else: lo = mid
            res["scaling"][str(target)] = round(hi, 3); print(f"   rating scale (hip+knee tau and omega) for {target} m/s: x{hi:.2f}", flush=True)
        out[robot] = res
    os.makedirs("../results/sprint", exist_ok=True); json.dump(out, open("../results/sprint/ceiling.json", "w"), indent=1, default=float)
