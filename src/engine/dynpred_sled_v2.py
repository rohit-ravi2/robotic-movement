"""SLED PREDICTOR v2 (track Q4, 2026-10-17): engine.dynpred_sled (T2) + the STOP requirement it lacked.

What T2 got wrong (pre-registered at 1c85c8f, both mu 0.25 cases > 2x optimistic, friction effect in the wrong direction):
the force bookkeeping has no sign error (checked: the sled slides iff s Fx + mu Fz_up >= mu m_s g, i.e. drive >= mu (m_s g - lift);
the foot pyramid uses the model's foot mu 1.0, and the foot-floor pair resolves to 1.0). The error is an omitted role of the sled's
friction. T2 treated it ONLY as the load to overcome (mu m_s g), so halving mu doubles the mass. But with the hand attached and the
feet planted for the whole horizon, the sled must also be STOPPED inside the arm's reach and HELD until the end: the gate fails any
rollout whose feet slide, and an attached sled that keeps moving drags the robot. Sliding friction is the sled's only brake, and
static friction is the anchor that holds the leaning robot afterwards. A low-friction, heavy sled carries the same breakaway force
mu m_s g but much more momentum per unit of drive, so it needs more time and more reach to accelerate AND stop; this is where the
mass enters at fixed mu m_s g, and it enters with the correct sign (lower mu -> heavier sled for the same force -> slower stroke).

Model (spec only; no constant fitted to a found displacement; every T2 term kept):
  poses   the T2 family (rigid ankle lean, waist pitch, crouch; right-arm IK to the handle displaced by s), extended past the 0.2 m
          stroke to the last displacement with a reachable, admissible pose (s_reach).
  drive   F(s)   = max over the best poses of (s Fx + mu Fz) at sled mass m_s (T2 LP, CoP at the sole middle).
  brake   D(s)   = MIN of (s Fx + mu Fz) at the SAME pose (posture cannot jump; forces can): the least hand force the robot's
          balance needs from the sled there. The sled decelerates at b(s)/M_eff, b = mu m_s g - D(s); it can be held at rest at s iff
          b(s) >= 0 (the anchor condition).
  stroke  start at rest; accelerate on ex(s) = F - mu m_s g (break-away ex(0) > 0, momentum may carry through weak stretches), switch
          at s1, decelerate on b(s) until the energy E is spent at s_stop; require 0.2 <= s_stop <= s_reach and b(s_stop) >= 0.
  time    t = sum ds / v, v = sqrt(2 E / M_eff), M_eff = m_s + kappa^2 M (kappa = robot COM travel per hand travel on the chosen
          poses); require t <= horizon - t_ramp. t_ramp = 0 (physics) is the prediction; the construction's 1.5 s posture ramp
          (sled_dyn: phase A ramps over 1.5 s) is reported as a variant.
  The predicted maximum is the largest m_s (descending scan, 0.5 kg resolution) meeting all of them.
Interfaces: WELD (as T2 and the search)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from engine import leverage_dyn as LD, dynpred_sled as DS
G = 9.80665


def build_family(S, sgn, s_max=0.40, ds=0.025, phis=np.arange(-30, 30.1, 5.0), waists=(-0.3, 0.0, 0.3, 0.5), crouches=(0.0, 0.25, 0.5)):
    """mu-independent pose family per hand displacement (cache it across mu). Stops at the first s with no reachable pose."""
    fam = []
    for sd in np.round(np.arange(0.0, s_max + 1e-9, ds), 4):
        f = LD.family(S, sd, sgn, phis, waists, crouches)
        if not f: break
        fam.append((float(sd), f))
    return fam


def predict_v2(S, sgn, mu, fam, stroke=0.2, horizon=4.0, t_ramps=(0.0, 1.5), top=6, m_start=None, verbose=False):
    m = S["m"]; rb = S["robot_bodies"]; fg = S["foot_geoms"]; hb = S["hand_body"]; M = float(sum(m.body_mass[b] for b in rb))
    cap = lambda q, tgt, **k: DS.capacity_dyn(m, q, hb, tgt, sgn, mu, rb, fg, interface="weld", cop_center=True, **k)
    per_s = []
    for sd, f in fam:
        rows = [(prm, q, tgt, r) for prm, q, tgt in f for r in [cap(q, tgt)] if r["status"] == "OK"]
        if not rows: break
        rows.sort(key=lambda t: -t[3]["m_s"]); per_s.append((sd, rows[:top]))
    sv = np.array([sd for sd, _ in per_s])
    if len(sv) == 0 or sv[-1] < stroke: return dict(status="STROKE_UNREACHABLE", s_reach=float(sv[-1]) if len(sv) else 0.0)
    m_qs = min(r[0][3]["m_s"] for sd, r in per_s if sd <= stroke + 1e-9)
    cache = {}

    def curves(ms):
        if ms in cache: return cache[ms]
        F, Dm, cx, pose = [], [], [], []
        for sd, rows in per_s:
            bb = None
            for prm, q, tgt, _ in rows:
                r = cap(q, tgt, m_fixed=ms)
                if r["status"] == "OK" and (bb is None or r["drive"] > bb[2]["drive"]): bb = (prm, q, r, tgt)
            if bb is None: F.append(-np.inf); Dm.append(np.inf); cx.append(np.nan); pose.append(None); continue
            rmin = cap(bb[1], bb[3], m_fixed=ms, minimize=True)
            F.append(bb[2]["drive"]); Dm.append(rmin["drive"] if rmin["status"] == "OK" else np.inf); cx.append(bb[2]["com"][0]); pose.append(bb[0])
        cache[ms] = (np.array(F), np.array(Dm), np.array(cx), pose); return cache[ms]

    def feasible(ms, t_ramp):
        F, Dm, cx, pose = curves(ms); fr = mu*G*ms; ex = F - fr; b = fr - Dm
        if not np.isfinite(ex[0]) or ex[0] <= 0: return False, dict(why="cannot break away: drive < friction at s=0")
        best = None; dsv = np.diff(sv)
        for i1 in range(1, len(sv)):
            # accelerate on [0, s_i1]
            E = np.concatenate([[0.0], np.cumsum(0.5*(ex[1:i1+1] + ex[:i1])*dsv[:i1])])
            if not np.isfinite(E).all() or (E[1:] <= 0).any(): break          # momentum exhausted while still driving: later switches fail too
            Es = [E[-1]]; s_stop = None
            for k in range(i1, len(sv) - 1):
                if not (np.isfinite(b[k]) and np.isfinite(b[k+1])): break
                dE = 0.5*(b[k] + b[k+1])*dsv[k]; En = Es[-1] - dE
                if En <= 0:
                    frac = Es[-1]/max(dE, 1e-12); s_stop = sv[k] + frac*dsv[k]; Es.append(0.0); break
                Es.append(En)
            if s_stop is None or s_stop < stroke - 1e-9: continue
            kstop = int(np.searchsorted(sv, s_stop)); kstop = min(kstop, len(sv) - 1)
            if b[kstop] < 0: continue                                            # cannot hold the sled at rest there (anchor)
            Eall = np.concatenate([E, Es[1:]]); sall = np.concatenate([sv[:i1+1], sv[i1+1:i1+len(Es)-1], [s_stop]])[:len(Eall)]
            kap = abs(np.nanmax(cx[:kstop+1]) - np.nanmin(cx[:kstop+1]))/max(s_stop, 1e-9); Meff = ms + kap**2*M
            v = np.sqrt(np.maximum(2*Eall/Meff, 0.0)); dss = np.diff(sall)
            vm = 0.5*(v[1:] + v[:-1]); t = float(np.sum(dss[1:]/np.maximum(vm[1:], 1e-9)))
            t += np.sqrt(2*dss[0]/max(ex[0]/Meff, 1e-12))                        # first step from rest: uniform acceleration
            if np.isfinite(t) and (best is None or t < best["t"]):
                best = dict(t=t, s_switch=float(sv[i1]), s_stop=float(s_stop), kappa=float(kap), M_eff=float(Meff), b_stop=float(b[kstop]),
                            pose_start=pose[0], pose_switch=pose[i1], pose_stop=pose[kstop])
        if best is None: return False, dict(why="cannot stop and hold the sled at rest inside the reach after >= 0.2 m")
        ok = best["t"] <= horizon - t_ramp
        if not ok: best["why"] = f"too slow: {best['t']:.2f} s > {horizon - t_ramp:.2f} s"
        return ok, best

    out = dict(status="OK", direction=("push" if sgn > 0 else "pull"), mu_sled=mu, M_robot=M, s_reach=float(sv[-1]), m_quasistatic_total=float(m_qs), m_breakaway_total=float(per_s[0][1][0][3]["m_s"]),
               friction_force_at_qs_N=float(mu*G*m_qs), horizon=horizon, top_poses=top)
    hi = float(np.ceil(m_start if m_start is not None else per_s[0][1][0][3]["m_s"]))   # break-away bound: the s = 0 capacity
    for tr in t_ramps:
        found = None; why_above = None; ms = hi
        while ms >= 1.0:
            ok, det = feasible(ms, tr)
            if verbose: print(f"  mu {mu} t_ramp {tr} ms {ms}: {ok} {det.get('why') if not ok else round(det['t'], 2)}", flush=True)
            if ok:
                found = (ms, det)
                ok2, det2 = feasible(ms + 0.5, tr)
                if ok2: found = (ms + 0.5, det2)
                break
            why_above = det.get("why"); ms -= 1.0
        key = "t_ramp_%g" % tr
        out[key] = dict(m_max_total=found and found[0], payload_max=found and found[0] - 1.0, at_max=found and found[1], why_heavier_fails=why_above)
    out["m_max_total"] = out["t_ramp_0"]["m_max_total"]; out["m_max_total_ramp1.5"] = out.get("t_ramp_1.5", {}).get("m_max_total")
    return out
