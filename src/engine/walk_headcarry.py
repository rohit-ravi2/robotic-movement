"""WALKING WHILE HEAD-CARRYING (M6, 2026-10-19). Stock G1H (G1 + Dex3 hands, Menagerie), simulator only.

Construction (body + objective + support contract -> scaffold -> search -> gates), assembled from two existing constructions with no
per-task choreography:
  body      = stock G1H (unitree_g1/scene_hands_palm.xml) + a FREE load (pot cylinder r 0.13 h 0.18 m, or box 0.30 x 0.30 x 0.16 m) of
              mass m in a world scene; nothing welded, no applied force on the load.
  objective = walk at the M5 command speed for the M5 cycle count (>= 3 m net while the load stays on the head), all M5 gates + the
              head-carry gates (load COM within the 7 cm head patch, not dropped, tilt <= 45 deg, the head carries >= 50 % of its weight,
              no load-floor contact).
  scaffold  = the M5 walking scaffold (engine.walk.scaffold: periodic LIPM COM, quintic swing, IK) built on the G1H with the ARMS AT THE
              HEAD-CARRY GRIP POSTURE (engine.headcarry.grip_geometry + clear_ik); the arm counter-swing of the M5 scaffold is replaced
              by the constant grip posture (the arms are busy). Stride and gains warm-started from the M5 gait library entry.
  control   = the M5 walking controller (engine.walk._rollout, copied below with the marked M6 edits) and the head-carry ARM-BALANCE LAW
              (engine.headcarry.law_torques: object COM error / tilt -> wrench -> capacity-weighted palm allocation + squeeze -> J^T),
              evaluated every 2 ms (the head-carry control period) and added to the arm joints' feedforward together with the arm
              gravity feed-forward (qfrc_bias). The balance (support force / CoP / foothold) reads the WORLD COM of robot + load and the
              total mass M + m (the head-carry stand's rule); the COM reference is the scaffold's plus the load's constant offset at
              placement; the attitude inertia includes the load.
  search    = CMA-ES (seed >= 1) over the M5 walk block (amp, tscale, lean + 13 gains) and the law's gain block (wn scale, zeta scale,
              squeeze s_N, N_min scale, waist pitch offset, arm servo frequency scale); grip geometry fixed at the warm start.
  gates     = engine.walk gates (both dt) + head-carry gates (above); robustness on fresh plants with >= 4 perturbations.
usage (from src/):
  python engine/walk_headcarry.py predict
  python engine/walk_headcarry.py replay pot 10 [law 0|1] [dt]
  python engine/walk_headcarry.py search pot 10 iters pop workers seed [warm_json]
  python engine/walk_headcarry.py validate <result json> [workers]
"""
import os, sys, json, time, hashlib, pickle
os.environ.setdefault("SPRINT_LAG", "0.010")
os.environ.setdefault("SPRINT_RATINGS", "published")
os.environ.setdefault("STOCK_CONTACT", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
import branch_i as BI, plant as P, throw as TH
from engine import core, sprint as SP, sprint_ceiling as SC, walk as W, headcarry as HC
from engine.walk import (N_TRANS, N_PER, G, LAG, K_V, K_XI_, OMA_F, COP_MARGIN, TD_DEPTH, TIMED, YAW_FULL, AVAIL_LOAD, TD_ACCEPT, DS_LOAD, DBG, DBG2,
                         PROJ, HIP_FULL, TORSION_MU, CLEAR, TD_FROM, SLIP_MAX, FLIGHT_CONFIRM, PER_TOL, SPEED_TOL, HEADING_TOL,
                         initial_state, rotz, rpy_mat, wrap, cross, _chord, score as walk_score)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = f"{ROOT}/results/movements/M6_walking_headcarry"
GAIT = {0.75: f"{ROOT}/results/movements/M5_walking/G1/search_v0.75_r2b.json", 0.6: f"{ROOT}/results/movements/M5_walking/G1/search_v0.6_r1c.json"}
LAW_WARM = {"pot": f"{ROOT}/results/overnight/K2/refine.json"}
CTRL_DT = 0.002
E_PATCH = HC.HEAD_PATCH_R; TILT_GATE = HC.TILT_GATE; HEAD_FRAC_MIN = 0.5
ARMSDOWN = os.environ.get("M6_ARMSDOWN", "0") == "1"   # diagnostic: arms down with the M5 counter-swing (code-path check against engine.walk)
COUNTER = os.environ.get("M6_COUNTER", "waist")   # yaw counter-rotation actuator while the arms carry: "waist" (default) or "none" (ablation)
NOPLACE = os.environ.get("M6_NOPLACE", "0") == "1"   # diagnostic: the load stays parked (walk with the arms at the grip posture, unloaded)
LOAD_STOP = os.environ.get("M6_LOADSTOP", "1") == "1"   # 0: diagnostic only (walk judged after the load is lost)
LAW_IDX = (2, 3, 4, 5, 9, 13)        # headcarry.decode indices searched here: wn_s, zeta_s, sN, Nmin_s, waist, warm_s (geometry 6,7,8,12 fixed)


def register_g1h():
    """G1H = the G1 with Dex3 hands: the same leg / centre / arm joint names and chains (registration only, no hardware change)."""
    SP.LEG.setdefault("G1H", SP.LEG["G1"]); SP.CENTRE.setdefault("G1H", SP.CENTRE["G1"]); SP.ARM.setdefault("G1H", SP.ARM["G1"])
    SC.CHAINS.setdefault("G1H", SC.CHAINS["G1"]); SC.ANKLE.setdefault("G1H", SC.ANKLE["G1"]); SP.VENDOR_MAX.setdefault("G1H", SP.VENDOR_MAX["G1"])
    if "G1H" not in SP._RT: SP._RT["G1H"] = dict(SP.rating_table("G1"))
register_g1h()


def ratings_table(S):
    """The torque rating actually applied per driven joint (N m) and the held joints' MJCF force limits."""
    out = {t: round(float(S["jarr"]["tau"][i]), 2) for i, t in enumerate(S["types"])}
    m = S["m"]
    for (a, dof, qad, tau, kp, kv) in S["held"]:
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a][0]))
        if "hand" not in jn: out[jn + " (held servo)"] = round(float(tau), 2)
    return out


# ====================================================================== plant
_PL = {}
def plant(obj, mass, dt, uid=0):
    """Sprint-style torque plant of the stock G1H in a world scene with the free load parked 3 m to the side (it is placed on the head at
    t = 0 of every rollout). Driven set = M5's G1 set (legs, waist, shoulder pitch/roll, elbow), published ratings; the other joints
    (shoulder yaw, wrists, fingers) are the project's held position servos. uid > 0 builds a FRESH copy (validation)."""
    key = (obj, round(float(mass), 4), dt, uid)
    if key in _PL: return _PL[key]
    from engine import world as Wd
    o = HC.OBJ[obj]; hw, hh = HC.half(obj)
    ob = Wd.obj("load", kind=o["kind"], size=o["size"], mass=float(mass), pos=(0.0, 3.0, hh + 0.01), graspable=False, rgba=(0.75, 0.45, 0.2, 1))
    path = Wd.build_scene("G1H", objects=[ob])
    drv = SP.driven("G1H")
    S0 = TH.setup("G1H", mjcf=path, ball=False, mass=0.0, driven=drv)
    S = TH.setup("G1H", mjcf=path, start=np.array(S0["q0"]) + 1e-9*(1 + len(_PL) + 7*uid), ball=False, mass=0.0, driven=drv)
    S["jarr"] = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in S["jarr"].items()}
    for i, t in enumerate(S["types"]):
        c = SP.rating_table("G1H").get(t)
        if c is not None: S["jarr"]["tau"][i] = min(S["jarr"]["tau"][i], c)
    BI.apply_dt(S, dt); S["scene"] = path; S["obj"] = obj; S["mass_load"] = float(mass)
    W.nominal_pose(S); _PL[key] = S; return S


def grip(S, gp, gap):
    """Head-carry grip posture on the settled stance (engine.headcarry: palm IK on the load's sides, then the clearance gap)."""
    m = S["m"]; bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "load"); ja = int(m.jnt_qposadr[m.body_jntadr[bid]])
    q0 = np.array(S["q_settled"]); c_w, tg = HC.grip_geometry(m, q0, "head", S["obj"], gp)
    qa, ik_err, palm = HC.clear_ik(m, q0, tg, bid, ja, c_w, gap); return qa, ik_err, palm


def body(S):
    """engine.walk.walk_body on THIS plant (arms at the grip posture): sprint body spec (rating table) + posture quantities."""
    B = dict(SP.body("G1H")); m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d); c = SC.CHAINS["G1H"]
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, c["hipb"]); ab = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, c["ankleb"])
    B["M"] = float(sum(m.body_mass[b] for b in S["robot_bodies"]))
    B["dcx"] = float(d.subtree_com[1][0] - d.xpos[hb][0]); B["dcz"] = float(d.subtree_com[1][2] - d.xpos[hb][2]); B["w"] = float(abs(d.xpos[ab][1] - d.subtree_com[1][1]))
    L1, L2 = B["L1"], B["L2"]; B["I_hip"] = B["m_thigh"]*(L1*L1/3) + B["m_shank"]*((L1 + 0.4*L2)**2 + L2*L2/12)
    B["I_knee"] = B["m_shank"]*((0.4*L2)**2 + L2*L2/12); B["mr_hip"] = B["m_thigh"]*L1/2 + B["m_shank"]*(L1 + 0.4*L2)
    zc_ = [d.contact[i].pos[2] for i in range(d.ncon) if d.contact[i].geom1 in S["feet"][0] or d.contact[i].geom2 in S["feet"][0]]
    if zc_: B["hf"] = float(d.xpos[ab][2] - np.mean(zc_))
    return B


def arm_cols(S):
    m = S["m"]; types = S["types"]
    armc = [i for i, t in enumerate(types) if any(k in t for k in ("shoulder", "elbow", "wrist"))]
    return armc


def scaffold(S, B, st):
    """engine.walk.scaffold on the grip-posture plant, then the arm columns are set to the constant grip posture (no counter-swing)
    and their scaffold feedforward removed (the rollout feeds the arms gravity + the law). Cached on disk by (stride, arm posture)."""
    import inspect
    armc = arm_cols(S); qad = S["jarr"]["qad"]
    src = hashlib.sha1((inspect.getsource(W.scaffold) + inspect.getsource(scaffold)).encode()).hexdigest()[:8]
    key = hashlib.sha1(json.dumps([round(st["v"], 4), round(st["s"], 4), round(st["delta"], 4), round(st["hr"], 4), CLEAR, LAG, src,
                                   np.round(np.asarray(S["q_settled"])[:S["m"].nq - 7], 4).tolist()], default=float).encode()).hexdigest()[:12]
    pth = f"{RES}/cache/scaffold_G1H{'_armsdown' if ARMSDOWN else ''}_c{COUNTER}_{key}.pkl"
    if os.path.exists(pth):
        try: return pickle.load(open(pth, "rb"))
        except Exception: pass
    R = W.scaffold(S, B, st)
    q_up = np.array(R["q_up"])
    if not ARMSDOWN:
        for i in armc: R["Q"][:, i] = q_up[i]; R["QD"][:, i] = 0.0
        R["counter"] = counter_rotation(S, R) if COUNTER == "waist" else None
        recompute_id(S, R)
        for i in armc: R["ID"][:, i] = 0.0
    R["arm"] = dict(mode="constant grip posture (head carry)", counter_swing_removed=True)
    os.makedirs(os.path.dirname(pth), exist_ok=True); tmp = pth + f".{os.getpid()}"; pickle.dump(R, open(tmp, "wb")); os.replace(tmp, pth); return R


def counter_rotation(S, R):
    """The M5 scaffold's ANGULAR-MOMENTUM COUNTER-ROTATION rule (cancel the cycle fundamental of the legs' yaw angular momentum about
    the COM), assigned to the WAIST YAW because the arms hold the load: w_wy = -L_z1 / (dL_z/dw_wy); the upper body (torso, arms,
    head) is the counter-rotating inertia. Same rule, same inputs, a different free joint (listed as supplied)."""
    m = S["m"]; types = S["types"]; N, res = R["N"], R["res"]; Q, QD = R["Q"], R["QD"]; VX, VY = R["VR"][:, 0], R["VR"][:, 1]
    wy = types.index("waist_yaw_joint"); dd = mujoco.MjData(m); q_up = np.array(R["q_up"])
    def angmom(k, extra=0.0):
        dd.qpos[:] = S["q_settled"]; dd.qpos[:3] = 0; dd.qpos[3:7] = [1, 0, 0, 0]; dd.qpos[S["jarr"]["qad"]] = Q[k]
        vv = np.zeros(m.nv); vv[0] = VX[k]; vv[1] = VY[k]; vv[S["jarr"]["dof"]] = QD[k]; vv[S["jarr"]["dof"][wy]] += extra
        dd.qvel[:] = vv; mujoco.mj_forward(m, dd); mujoco.mj_subtreeVel(m, dd); return float(dd.subtree_angmom[1][2])
    Lz = np.array([angmom(k) for k in range(N)])
    b = float(np.mean([angmom(k, 1.0) - Lz[k] for k in range(0, N, max(1, N//20))]))
    F_ = np.fft.rfft(Lz); F_[0] = 0; F_[2:] = 0; Lz1 = np.fft.irfft(F_, N); w_ = -Lz1/b; ang = np.cumsum(w_)*res; ang -= ang.mean()
    Q[:, wy] = q_up[wy] + ang; R["QD"] = (np.roll(Q, -1, 0) - np.roll(Q, 1, 0))/(2*res)
    return dict(joint="waist_yaw_joint", Lz_ptp=float(np.ptp(Lz)), Lz1_ptp=float(np.ptp(Lz1)), dLz_dw=b, ang_ptp=float(np.ptp(ang)), rate_max=float(np.abs(w_).max()))


def recompute_id(S, R):
    """engine.walk.scaffold's inverse-dynamics feedforward, recomputed for the edited reference (stance legs zero)."""
    m = S["m"]; types = S["types"]; nT = len(types); N, res = R["N"], R["res"]; Q, QD = R["Q"], R["QD"]
    VX, VY, AX, AY = R["VR"][:, 0], R["VR"][:, 1], R["AR"][:, 0], R["AR"][:, 1]
    wq = max(1, int(round(0.5*SP.TC_LAG/res))); kq = np.ones(wq)/wq
    QDs = np.stack([np.convolve(np.concatenate([QD[-wq:, i], QD[:, i], QD[:wq, i]]), kq, "same")[wq:-wq] for i in range(nT)], 1)
    QDD = (np.roll(QDs, -1, 0) - np.roll(QDs, 1, 0))/(2*res); ID = np.zeros((N, nT)); qa_, dof_ = S["jarr"]["qad"], S["jarr"]["dof"]; dd = mujoco.MjData(m)
    for k in range(N):
        dd.qpos[:] = S["q_settled"]; dd.qpos[:3] = [0, 0, 5.0]; dd.qpos[3:7] = [1, 0, 0, 0]; dd.qpos[qa_] = Q[k]
        dd.qvel[:] = 0; dd.qvel[0] = VX[k]; dd.qvel[1] = VY[k]; dd.qvel[dof_] = QD[k]; dd.qacc[:] = 0; dd.qacc[0] = AX[k]; dd.qacc[1] = AY[k]; dd.qacc[dof_] = QDD[k]
        mujoco.mj_inverse(m, dd); ID[k] = dd.qfrc_inverse[dof_]
        for li in (0, 1):
            if R["SH"][k, li] > 0 or R["SWP"][k, li] == 0: ID[k, R["legcols"][li]] = 0.0
    R["ID"] = ID


# ====================================================================== problem
def law_params(zl, base_z):
    """Full 14-vector of engine.headcarry.decode: the warm (K2) vector with the searched LAW_IDX entries replaced by zl."""
    z = np.array(base_z, float)
    if zl is not None:
        for k, i in enumerate(LAW_IDX): z[i] = float(np.clip(zl[k], -1, 1))
    return HC.decode(z, "head"), z


def gait_record(v):
    g = json.load(open(GAIT[v])); return g


class Carry:
    def __init__(self, obj, mass, v=0.75, dt_list=(core.DT_OPT, core.DT_CHECK), uid=0, law_base=None):
        self.obj, self.mass, self.v = obj, float(mass), float(v); g = gait_record(v); self.gait = g
        self.st = dict(g["stride"]); self.nwalk = len(g["z"]); self.z_walk0 = np.array(g["z"], float)
        lb = law_base if law_base is not None else json.load(open(LAW_WARM["pot"]))["pot_lag20ms"]["z"]
        self.law_base = np.array(lb, float); P_, _ = law_params(None, self.law_base); self.P0 = P_
        self.S = {}; self.R = None; self.uid = uid
        for dt in dt_list: self.S[dt] = self._prep(plant(obj, mass, dt, uid))
        S1 = self.S[dt_list[0]]; self.B = body(S1); self.R = scaffold(S1, self.B, self.st)
    def _prep(self, S):
        if S.get("_grip_ready"): return S
        if ARMSDOWN: S["qa"] = np.array(S["q_settled"]); S["grip_ik_err"] = None; S["grip_palm"] = None; S["_grip_ready"] = True; return S
        qa, ik_err, palm = grip(S, self.P0["gp"], self.P0["gap"]); m = S["m"]
        arm_q = [int(m.jnt_qposadr[j]) for s in "rl" for j in HC.parts(m)[2][s]]
        q = np.array(S["q_settled"]); q[arm_q] = qa[arm_q]; S["q_settled"] = q; S["qa"] = qa; S["grip_ik_err"] = ik_err; S["grip_palm"] = palm; S["_grip_ready"] = True
        return S
    def n(self): return 16 + len(LAW_IDX)
    def split(self, x):
        """search vector x (16 walk: amp/tscale/lean + 13 gains; 6 law) -> full walk z, law z"""
        zw = self.z_walk0.copy(); nS = SP.n_params(next(iter(self.S.values()))); x = np.asarray(x, float)
        zw[nS - 3:nS] = x[:3]; zw[nS:nS + W.N_GAIN] = x[3:16]; return zw, x[16:16 + len(LAW_IDX)]
    def x0(self, law_x=None):
        nS = SP.n_params(next(iter(self.S.values()))); zw = self.z_walk0
        lx = np.array([self.law_base[i] for i in LAW_IDX]) if law_x is None else np.asarray(law_x, float)
        return np.concatenate([zw[nS - 3:nS], zw[nS:nS + W.N_GAIN], lx])
    def run(self, x, dt=core.DT_OPT, law=True, **kw):
        zw, zl = self.split(x); p = W.unpack(self.S[dt], zw); PL, zfull = law_params(zl, self.law_base)
        return hc_rollout(self.S[dt], self.R, p, self.law_setup(self.S[dt], PL), law=law, **kw)
    def law_setup(self, S, PL):
        """Head-carry law constants on this plant (engine.headcarry's gain rule, from the object and the arm capacity at the grip)."""
        m = S["m"]; H, wb, arm = HC.parts(m); bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "load"); hw, hh = HC.half(self.obj)
        d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d); h_loc = HC.head_local(m, d, H)
        d2 = mujoco.MjData(m); d2.qpos[:] = S["qa"]; mujoco.mj_forward(m, d2)
        aid = {s: [[a for a in range(m.nu) if int(m.actuator_trnid[a][0]) == j][0] for j in arm[s]] for s in "rl"}
        dofs = {s: [int(m.jnt_dofadr[j]) for j in arm[s]] for s in "rl"}
        Fcap = min(HC.arm_capacity(m, d2, wb, arm, s, d2.xpos[wb[s]] + d2.xmat[wb[s]].reshape(3, 3) @ HC.PALM_C[s]) for s in "rl")
        lam = np.sqrt(G/hh); wn_cap = np.sqrt(2*Fcap/(self.mass*E_PATCH)); wn0 = min(2*lam, wn_cap); wn = wn0*PL["wn_s"]
        types = S["types"]; J = S["jarr"]; a2col = {int(J["aid"][i]): i for i in range(len(types))}
        held_arm = [(a_, dof, qad, kp_) for (a_, dof, qad, tau, kp_, kd_) in S["held"] if a_ in aid["r"] + aid["l"]]
        wp = [i for i, t in enumerate(types) if t == "waist_pitch_joint"]
        return dict(mass=self.mass, obj=self.obj, bid=bid, tb=H["torso"], h_loc=h_loc, hh=hh, wb=wb, aid=aid, dofs=dofs, Iobj=np.array(m.body_inertia[bid]),
                    wn=wn, wn_rule=wn0, lam=lam, wn_cap=wn_cap, F_cap=Fcap, zeta=1.0*PL["zeta_s"], sN=PL["sN"], Nmin=0.02*self.mass*G*PL["Nmin_s"], w_arm=min(wn*PL["warm_s"], 0.5/LAG),
                    waist=PL["waist"], wpc=wp, armc=arm_cols(S), a2col=a2col, held_arm=held_arm, qa=np.array(S["qa"]), law_params=PL)


# ====================================================================== rollout (engine.walk._rollout + the marked M6 edits)
def hc_rollout(S, R, p, L, n_cyc=None, turn=0.0, record=False, shove=None, noise=None, lag_plant=None, friction=None, mass_noise=None, law=True, load_scale=1.0):
    """mass_noise: robot bodies' mass/inertia x N(1, sd) (the load is not scaled); load_scale: the LOAD's true mass x this (the
    controller keeps the nominal); both restored after the run."""
    m = S["m"]; bid = L["bid"]; bm0 = m.body_mass.copy(); bi0 = m.body_inertia.copy(); changed = False
    if mass_noise is not None:
        rng = np.random.default_rng(mass_noise[0]); f = np.clip(rng.normal(1.0, mass_noise[1], m.nbody), 1 - 2*mass_noise[1], 1 + 2*mass_noise[1])
        for b in S["robot_bodies"]: m.body_mass[b] *= f[b]; m.body_inertia[b] *= f[b]
        changed = True
    if load_scale != 1.0: m.body_mass[bid] *= load_scale; m.body_inertia[bid] *= load_scale; changed = True
    if changed: mujoco.mj_setConst(m, mujoco.MjData(m))
    try: return _hc_rollout(S, R, p, L, n_cyc, turn, record, shove, noise, lag_plant, friction, law)
    finally:
        if changed: m.body_mass[:] = bm0; m.body_inertia[:] = bi0; mujoco.mj_setConst(m, mujoco.MjData(m))


def _place_load(m, d, L):
    """M6: the load is set down on the head top at t = 0, at rest RELATIVE to the head (a helper's placement; initial condition)."""
    tb = L["tb"]; Rt = d.xmat[tb].reshape(3, 3); ht = d.xpos[tb] + Rt @ L["h_loc"]; c = np.array([ht[0], ht[1], ht[2] + L["hh"] + 0.0005])
    yaw = np.arctan2(Rt[1, 0], Rt[0, 0]); bid = L["bid"]; ja = int(m.jnt_qposadr[m.body_jntadr[bid]]); jv = int(m.jnt_dofadr[m.body_jntadr[bid]])
    d.qpos[ja:ja+3] = c; d.qpos[ja+3:ja+7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
    vt = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, tb, vt, 0)
    d.qvel[jv:jv+3] = vt[3:] + np.cross(vt[:3], c - d.xpos[tb]); Rl = rotz(yaw); d.qvel[jv+3:jv+6] = Rl.T @ vt[:3]
    mujoco.mj_forward(m, d)


def _comb_inertia(m, d, S, bid, cc):
    I_c = np.zeros(3)
    for bb in list(S["robot_bodies"]) + [bid]:
        if bb == 0: continue
        r = d.xipos[bb] - cc; Rb = d.ximat[bb].reshape(3, 3); Ic = Rb @ np.diag(m.body_inertia[bb]) @ Rb.T
        for ax in range(3): I_c[ax] += float(Ic[ax, ax] + m.body_mass[bb]*(r @ r - r[ax]**2))
    return I_c


def _hc_rollout(S, R, p, L, n_cyc=None, turn=0.0, record=False, shove=None, noise=None, lag_plant=None, friction=None, law=True):
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
    # ---- M6: load placed on the head; combined-COM offset (heading frame) and attitude inertia about the combined COM
    if not NOPLACE: _place_load(m, d, L)
    bid = L["bid"]; tb = L["tb"]; mujoco.mj_subtreeVel(m, d)
    yaw0 = rpy_mat(d.xmat[1].reshape(3, 3))[2]; dlt = rotz(yaw0).T @ (d.subtree_com[0] - d.subtree_com[1]); I_cL = _comb_inertia(m, d, S, bid, d.subtree_com[0].copy())
    armc = np.array(L["armc"]); a2col = L["a2col"]; arm_aids = L["aid"]["r"] + L["aid"]["l"]; drv_arm = [(a, a2col[a]) for a in arm_aids if a in a2col]
    arm_dof_drv = np.array([J["dof"][c_] for (a, c_) in drv_arm]); arm_col_drv = np.array([c_ for (a, c_) in drv_arm]); held_arm = L["held_arm"]
    tm_all = np.where(m.actuator_forcelimited > 0, np.abs(m.actuator_forcerange[:, 1]), np.inf); jpL = np.zeros((3, m.nv)); jrL = np.zeros((3, m.nv))
    n_ctrl = max(1, int(round(CTRL_DT/m.opt.timestep))); tau_law = np.zeros(m.nu); ff_arm = np.zeros(len(drv_arm)); hgeo = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == tb)
    ogeo = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == bid); load_reason = None; t_load = None; LS = dict(e=[], tilt=[], hf=[], emax=0.0, tiltmax=0.0, armu=[]); cfv = np.zeros(6)
    c_place = d.subtree_com[1][:2].copy(); lgeo_floor = False
    for (a_, dof, qad, tau, kp_, kd_) in S["held"]: d.ctrl[a_] = q0[qad]
    for (a_, dof, qad, kp_) in held_arm: d.ctrl[a_] = L["qa"][qad]
    N, res, ts, amp, lean = R["N"], R["res"], p["tscale"], p["amp"], p["lean"]; T, Tds, Tc = R["T"], R["Tds"], R["Tc"]; tsw = T - Tds
    Q, QD, ID, JI, REL, FT, SH, SWP, CR, VR, AR, PX, PY = (R[k] for k in ("Q", "QD", "ID", "JI", "REL", "FT", "SH", "SWP", "CR", "VR", "AR", "PX", "PY"))
    KN = SP.knot_table(S, p["x"], N); nadv = int(round(LAG/(res*ts)))
    gn = p.get("g", {}); KV = np.array([gn.get("kvx", K_V[0]), gn.get("kvy", K_V[1])]); OMAF = gn.get("oma", OMA_F); COPM = gn.get("copm", COP_MARGIN); TDD = gn.get("tdd", TD_DEPTH)
    OML = gn.get("oml", SP.OM_LEG); timed = gn.get("timed", TIMED); yawfull = gn.get("yawfull", YAW_FULL); AVL = gn.get("avail", AVAIL_LOAD); TDA = gn.get("td_accept", TD_ACCEPT)
    CLIPF = gn.get("clipf", 1.0); MINW = gn.get("minw", 0.5)
    ieff = R["ieff"]; omj = np.array([OML if any(k_ in t for k_ in ("hip", "knee", "ankle")) else SP.OM_UP for t in types]); omj[armc] = (omj[armc] if ARMSDOWN else L["w_arm"]); kp = omj**2*ieff; kd = 2*omj*ieff   # M6: arm servo frequency = the head-carry rule w_arm
    legcols = [np.array(c) for c in R["legcols"]]; Mtot = R["M"] + L["mass"]; Mg = Mtot*G; om = R["om"]   # M6: robot + load
    K_XI = gn.get("kxi", K_XI_); om_z = min(1.0/SP.TC_LAG, 0.5/LAG); om_att = OMAF*om; Ka, Da = om_att**2, 2*om_att; Kz, Dz = om_z**2, 2*om_z
    lagp = LAG if lag_plant is None else lag_plant; alpha = 1.0 - np.exp(-dt/lagp); act = ID[int(ph0/res) % N]/tau_v
    ankb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in BI.ROBOTS[S["robot"]]["foot_bodies"]]   # [left, right]
    fb = [ankb[1], ankb[0]]                                                                                     # per leg li: 0 right, 1 left
    feet = S["feet"]; fgeo = [feet[1], feet[0]]
    cg = [g for g in range(m.ngeom) if m.geom_bodyid[g] in S["robot_bodies"] and g not in S["foot_geoms"] and m.geom_group[g] == 3] if S["robot"] in ("G1", "G1H") else []
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
        mujoco.mj_subtreeVel(m, d); c = d.subtree_com[0].copy(); vc = d.subtree_linvel[0].copy()   # M6: WORLD COM (robot + load)
        Rp = d.xmat[1].reshape(3, 3); r_, p_, y_ = rpy_mat(Rp); wW = Rp @ d.qvel[3:6]
        qn = d.qpos[J["qad"]].copy(); wv = d.qvel[J["dof"]].copy()
        # lag-aware: feedback on the state predicted one lag ahead
        qn_p = qn + wv*LAG; c_p = c + vc*LAG; att_p = np.array([r_, p_, y_]) + wW*LAG
        psi = psi_ref(phi); Rz = rotz(anc["psi"])
        cyc_off = np.array([kca*2*amp*R["s"], 0.0])
        cr_rob = to_world(np.array([CR[kk, 0]*amp, CR[kk, 1]]) + cyc_off, phi); vr = Rz[:2, :2] @ np.array([VR[kk, 0]*amp/ts, VR[kk, 1]/ts])
        cr_xy = cr_rob + Rz[:2, :2] @ dlt[:2]                                    # M6: combined-COM reference = scaffold COM + the load's offset at placement
        zr = CR[kk, 2] + dlt[2]; pr_xy = to_world(np.array([PX[kk]*amp, PY[kk]]) + cyc_off, phi)
        # ---------------- support force: DCM tracking through the CoP (p_cmd = p_ref + K_XI (xi - xi_ref)), LIPM force M om^2 (c - p_cmd)
        xi_e = (c_p[:2] + vc[:2]/om) - (cr_xy + vr/om); p_cmd = pr_xy + K_XI*xi_e
        att_err = np.array([att_p[0], att_p[1] - lean, wrap(att_p[2] - psi)]); Mc = I_cL*(-Ka*att_err) - Da*I_cL*wW
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
        if not ARMSDOWN: qt[armc] = R["q_up"][armc]; qdt[armc] = 0.0; qt[L["wpc"]] += L["waist"]          # M6: arms at the grip posture; law's waist pitch offset
        pel_ref_xy = cr_rob - Rz[:2, :2] @ (np.array([CR[kk, 0], CR[kk, 1]]) - R["PEL"][kk][:2])        # pelvis = COM ref - the posture's COM offset
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
        if st_ % n_ctrl == 0:                                                    # M6: arm gravity feed-forward + head-carry law, every 2 ms
            tau_law[:] = 0.0
            if law and load_reason is None:
                tau_law[:] = HC.law_torques(m, d, "head", bid, tb, None, L["h_loc"], L["mass"], L["Iobj"], L["wn"], L["zeta"], L["sN"], L["Nmin"], L["wb"], L["aid"], L["dofs"], tm_all, jpL, jrL)
            ff_arm = d.qfrc_bias[arm_dof_drv] + np.array([tau_law[a] for (a, c_) in drv_arm])
            for (a_, dof, qad, kp_) in (held_arm if not ARMSDOWN else []):
                d.ctrl[a_] = L["qa"][qad] + np.clip(d.qfrc_bias[dof] + tau_law[a_], -0.9*tm_all[a_], 0.9*tm_all[a_])/kp_
        if not ARMSDOWN: idr[arm_col_drv] = ff_arm
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
        # ---- M6 head-carry gates: load COM within the head patch, above the head top, tilt, head share of the weight, no floor contact
        if load_reason is None and not NOPLACE:
            Rt_ = d.xmat[tb].reshape(3, 3); ht_ = d.xpos[tb] + Rt_ @ L["h_loc"]; cl_ = d.xipos[bid]; en_ = float(np.linalg.norm(cl_[:2] - ht_[:2]))
            tl_ = float(np.degrees(np.arccos(np.clip(d.xmat[bid][8], -1, 1)))); LS["emax"] = max(LS["emax"], en_); LS["tiltmax"] = max(LS["tiltmax"], tl_)
            if en_ > E_PATCH: load_reason = "LOAD_OFF_PATCH"
            elif cl_[2] < ht_[2]: load_reason = "LOAD_DROPPED"
            elif tl_ > TILT_GATE: load_reason = "LOAD_TILT"
            if st_ % 10 == 0:
                hn_ = 0.0
                for ci in range(d.ncon):
                    cn = d.contact[ci]; g1, g2 = cn.geom1, cn.geom2
                    if g1 in ogeo or g2 in ogeo:
                        oth = g2 if g1 in ogeo else g1
                        if oth == fl: lgeo_floor = True
                        if oth in hgeo: mujoco.mj_contactForce(m, d, ci, cfv); hn_ += abs(cfv[0])*abs(cn.frame[2])
                LS["e"].append(en_); LS["tilt"].append(tl_); LS["hf"].append(hn_/(L["mass"]*G))
                LS["armu"].append(float(np.max(np.abs(d.actuator_force[arm_aids])/tm_all[arm_aids])))
                if lgeo_floor: load_reason = "LOAD_FLOOR"
            if load_reason is not None: t_load = t
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
        if load_reason is not None and LOAD_STOP: break                           # M6: the carry failed; the rest of the walk is not judged
        if d.qpos[2] < 0.6*z0 or abs(rr) > 0.7 or abs(pp - lean) > 0.7 or not np.isfinite(d.qpos[2]):
            fell = True; t_fall = t; break
    if friction is not None: m.geom_friction[fric_g] = fric0
    slip_max = slide_max                                                    # SLIP GATE: sliding of the sole point under the CoP (rolling is not slip)
    E1 = float(d.energy[0] + d.energy[1]); closure = (E1 - E0) - W_act - W_con - W_pas; ctol = max(5.0, 0.02*(abs(W_act) + abs(W_con)))
    if shv is not None: ctol = np.inf                                                       # the external shove's work is not tracked
    t_run = t_fall if fell else (t_load if (load_reason is not None and LOAD_STOP) else nst*dt)
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
    if load_reason is not None: reasons.append(load_reason)
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
    if load_reason is not None: viol += 5.0 + 10.0*(1 - t_load/T_end)
    hf_mean = float(np.mean(LS["hf"])) if LS["hf"] else 0.0
    if load_reason is None and hf_mean < HEAD_FRAC_MIN: reasons.append("HEAD_FRAC"); viol += 1.0 + 5.0*(HEAD_FRAC_MIN - hf_mean)
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
               load=dict(reason=load_reason, t_lost=t_load, t_held=(t_load if load_reason else float(t_run)), e_max=LS["emax"], e_mean=(float(np.mean(LS["e"])) if LS["e"] else None), tilt_max=LS["tiltmax"],
                         head_frac=hf_mean, arm_util_p95=(float(np.percentile(LS["armu"], 95)) if LS["armu"] else None), mass=L["mass"], law=bool(law),
                         dist_held=float(np.linalg.norm(d.subtree_com[1][:2] - c_place)) if load_reason is None else float(np.linalg.norm(d.subtree_com[1][:2] - c_place))),
               W_pos=float(work.sum()), J_per_m=(float(work.sum())/max(dist, 1e-3) if nb > 1 else None), cot=(float(work.sum())/(Mg*max(dist, 1e-3)) if nb > 1 else None), dist=float(dist), dt=dt)
    if record:
        out["td_log"] = td_log; out["slip_log"] = slip_log; out["td_all"] = td_all; out["trace"] = {k: np.asarray(v).tolist() for k, v in trace.items()}; out["bounds"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in b.items()} for b in bounds]
    return out



# ====================================================================== prediction (spec only, before any search)
def predict(objs=("pot", "box"), v=0.75):
    """MAX LOAD CARRIED ON THE HEAD WHILE WALKING, from the spec. Inputs: stock G1H model + published ratings, the M5 stride at v (the
    warm-start gait's spec stride), the object geometry, the 7 cm head-patch gate. Terms (kg, per axis where relevant):
      HC_v2        the standing dynamic head-carry predictor (engine.dynpred_headcarry_v2; Q4), unchanged: D1 arm topple authority,
                   D2 static LP, D3s/D3t ZMP (two feet), D5 waist, D4 friction.
      WALKING ACCELERATION a_ax = max |omega^2 (c - p)| of the stride's periodic LIPM (x: sagittal, y: lateral) -- the head moves with
      the trunk, so the load sees the COM's horizontal acceleration.
      D1w  arm authority: the palms must also cancel the moment of the load's inertial force about the head contact,
           m (g (h - R) e_max / h + a h) <= M_arm                                  (M_arm, R, h from the D1 LP of the T2 predictor)
      D5w  waist: m (g e_max (1 + lambda^2 (z_c - z_w)/g) + a (z_c - z_w)) <= tau_waist - |gravity| - m_up a (z_up - z_w)
           (the upper body's own inertial moment about the waist joint is subtracted from the rating)
      D3s_ss  single support: the load's sustained offset at the patch edge moves the combined COM by m e/(M + m), which the stance
           sole's CoP must cover: m e_max/(M + m) <= min(sole half-span, 2 tau_ankle/((M + m) g))   (one foot, not two)
      D3t_ss  as D3s_ss with the recovery transient through the hip strategy (v2's D3t on one sole)
      WLEG the stride's stance hip/knee utilisation (engine.walk.gait_check) with the body mass M + m: largest m with u_stance <= 1
    Prediction = min of all terms (HC_v2's terms included)."""
    from engine import dynpred_headcarry as D1M, dynpred_headcarry_v2 as D2
    D2.register_objects(); out = dict(v=v, rule=predict.__doc__, objects={})
    S = TH.setup("G1H", ball=False); m = S["m"]; g = gait_record(v); st = g["stride"]
    B = W.walk_body("G1H"); M = float(sum(m.body_mass[b] for b in S["robot_bodies"]))
    r = W.gait_check(B, v, st["s"], st["delta"], st["hr"], full=True, strict=False)
    om = r["om"]; ax = float(np.max(np.abs(om*om*(r["X"] - r["px"])))); ay = float(np.max(np.abs(om*om*(r["Y"] - r["py"]))))
    def wleg_ok(mm):
        B2 = dict(B); B2["M"] = B["M"] + mm; rr = W.gait_check(B2, v, st["s"], st["delta"], st["hr"], strict=False)
        return rr is not None and rr["u_stance"] <= 1.0
    wleg = D2._bisect(wleg_ok)
    # single-support sole polygon (right foot, settled stance) about the foot's own centre
    d = mujoco.MjData(m); d.qpos[:] = S["q_settled"]; mujoco.mj_forward(m, d)
    pts = np.array([d.contact[i].pos[:2] for i in range(d.ncon) if d.contact[i].geom1 in S["feet"][1] or d.contact[i].geom2 in S["feet"][1]])
    ss_half = {"pitch": float(np.ptp(pts[:, 0])/2), "roll": float(np.ptp(pts[:, 1])/2)}
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link"); m_up = float(sum(m.body_mass[b] for b in range(m.nbody) if HC._under(m, [g_ for g_ in range(m.ngeom) if m.geom_bodyid[g_] == b][0], tb)) if False else 0.0)
    up = [b for b in range(m.nbody) if _desc(m, b, tb)]; m_up = float(sum(m.body_mass[b] for b in up)); z_up = float(sum(m.body_mass[b]*d.xipos[b][2] for b in up)/m_up)
    out.update(M_robot=M, a_walk=dict(x=ax, y=ay, source="max |omega^2 (c - p)| of the periodic LIPM at the M5 stride"), stride={k: st[k] for k in ("s", "delta", "hr", "T", "Tds", "zc", "om")},
               WLEG=wleg, single_support_half=ss_half, upper_body=dict(m=m_up, z_com=z_up))
    for obj in objs:
        base = D1M.predict("head", obj, S=S); v2 = D2.predict("head", obj, S=S, static=base["terms"].get("D2_static")); hw, hh = HC.half(obj)
        terms = {f"HC_v2:{k}": (np.inf if v_ is None else float(v_)) for k, v_ in v2["terms"].items()}; axes = {}
        # PUBLISHED RATINGS (the job's rating table): v2 reads the MJCF peak from rb; the walking terms use min(MJCF, URDF effort, 120)
        rt = SP.rating_table("G1H"); pk = lambda jn: float(S["rb"].J[jn]["tau"]); pub = lambda jn: min(pk(jn), rt.get(jn, pk(jn)))
        for ax_ in ("pitch", "roll"):
            v2["waist_avail"][ax_] -= pk(f"waist_{ax_}_joint") - pub(f"waist_{ax_}_joint")
            v2["hip_avail"][ax_] -= sum(pk(f"{s_}_hip_{ax_}_joint") - pub(f"{s_}_hip_{ax_}_joint") for s_ in ("left", "right"))
            v2["tau_ankle"][ax_] = pub(f"left_ankle_{ax_}_joint")
        out.setdefault("ratings_used", {jn: pub(jn) for jn in ("waist_pitch_joint", "waist_roll_joint", "left_hip_pitch_joint", "left_hip_roll_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", "left_knee_joint")})
        out.setdefault("ratings_mjcf", {jn: pk(jn) for jn in ("waist_pitch_joint", "waist_roll_joint", "left_hip_pitch_joint", "left_hip_roll_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", "left_knee_joint")})
        for ax_name, a1 in base["D1_topple"].items():
            lat = ax_name.startswith("roll"); key = "roll" if lat else "pitch"; a = ay if lat else ax; a2 = v2["axes"][ax_name]
            lam2 = a2["lam2"]; zcw = a2["z_c_minus_z_waist"]; zw = v2["z_waist"][key]
            if a1.get("stable"): d1w = np.inf
            else: d1w = a1["M_arm"]/(G*(hh - a1["R"])*E_PATCH/hh + a*hh)
            avail = v2["waist_avail"][key] - m_up*a*(z_up - zw)
            d5w = D2._bisect(lambda mm: mm*(G*E_PATCH*(1 + lam2*zcw/G) + a*zcw) <= avail)
            Mm = M; ta = v2["tau_ankle"][key]; auth = lambda mm: min(ss_half[key], 2*ta/((Mm + mm)*G))
            d3s = D2._bisect(lambda mm: mm*E_PATCH/(Mm + mm) <= auth(mm))
            d3t = D2._bisect(lambda mm: mm*G*E_PATCH*(1 + lam2*v2["z_head"]/G) <= (Mm + mm)*G*auth(mm) + v2["hip_avail"][key])
            axes[ax_name] = dict(a=a, D1w=d1w, D5w=d5w, D3s_ss=d3s, D3t_ss=d3t, waist_avail_walking=avail)
        for t in ("D1w", "D5w", "D3s_ss", "D3t_ss"): terms[t] = min(x[t] for x in axes.values())
        terms["WLEG"] = wleg
        b = min(terms, key=lambda k: terms[k])
        out["objects"][obj] = dict(m_max=round(float(terms[b]), 2), binding=b, terms={k: (None if not np.isfinite(x) else round(x, 2)) for k, x in terms.items()}, axes=axes,
                                   standing_v2=dict(m_max=v2["m_max"], binding=v2["binding"]))
        print(obj, out["objects"][obj]["m_max"], b, out["objects"][obj]["terms"], flush=True)
    os.makedirs(RES, exist_ok=True); json.dump(out, open(f"{RES}/prediction{'' if v == 0.75 else f'_v{v}'}.json", "w"), indent=1, default=float); return out


def _desc(m, b, root):
    while b > 0:
        if b == root: return True
        b = int(m.body_parentid[b])
    return False


# ====================================================================== search, ladder, validation, ablation
SEARCH_DRAW = dict(noise=None, mass_noise=(6, 0.05), qseed=5)          # search-only draw (registered in PREDICTION.md)
PERT = {   # registered in PREDICTION.md before any search; never scored in a search
    "P1": dict(noise="start", seed=9101, vx=0.04),
    "P2": dict(mass_noise=(9102, 0.05), load_scale=1.10),
    "P3": dict(lag_plant=1.15*LAG, friction=0.7),
    "P4": dict(shove=(0.0, 0.1, 4.90, 0.2), n_cyc=N_TRANS + N_PER + 8),
}


# ROBUST-REPAIR search draws (amendment D; search-only, seeds/values distinct from P1-P4): used only when M6_DRAWS lists them
DRAWS = {
    "d_start": dict(noise="start", seed=11, vx=0.03),
    "d_mass": dict(mass_noise=(12, 0.05), load_scale=1.08),
    "d_lagfric": dict(lag_plant=1.10*LAG, friction=0.8),
    "d_shove": dict(shove=(0.0, -0.08, 4.30, 0.2), n_cyc=N_TRANS + N_PER + 8),
}
# REPAIR ROUND 2 (amendment I; general, every rung/object alike): a second search-only draw per perturbation class, SAMPLED from declared
# class distributions with a fixed generator (seed 2027), so the robust score covers each class at two points instead of one.
# Distributions: start joint sd 0.02 with vx ~ U(-0.05, 0.05); robot mass sd 0.05 with true load x U(0.92, 1.12); plant lag x U(1.0, 1.2)
# with mu ~ U(0.7, 0.9); shove 0.08 (M+m) g in a direction ~ U(0, 2 pi) at cycle ~ U(3, 8). Seeds/values are distinct from P1-P4, V1-V4, W1-W4.
_rg = np.random.default_rng(2027); _ang = float(_rg.uniform(0, 2*np.pi))
DRAWS.update({
    "e_start": dict(noise="start", seed=31, vx=round(float(_rg.uniform(-0.05, 0.05)), 4)),
    "e_mass": dict(mass_noise=(32, 0.05), load_scale=round(float(_rg.uniform(0.92, 1.12)), 4)),
    "e_lagfric": dict(lag_plant=round(float(_rg.uniform(1.0, 1.2)), 4)*LAG, friction=round(float(_rg.uniform(0.7, 0.9)), 4)),
    "e_shove": dict(shove=(round(float(0.08*np.cos(_ang)), 4), round(float(0.08*np.sin(_ang)), 4), round(float(_rg.uniform(3.0, 8.0)), 2), 0.2), n_cyc=N_TRANS + N_PER + 8),
})
ROBUST_DRAWS = [x for x in os.environ.get("M6_DRAWS", "").split(",") if x]


def pert_kw(name, nT):
    if name == "nominal": return {}
    if name in DRAWS:
        kw = dict(DRAWS[name])
        if kw.get("noise") == "start":
            rng = np.random.default_rng(kw.pop("seed")); kw["noise"] = dict(q=rng.normal(0, 0.02, nT), vx=kw.pop("vx"))
        return kw
    if name == "search_draw":
        rng = np.random.default_rng(SEARCH_DRAW["qseed"]); return dict(noise=dict(q=rng.normal(0, 0.02, nT), vx=0.0), mass_noise=SEARCH_DRAW["mass_noise"])
    kw = dict(PERT[name])
    if kw.get("noise") == "start":
        rng = np.random.default_rng(kw.pop("seed")); kw["noise"] = dict(q=rng.normal(0, 0.02, nT), vx=kw.pop("vx"))
    return kw


def hc_score(r, v):
    return walk_score(r, v)


_WK = {}
def _winit(obj, mass, v, law_base):
    _WK["C"] = Carry(obj, mass, v, law_base=law_base)


def _weval(x):
    C = _WK["C"]; nT = len(C.S[core.DT_OPT]["types"])
    try:
        r1 = C.run(x, core.DT_OPT)
        if r1["reasons"]: return 300.0 + r1["violation"], summ(r1), summ(r1)
        if ROBUST_DRAWS:      # amendment D: every draw runs once the nominal is gate-clean; lexicographic in the number of failing draws
            worst = hc_score(r1, C.v); fails = []; wr = r1
            for nm in ["search_draw"] + ROBUST_DRAWS:
                rp = C.run(x, core.DT_OPT, **pert_kw(nm, nT))
                if rp["reasons"]: fails.append(rp); continue
                sp = hc_score(rp, C.v)
                if sp > worst: worst, wr = sp, rp
            if fails: return 100.0 + 20.0*len(fails) + float(np.mean([f["violation"] for f in fails])), summ(r1), summ(max(fails, key=lambda f: f["violation"]))
            if worst >= 0: return worst + 1.0, summ(r1), summ(wr)
            r2 = C.run(x, core.DT_CHECK); return max(worst, hc_score(r2, C.v)), summ(r1), summ(r2)
        rd = C.run(x, core.DT_OPT, **pert_kw("search_draw", nT))
        if rd["reasons"]: return 100.0 + rd["violation"], summ(r1), summ(rd)
        s1, sd = hc_score(r1, C.v), hc_score(rd, C.v)
        if max(s1, sd) >= 0: return max(s1, sd) + 1.0, summ(r1), summ(rd)
        r2 = C.run(x, core.DT_CHECK); return max(s1, hc_score(r2, C.v)), summ(r1), summ(r2)
    except Exception as ex:
        return 1000.0, dict(error=f"{type(ex).__name__}: {ex}"), {}


SPEED_TRIM_MODE = os.environ.get("M6_TRIM", "split")   # amendment F; "amp" reproduces the round-0 trim


def search(obj, mass, iters, pop, workers, seed, x0=None, v=0.75, sigma=0.15, tag=""):
    import cma, multiprocessing as mp
    assert seed >= 1, "pycma seed 0 is random"
    C = Carry(obj, mass, v); x0 = C.x0([0.0]*len(LAW_IDX)) if x0 is None else np.asarray(x0, float)
    out_dir = f"{RES}/search"; os.makedirs(out_dir, exist_ok=True); name = f"{out_dir}/{obj}_{mass:g}kg_v{v}_s{seed}{tag}"
    lg = open(name + ".log", "a"); pool = mp.get_context("fork").Pool(workers, initializer=_winit, initargs=(obj, mass, v, C.law_base.tolist()))
    t0 = time.time(); s0, a0, b0 = pool.apply(_weval, (x0,)); best = (s0, x0.copy(), a0, b0); hist = [(0, float(s0))]; last = 0; trims = []
    lg.write(f"warm score {s0:.4f} r1 {a0}\n"); lg.flush()
    def trim(best, it, es=None):
        # M5 SPEED TRIM (engine.walk.search, repair round 1; amendment B): a best that is gate-clean and periodic but off the net-speed
        # band has its step-length amplitude rescaled by v_cmd / v_net (<= 3 passes); kept only if the score improves (and injected
        # into the CMA population)
        for _ in range(3):
            r1 = best[2]
            if not (0 <= best[0] < 100) or r1.get("reasons") or not r1.get("periodic") or not r1.get("value") or abs(r1["value"] - v)/v <= 0.5*SPEED_TOL: break
            # repair round 1 (amendment F, general): the correction ratio v_cmd/v_net is applied three ways and the best is kept:
            # step length only (the M5 trim), cadence only (tscale / ratio, step length kept), and split evenly (sqrt each)
            ratio = v/r1["value"]; amp = 1.0 + 0.2*best[1][0]; ts = 1.0 + 0.15*best[1][1]; C_ = []
            for fa, ft in ([(ratio, 1.0), (1.0, 1.0/ratio), (np.sqrt(ratio), 1.0/np.sqrt(ratio))] if SPEED_TRIM_MODE == "split" else [(ratio, 1.0)]):
                xt = best[1].copy(); xt[0] = float(np.clip((amp*fa - 1.0)/0.2, -1, 1)); xt[1] = float(np.clip((ts*ft - 1.0)/0.15, -1, 1)); C_.append(xt)
            R_ = pool.map(_weval, C_); k_ = int(np.argmin([o[0] for o in R_])); xt = C_[k_]; st_, a1, a2 = R_[k_]
            trims.append(dict(it=it, amp=round(1.0 + 0.2*xt[0], 4), tscale=round(1.0 + 0.15*xt[1], 4), mode=["amp", "cadence", "split"][k_] if len(C_) > 1 else "amp",
                              v_before=r1["value"], v_after=a1.get("value"), score=round(float(st_), 4), all=[round(float(o[0]), 4) for o in R_]))
            lg.write(f"trim {trims[-1]}\n"); lg.flush()
            if st_ < best[0]:
                best = (st_, xt, a1, a2)
                if es is not None: es.inject([xt])
            else: break
        return best
    best = trim(best, 0)
    def dump(done):
        json.dump(dict(obj=obj, mass=mass, v=v, seed=seed, score=float(best[0]), x=best[1].tolist(), r1=best[2], r2=best[3], hist=hist, secs=time.time() - t0, done=done,
                       law_base=C.law_base.tolist(), law_idx=LAW_IDX, stride=C.st, trims=trims, gait=GAIT[v], valid_search=bool(best[0] < 0)), open(name + ".json", "w"), indent=1, default=float)
    dump(False)
    if s0 < 0 and os.environ.get("M6_STOP_ON_VALID", "1") == "1" and iters <= 0: pool.close(); return name + ".json"
    es = cma.CMAEvolutionStrategy(x0, sigma, {"bounds": [-1, 1], "seed": seed, "verbose": -9, "maxiter": iters, "popsize": pop})
    try:
        while not es.stop():
            X = es.ask(); res_ = pool.map(_weval, X); Fv = [o[0] for o in res_]; es.tell(X, Fv); k = int(np.argmin(Fv))
            if Fv[k] < best[0]:
                best = (Fv[k], np.array(X[k]), res_[k][1], res_[k][2]); last = es.countiter
                best = trim(best, es.countiter, es)
            hist.append((es.countiter, float(best[0])))
            r1 = best[2]; lg.write(f"it {es.countiter} best {best[0]:.4f} v {r1.get('value')} {r1.get('reasons')} load {r1.get('load', {}).get('reason')} t_held {r1.get('load', {}).get('t_held')} | {time.time() - t0:.0f}s\n"); lg.flush(); dump(False)
            if best[0] < 0 and es.countiter - last >= 6: break
    finally:
        pool.close(); pool.join()
    dump(True); return name + ".json"


def _vjob(a):
    obj, mass, v, law_base, x, name, dt, law = a
    C = Carry(obj, mass, v, dt_list=(dt,), uid=1, law_base=law_base); kw = pert_kw(name, len(C.S[dt]["types"]))
    t0 = time.time(); r = C.run(x, dt, law=law, **kw); o = summ(r); o["secs"] = round(time.time() - t0, 1); return name, dt, law, o


def validate(path, workers=6, ablation=True):
    """Fresh plants (uid 1, spawned processes) at 1 ms and 0.5 ms: nominal + P1-P4 (registered), and the law-off ablation (same x)."""
    import multiprocessing as mp
    g = json.load(open(path)); obj, mass, v, x, lb = g["obj"], g["mass"], g["v"], g["x"], g["law_base"]
    Carry(obj, mass, v, law_base=lb)       # scaffold cache once
    jobs = [(obj, mass, v, lb, x, nm, dt, True) for nm in ["nominal"] + list(PERT) for dt in (core.DT_OPT, core.DT_CHECK)]
    if ablation: jobs += [(obj, mass, v, lb, x, "nominal", dt, False) for dt in (core.DT_OPT, core.DT_CHECK)]
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool: res = pool.map(_vjob, jobs, chunksize=1)
    out = {}; abl = {}
    for nm, dt, law, r in res:
        (out.setdefault(nm, {}) if law else abl)[str(dt)] = r
    ok = {nm: all(out[nm][k]["valid"] for k in out[nm]) for nm in out}; n_p = sum(ok[nm] for nm in PERT)
    val = dict(source=path, obj=obj, mass=mass, v=v, nominal_valid_both_dt=ok["nominal"], perturb_pass=n_p, perturb_n=len(PERT), robust=bool(ok["nominal"] and n_p >= 3), ok=ok, runs=out,
               ablation_law_off=abl, perturbations={k: str(x_) for k, x_ in PERT.items()}, note="fresh plants (uid 1) in spawned processes; shove runs: energy closure waived (external work), as M5")
    json.dump(val, open(path.replace(".json", "_validation.json"), "w"), indent=1, default=float); return val


def ladder(obj, masses, iters, pop, workers, v=0.75, warm=None):
    """The registered ladder: per rung CMA seed 1 warm from the previous rung's best; if not valid, seed 2 (sigma 0.25) once; the
    first rung that still has no valid gait ends the ladder. Every valid rung is validated (fresh plants, P1-P4, law-off ablation)."""
    lp = f"{RES}/ladder_{obj}_v{v}.json"; L_ = json.load(open(lp)) if os.path.exists(lp) else dict(obj=obj, v=v, rungs={})
    x0 = None if warm is None else json.load(open(warm))["x"]
    for mass in masses:
        key = f"{mass:g}"
        if key in L_["rungs"] and L_["rungs"][key].get("done"):
            if L_["rungs"][key].get("valid"): x0 = json.load(open(L_["rungs"][key]["best"]))["x"]; continue
            break
        rec = dict(mass=mass, runs=[])
        seeds = [(int(a.split(":")[0]), float(a.split(":")[1])) for a in os.environ.get("M6_SEEDS", "1:0.15,2:0.25").split(",")]
        for seed, sg in seeds:
            pth = f"{RES}/search/{obj}_{mass:g}kg_v{v}_s{seed}.json"
            if not (os.path.exists(pth) and json.load(open(pth)).get("done")): pth = search(obj, mass, iters, pop, workers, seed, x0=x0, v=v, sigma=sg)   # resume: a finished search is reused
            g = json.load(open(pth)); rec["runs"].append(dict(path=pth, score=g["score"], valid_search=g["valid_search"]))
            if g["valid_search"]: break
        best = min(rec["runs"], key=lambda r: r["score"]); rec["best"] = best["path"]; rec["valid"] = bool(best["valid_search"])
        if rec["valid"]:
            val = validate(best["path"], workers); rec.update(nominal_fresh=val["nominal_valid_both_dt"], perturb_pass=val["perturb_pass"], robust=val["robust"],
                                                           ablation={k: dict(reasons=x["reasons"], t_held=x["load"]["t_held"]) for k, x in val["ablation_law_off"].items()})
            x0 = json.load(open(best["path"]))["x"]
        rec["done"] = True; L_["rungs"][key] = rec; json.dump(L_, open(lp, "w"), indent=1, default=float)
        os.system(f'{os.path.expanduser("~/bin/notify")} "M6 {obj} {mass:g} kg: valid={rec["valid"]} robust={rec.get("robust")} P={rec.get("perturb_pass")}/4" >/dev/null 2>&1')
        if not rec["valid"]: break
    return L_


def _alive(pid):
    try: os.kill(int(pid), 0); return True
    except Exception: return False


def robust_follow(obj, v, masses, iters, pop, workers):
    """Amendment D: follow the ladder; for each VALID rung run the robust-repair search (M6_DRAWS in the score), warm from the rung's
    best, then validate on fresh plants (P1-P4 + ablation). Waits for the ladder (PID in M6_LADDER_PID); stops at the first invalid
    rung or when the ladder has exited without recording the rung."""
    lp = f"{RES}/ladder_{obj}_v{v}.json"; out_p = f"{RES}/robust_{obj}_v{v}.json"; out = json.load(open(out_p)) if os.path.exists(out_p) else {}
    for mass in masses:
        key = f"{mass:g}"
        while True:
            L_ = json.load(open(lp)) if os.path.exists(lp) else {"rungs": {}}
            if key in L_["rungs"] and L_["rungs"][key].get("done"): break
            if not _alive(os.environ.get("M6_LADDER_PID", "0")): return out
            time.sleep(30)
        rung = L_["rungs"][key]
        if not rung.get("valid"): return out
        if key in out and out[key].get("done"): continue
        pth = search(obj, mass, iters, pop, workers, 1, x0=json.load(open(rung["best"]))["x"], v=v, sigma=0.15, tag="_rob")
        g = json.load(open(pth)); rec = dict(path=pth, score=g["score"], valid_search=g["valid_search"])
        if g["valid_search"]:
            val = validate(pth, workers); rec.update(nominal_fresh=val["nominal_valid_both_dt"], perturb_pass=val["perturb_pass"], robust=val["robust"], ok=val["ok"])
        rec["done"] = True; out[key] = rec; json.dump(out, open(out_p, "w"), indent=1, default=float)
        os.system(f'{os.path.expanduser("~/bin/notify")} "M6 robust-repair {obj} {mass:g} kg v{v}: valid={rec["valid_search"]} P={rec.get("perturb_pass")}/4" >/dev/null 2>&1')
    return out


def technique(path, law=True, dt=core.DT_OPT):
    """Measured technique of a stored result on a fresh plant (uid 1): torque utilisation per driven joint (p95 and max of |tau|/rating
    over the steady part), waist-yaw counter-rotation amplitude, load excursion/tilt/head share, cadence, step, distance."""
    g = json.load(open(path)); C = Carry(g["obj"], g["mass"], g["v"], dt_list=(dt,), uid=1, law_base=g["law_base"]); S_ = C.S[dt]
    r = C.run(g["x"], dt, law=law, record=True); tr = r["trace"]; t = np.asarray(tr["t"]); tq = np.asarray(tr["tq"]); tau = S_["jarr"]["tau"]; types = S_["types"]
    k = t >= N_TRANS*C.R["Tc"]*W.unpack(S_, C.split(g["x"])[0])["tscale"]
    u = np.abs(tq[k])/tau if k.any() else np.zeros((1, len(types)))
    util = {tn: dict(p95=round(float(np.percentile(u[:, i], 95)), 3), max=round(float(u[:, i].max()), 3)) for i, tn in enumerate(types)}
    top = sorted(((tn, v_["p95"]) for tn, v_ in util.items()), key=lambda kv: -kv[1])[:6]
    qp = np.asarray(tr["qpos"]); wy = S_["jarr"]["qad"][types.index("waist_yaw_joint")]; wp = S_["jarr"]["qad"][types.index("waist_pitch_joint")]
    zw, zl = C.split(g["x"]); PL, _ = law_params(zl, C.law_base); L = C.law_setup(S_, PL)
    return dict(source=path, law=law, dt=dt, valid=r["valid"], reasons=r["reasons"], v_net=r["value"], dist=r["dist"], t_run=r["t_run"], cadence_spm=r["cadence_spm"], step_len=r["step_len"],
                ds_frac=r["ds_frac"], slip_max=r["slip_max"], load=r["load"], torque_util=util, top_util_p95=top,
                waist_yaw_ptp_rad=round(float(np.ptp(qp[k, wy])), 4) if k.any() else None, waist_pitch_mean_rad=round(float(np.mean(qp[k, wp])), 4) if k.any() else None,
                law_gains=dict(wn=L["wn"], wn_rule=L["wn_rule"], zeta=L["zeta"], sN=L["sN"], Nmin=L["Nmin"], w_arm=L["w_arm"], waist_offset=L["waist"], F_cap=L["F_cap"]),
                walk_params={k_: (v_ if isinstance(v_, dict) else float(v_)) for k_, v_ in W.unpack(S_, zw).items() if k_ != "x"}, counter=C.R.get("counter"))


def summary():
    """Collect every validated M6 result into summary.json (one row per validated search result)."""
    import glob
    rows = []
    for vp in sorted(glob.glob(f"{RES}/search/*_validation.json")):
        v_ = json.load(open(vp)); g = json.load(open(v_["source"])); n = v_["runs"]["nominal"]
        row = dict(result=os.path.relpath(v_["source"], RES), obj=v_["obj"], mass=v_["mass"], v_cmd=v_["v"], nominal_valid_both_dt=v_["nominal_valid_both_dt"], perturb_pass=v_["perturb_pass"], robust=v_["robust"], ok=v_["ok"])
        for dt, r in n.items():
            row[f"dt{dt}"] = dict(v_net=r["value"], dist=r["dist"], t_run=r["t_run"], e_max=r["load"]["e_max"], tilt_max=r["load"]["tilt_max"], head_frac=r["load"]["head_frac"], arm_util_p95=r["load"]["arm_util_p95"],
                                  slip=r["slip_max"], ds_min=r["ds_min"], cadence=r["cadence_spm"], step=r["step_len"], J_per_m=r["J_per_m"], reasons=r["reasons"])
        row["fail_reasons"] = {nm: {dt: r["reasons"] for dt, r in rr.items() if r["reasons"] or not r["valid"]} for nm, rr in v_["runs"].items() if not v_["ok"][nm]}
        row["law_off"] = {dt: dict(reason=r["load"]["reason"], t_held=r["load"]["t_held"], dist_held=r["load"]["dist_held"]) for dt, r in v_["ablation_law_off"].items()}
        row["search"] = dict(seed=g["seed"], score=g["score"], secs=g.get("secs"), n_hist=len(g.get("hist", [])), trims=g.get("trims"))
        rows.append(row)
    json.dump(dict(rows=rows, perturbations={k: str(x) for k, x in PERT.items()}, search_draws={k: str(x) for k, x in DRAWS.items()}), open(f"{RES}/summary.json", "w"), indent=1, default=float)
    for r in rows: print(r["obj"], r["mass"], r["v_cmd"], r["result"], r["nominal_valid_both_dt"], f"{r['perturb_pass']}/4", r["robust"], {k: round(x["v_net"], 3) for k, x in r.items() if k.startswith("dt")}, {k: round(x["dist"], 2) for k, x in r.items() if k.startswith("dt")}, r["law_off"])
    return rows


SUMKEYS = W.KEYS + ("dist", "fell")
def summ(r):
    o = {k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in SUMKEYS if k in r}; o["load"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["load"].items()}
    return o


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "predict": predict(v=float(sys.argv[2]) if len(sys.argv) > 2 else 0.75)
    elif cmd == "search":
        obj, mass, iters, pop, workers, seed = sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
        x0 = json.load(open(sys.argv[8]))["x"] if len(sys.argv) > 8 and sys.argv[8] != "-" else None
        print(search(obj, mass, iters, pop, workers, seed, x0=x0, v=(float(sys.argv[9]) if len(sys.argv) > 9 else 0.75), tag=os.environ.get("M6_TAG", "")), flush=True)
    elif cmd == "ladder":
        obj = sys.argv[2]; masses = [float(x) for x in sys.argv[3].split(",")]; iters, pop, workers = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
        ladder(obj, masses, iters, pop, workers, warm=(sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] != "-" else None), v=(float(sys.argv[8]) if len(sys.argv) > 8 else 0.75))
    elif cmd == "robust":
        robust_follow(sys.argv[2], float(sys.argv[3]), [float(x) for x in sys.argv[4].split(",")], int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]))
    elif cmd == "summary": summary()
    elif cmd == "technique":
        print(json.dumps(technique(sys.argv[2], law=(sys.argv[3] != "0") if len(sys.argv) > 3 else True), default=float), flush=True)
    elif cmd == "render":
        from engine import render_any as RA
        g = json.load(open(sys.argv[2])); law = (sys.argv[3] != "0") if len(sys.argv) > 3 else True
        C = Carry(g["obj"], g["mass"], g["v"], dt_list=(core.DT_OPT,), uid=1, law_base=g["law_base"]); S_ = C.S[core.DT_OPT]
        r = C.run(g["x"], core.DT_OPT, law=law, record=True); Q = np.asarray(r["trace"]["qpos"]); ld = r["load"]
        tag = f"{g['obj']}_{g['mass']:g}kg_v{g['v']}_{'law_on' if law else 'law_off'}"; os.makedirs(f"{RES}/render", exist_ok=True)
        np.save(f"{RES}/render/{tag}_trace.npy", Q)
        sub = (f"{'held' if ld['reason'] is None else 'LOST at %.2f s' % ld['t_lost']} | net {r['value']:.2f} m/s, {r['dist']:.1f} m | valid={r['valid']} {r['reasons']}")
        RA.render(S_["scene"], Q, 4*core.DT_OPT, f"{RES}/render/{tag}.mp4", title=f"G1H {g['v']} m/s, {g['mass']:g} kg {g['obj']} on head, law {'ON' if law else 'OFF'}",
                  sub=sub, distance=3.2, azimuth=150)
        print(tag, sub, flush=True)
    elif cmd == "validate":
        v_ = validate(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6); print(json.dumps({k: v_[k] for k in ("obj", "mass", "nominal_valid_both_dt", "perturb_pass", "robust", "ok")}), flush=True)
    elif cmd == "replay":
        obj, mass = sys.argv[2], float(sys.argv[3]); law = (sys.argv[4] != "0") if len(sys.argv) > 4 else True; dt = float(sys.argv[5]) if len(sys.argv) > 5 else core.DT_OPT
        t0 = time.time(); C = Carry(obj, mass, dt_list=(dt,)); print("setup", round(time.time() - t0, 1), "s; ratings", ratings_table(C.S[dt]), flush=True)
        t0 = time.time(); r = C.run(C.x0(), dt, law=law); print(json.dumps(summ(r), default=float), "\n", round(time.time() - t0, 1), "s", flush=True)
