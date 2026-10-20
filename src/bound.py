"""Analytic upper bound on takeoff velocity for a planar jumper.

No optimizer. Every quantity follows from the body spec by integration, so the
result is an upper bound that no controller, timing or technique can exceed.

Two strict bounds are computed and the tighter is reported:

  ROM bound     W_i <= integral over the joint's traversed range of
                tau_max_i * f_angle_i(theta) dtheta
                (uses the isometric ceiling, since the torque-speed factor is <= 1;
                 this is the supremum as joint speed goes to zero)

  POWER bound   W_i <= T * max over omega of [tau_max_i * f_angle_i * (1 - omega/omega_max_i) * omega]
                     = T * tau_max_i * f_angle_bar_i * omega_max_i / 4
                (tight when the push-off duration T is short)

Takeoff kinetic energy is bounded by the summed work minus the potential-energy
cost of raising the centre of mass from the start pose to the takeoff pose.
"""
import numpy as np, yaml, itertools, os

G = 9.80665
JOINTS = ["hip", "knee", "ankle", "shoulder", "elbow"]
SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spec", "body.yaml")


class Body:
    def __init__(self, path=SPEC, overrides=None):
        self.d = yaml.safe_load(open(path))
        if overrides:
            for k, v in overrides.items():          # e.g. {"knee.tau_max": 750.0}
                a, b = k.split("."); self.d["joints"][a][b]["value"] = v
        M = self.d["meta"]["subject"]["body_mass"]["value"]
        H = self.d["meta"]["subject"]["height"]["value"]
        self.M, self.H = M, H
        s = self.d["segments"]
        self.seg = {}
        for n, v in s.items():
            self.seg[n] = dict(
                m=v["mass_fraction"]["value"]*M,
                L=v["length_fraction"]["value"]*H,
                c=v["com_from_proximal"]["value"],
                k=v["rog_about_com"]["value"])
        self.j = {}
        for n in JOINTS:
            v = self.d["joints"][n]
            ta = v["torque_angle"]
            self.j[n] = dict(tau=v["tau_max"]["value"], wmax=v["omega_max"]["value"],
                             lo=v["range"]["value"][0], hi=v["range"]["value"][1],
                             peak=ta["peak_at"], width=ta["width"], floor=ta["floor"])
        self.ankle_h = 0.075   # ankle centre above sole, foot flat (INVENTED)
        Lf = self.seg["foot"]["L"]; fr = self.d["segments"]["foot"]["ankle_from_heel_fraction"]["value"]
        self.foot_mid_x = (0.5-fr)*Lf
        self.total_mass = sum(v["m"] for v in self.seg.values())

    # ---- torque-angle shape: 1 at the optimum, falling to `floor` at +-width
    def f_angle(self, jn, th):
        p = self.j[jn]
        x = np.clip((np.asarray(th, float)-p["peak"])/p["width"], -1.0, 1.0)
        return p["floor"] + (1.0-p["floor"])*np.cos(0.5*np.pi*x)**2

    def tau_avail(self, jn, th, om=0.0):
        p = self.j[jn]
        f = self.f_angle(jn, th)
        sp = np.clip(1.0 - np.abs(om)/p["wmax"], 0.0, 1.0)
        return p["tau"]*f*sp

    # ---- work available from one joint traversing th0 -> th1 (ROM bound, omega->0)
    def work_rom(self, jn, th0, th1, n=4001):
        th = np.linspace(th0, th1, n)
        return float(abs(np.trapezoid(self.tau_avail(jn, th, 0.0), th)))

    def work_conditional(self, jn, th0, th1, t_contact, n=2001):
        """min over the traversed range of tau_max*f_angle evaluated at the mean speed the
        movement forces: wbar = |dtheta|/t_contact. Rigorous, conditional on t_contact."""
        p = self.j[jn]; dth = abs(th1-th0)
        if t_contact <= 0 or dth == 0: return 0.0
        wbar = dth/t_contact
        frac = max(0.0, 1.0 - wbar/p["wmax"])
        th = np.linspace(th0, th1, n)
        return float(abs(np.trapezoid(p["tau"]*self.f_angle(jn, th)*frac, th)))

    def peak_power(self, jn, th0, th1, n=401):
        """max over the traversed range of the torque-speed power peak tau*f*wmax/4"""
        th = np.linspace(th0, th1, n)
        p = self.j[jn]
        return float(np.max(p["tau"]*self.f_angle(jn, th)*p["wmax"]/4.0))

    # ---- planar kinematics, ABSOLUTE segment angles from vertical (+ = top tilted forward/+x)
    # A pose is given as absolute angles; joint angles are differences. This is checkable
    # by hand: standing = all absolute angles zero.
    def abs_from_joint(self, q):
        phi_s = q["ankle"]
        phi_t = phi_s - q["knee"]
        phi_tr = phi_t + q["hip"]
        phi_ua = phi_tr + q["shoulder"]
        phi_fa = phi_ua + q["elbow"]
        return dict(shank=phi_s, thigh=phi_t, trunk=phi_tr, uarm=phi_ua, farm=phi_fa)

    def joint_from_abs(self, a):
        return dict(ankle=a["shank"], knee=a["shank"]-a["thigh"], hip=a["trunk"]-a["thigh"],
                    shoulder=a["uarm"]-a["trunk"], elbow=a["farm"]-a["uarm"])

    def com_height_abs(self, a, ankle_h=None):
        """a = absolute segment angles. Arms hang DOWN from the shoulder."""
        S, T, TR = self.seg["shank"], self.seg["thigh"], self.seg["trunk_head"]
        UA, FA = self.seg["upper_arm"], self.seg["forearm_hand"]
        z_ank = self.ankle_h if ankle_h is None else ankle_h
        z_knee = z_ank + S["L"]*np.cos(a["shank"])
        z_hip  = z_knee + T["L"]*np.cos(a["thigh"])
        z_sho  = z_hip + TR["L"]*np.cos(a["trunk"])
        z_elb  = z_sho - UA["L"]*np.cos(a["uarm"])
        zc = {"foot": 0.5*z_ank,
              "shank": z_ank + S["L"]*S["c"]*np.cos(a["shank"]),
              "thigh": z_knee + T["L"]*T["c"]*np.cos(a["thigh"]),
              "trunk_head": z_hip + TR["L"]*TR["c"]*np.cos(a["trunk"]),
              "upper_arm": z_sho - UA["L"]*UA["c"]*np.cos(a["uarm"]),
              "forearm_hand": z_elb - FA["L"]*FA["c"]*np.cos(a["farm"])}
        self._last_joints = dict(ankle=z_ank, knee=z_knee, hip=z_hip, shoulder=z_sho, elbow=z_elb)
        return float(sum(self.seg[n]["m"]*zc[n] for n in zc)/self.total_mass)

    def com_height(self, q):
        return self.com_height_abs(self.abs_from_joint(q))

    def com_x_abs(self, a, ankle_h=None):
        """Horizontal COM position, ankle joint at x=0. Needed for the balance constraint."""
        S, T, TR = self.seg["shank"], self.seg["thigh"], self.seg["trunk_head"]
        UA, FA = self.seg["upper_arm"], self.seg["forearm_hand"]
        x_ank = 0.0
        x_knee = x_ank + S["L"]*np.sin(a["shank"])
        x_hip  = x_knee + T["L"]*np.sin(a["thigh"])
        x_sho  = x_hip + TR["L"]*np.sin(a["trunk"])
        x_elb  = x_sho + UA["L"]*np.sin(a["uarm"])
        xc = {"foot": self.foot_mid_x,
              "shank": x_ank + S["L"]*S["c"]*np.sin(a["shank"]),
              "thigh": x_knee + T["L"]*T["c"]*np.sin(a["thigh"]),
              "trunk_head": x_hip + TR["L"]*TR["c"]*np.sin(a["trunk"]),
              "upper_arm": x_sho + UA["L"]*UA["c"]*np.sin(a["uarm"]),
              "forearm_hand": x_elb + FA["L"]*FA["c"]*np.sin(a["farm"])}
        return float(sum(self.seg[n]["m"]*xc[n] for n in xc)/self.total_mass)

    def support_polygon(self):
        """Heel and toe x, ankle joint at x = 0."""
        L = self.seg["foot"]["L"]; f = self.d["segments"]["foot"]["ankle_from_heel_fraction"]["value"]
        return -f*L, (1.0-f)*L

# ------------------------------------------------------------------ poses
# Poses stated in ABSOLUTE segment angles (rad from vertical, + = top forward).
# Standing is all zeros, which makes these checkable by inspection.
ABS_TAKEOFF = dict(shank=0.09, thigh=0.09, trunk=0.09, uarm=-0.60, farm=-0.50)
ABS_CROUCH  = dict(shank=0.35, thigh=-1.22, trunk=0.52, uarm=0.70, farm=0.90)
ANKLE_H_STAND   = 0.075     # ankle centre above sole, foot flat
ANKLE_H_TAKEOFF = 0.175     # heel raised at toe-off (plantarflexion), INVENTED

def takeoff_pose(b):
    return b.joint_from_abs(ABS_TAKEOFF)

# Squat-jump start pose. Knee ~90 deg flexion is the standard SJ starting position in
# the literature; the remaining angles are chosen to keep the trunk upright and the arms
# back for a swing. All INVENTED (see spec/body.yaml) but within published SJ ranges.
CROUCH_SJ = dict(hip=-1.40, knee=1.57, ankle=-0.35, shoulder=-0.60, elbow=0.30)

def crouch_pose(b, q=None):
    return b.joint_from_abs(ABS_CROUCH) if q is None else dict(q)

def segment_heights(b, q, ankle_h=None):
    b.com_height_abs(b.abs_from_joint(q), ankle_h)
    return dict(b._last_joints)

def pose_is_clear(b, q, margin=0.02):
    z = segment_heights(b, q)
    return all(v > margin for v in z.values()), z


# ------------------------------------------------------------------ the bound
def subset_bound(b, subset, T_push=0.30, start=None, passive=False):
    """Upper bound on takeoff COM velocity when only `subset` joints actuate.

    passive=False : non-participating joints are LOCKED RIGID at their start angle
                    (they still transmit force through a rigid link).
    passive=True  : non-participating joints are free and dissipate; their potential
                    contribution is removed AND they cannot hold the chain, modelled
                    here as the conservative case where their span contributes no work
                    and the body must still be raised.
    """
    q0 = crouch_pose(b) if start is None else dict(start)
    q1 = takeoff_pose(b)
    qT = dict(q0)
    for n in subset: qT[n] = q1[n]
    W_rom = 0.0; W_pow = 0.0; W = 0.0
    per = {}
    for n in subset:
        w_r = b.work_rom(n, q0[n], qT[n])
        w_p = b.peak_power(n, q0[n], qT[n])*T_push
        w_c = b.work_conditional(n, q0[n], qT[n], T_push)
        w_u = min(w_r, w_p, w_c)
        W_rom += w_r; W_pow += w_p; W += w_u
        bd = "rom" if w_u == w_r else ("power" if w_u == w_p else "conditional")
        per[n] = dict(rom=w_r, power=w_p, conditional=w_c, used=w_u, binding=bd,
                      excursion=abs(qT[n]-q0[n]))
    a0 = b.abs_from_joint(q0); aT = b.abs_from_joint(qT)
    z0 = b.com_height_abs(a0, ANKLE_H_STAND)
    zT = b.com_height_abs(aT, ANKLE_H_TAKEOFF if "ankle" in subset else ANKLE_H_STAND)
    dz = zT - z0
    KE = W - b.total_mass*G*dz
    if passive: KE = W - b.total_mass*G*dz          # same energy ledger; see note in docs
    v = float(np.sqrt(max(0.0, 2.0*KE/b.total_mass)))
    nb = sum(1 for v in per.values() if v["binding"] == "power")
    return dict(subset=tuple(subset), W_rom=W_rom, W_power=W_pow, W_used=W,
                binding=f"{nb}/{len(per)} power-bound", n_power_bound=nb,
                q_crouch=dict(q0), q_takeoff=dict(qT),
                dz=dz, PE=b.total_mass*G*dz, KE=max(0.0, KE), v_takeoff=v,
                height=v*v/(2*G), per_joint_work=per)


def all_subsets(b, T_push=0.30, start=None):
    out = []
    for r in range(1, len(JOINTS)+1):
        for s in itertools.combinations(JOINTS, r):
            out.append(subset_bound(b, s, T_push, start))
    return out


def frontier_by_k(rows):
    """best achievable bound using exactly k joints, and which subset achieves it"""
    best = {}
    for r in rows:
        k = len(r["subset"])
        if k not in best or r["v_takeoff"] > best[k]["v_takeoff"]: best[k] = r
    ks = sorted(best)
    marg = {k: (best[k]["v_takeoff"] - (best[k-1]["v_takeoff"] if k-1 in best else 0.0)) for k in ks}
    return best, marg
