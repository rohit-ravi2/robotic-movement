"""Planar human jumper generated FROM spec/body.yaml so the dynamic model and the analytic
bound describe the same body. Bilaterally lumped, sagittal, floating base, foot contact."""
import numpy as np, yaml, mujoco, os
# qpos = QSIGN * (spec anatomical angle). DECLARED here and used both to write the MJCF joint ranges
# and to evaluate every angle-dependent capacity (Phase 10, defect P15: the hip range was negated in
# the XML but the torque-angle curve was evaluated at raw qpos, so the hip extensor's peak sat at
# hyperextension and the crouch ran at the 45% floor; defect P16: the elbow flexed posteriorly).
QSIGN = {"hip": -1.0, "knee": 1.0, "ankle": 1.0, "shoulder": 1.0, "elbow": -1.0}
def _rng(r, sgn):
    lo, hi = r
    return (lo, hi) if sgn > 0 else (-hi, -lo)
SPEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spec", "body.yaml")

def build_xml(spec=SPEC):
    d = yaml.safe_load(open(spec))
    M = d["meta"]["subject"]["body_mass"]["value"]; H = d["meta"]["subject"]["height"]["value"]
    s = {k: dict(m=v["mass_fraction"]["value"]*M, L=v["length_fraction"]["value"]*H,
                 c=v["com_from_proximal"]["value"], k=v["rog_about_com"]["value"])
         for k, v in d["segments"].items()}
    j = d["joints"]
    fa = d["segments"]["foot"]["ankle_from_heel_fraction"]["value"]
    Lf = s["foot"]["L"]; heel = -fa*Lf; toe = (1-fa)*Lf
    def cap(n, L, mass, r):   # capsule along -z from origin
        return f'<geom name="{n}" type="capsule" fromto="0 0 0 0 0 {-L}" size="{r}" mass="{mass}"/>'
    xml = f'''<mujoco model="human_planar">
  <compiler angle="radian"/>
  <option timestep="0.001" gravity="0 0 -9.80665" integrator="implicitfast" cone="pyramidal"/>
  <default><joint damping="0.8" armature="0.02" limited="true"/>
    <geom friction="0.9 0.005 0.0001" condim="3" solref="0.008 1" solimp="0.95 0.99 0.001"
          contype="0" conaffinity="0"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 .1" pos="0 0 0" contype="1" conaffinity="2"/>
    <body name="pelvis" pos="0 0 1.0">
      <joint name="bx" type="slide" axis="1 0 0" limited="false"/>
      <joint name="bz" type="slide" axis="0 0 1" limited="false"/>
      <joint name="bp" type="hinge" axis="0 1 0" limited="false"/>
      <body name="trunk" pos="0 0 0">
        <geom name="trunk" type="capsule" fromto="0 0 0 0 0 {s['trunk_head']['L']}" size="0.10" mass="{s['trunk_head']['m']}"/>
        <body name="uarm" pos="0 0 {s['trunk_head']['L']*0.92}">
          <joint name="shoulder" axis="0 1 0" range="{_rng(j['shoulder']['range']['value'], QSIGN['shoulder'])[0]} {_rng(j['shoulder']['range']['value'], QSIGN['shoulder'])[1]}"/>
          {cap('uarm', s['upper_arm']['L'], s['upper_arm']['m'], 0.045)}
          <body name="farm" pos="0 0 {-s['upper_arm']['L']}">
            <joint name="elbow" axis="0 1 0" range="{_rng(j['elbow']['range']['value'], QSIGN['elbow'])[0]} {_rng(j['elbow']['range']['value'], QSIGN['elbow'])[1]}"/>
            {cap('farm', s['forearm_hand']['L'], s['forearm_hand']['m'], 0.038)}
          </body></body></body>
      <body name="thigh" pos="0 0 0">
        <joint name="hip" axis="0 1 0" range="{_rng(j['hip']['range']['value'], QSIGN['hip'])[0]} {_rng(j['hip']['range']['value'], QSIGN['hip'])[1]}"/>
        {cap('thigh', s['thigh']['L'], s['thigh']['m'], 0.058)}
        <body name="shank" pos="0 0 {-s['thigh']['L']}">
          <joint name="knee" axis="0 1 0" range="{_rng(j['knee']['range']['value'], QSIGN['knee'])[0]} {_rng(j['knee']['range']['value'], QSIGN['knee'])[1]}"/>
          {cap('shank', s['shank']['L'], s['shank']['m'], 0.048)}
          <body name="foot" pos="0 0 {-s['shank']['L']}">
            <joint name="ankle" axis="0 1 0" range="{_rng(j['ankle']['range']['value'], QSIGN['ankle'])[0]} {_rng(j['ankle']['range']['value'], QSIGN['ankle'])[1]}"/>
            <geom name="foot" contype="2" conaffinity="1" type="box" size="{Lf/2} 0.05 0.02" pos="{(heel+toe)/2} 0 -0.055" mass="{s['foot']['m']}"/>
          </body></body></body></body>
  </worldbody>
  <actuator>
    <motor joint="hip"/><motor joint="knee"/><motor joint="ankle"/>
    <motor joint="shoulder"/><motor joint="elbow"/>
  </actuator>
</mujoco>'''
    return xml, d

JOINTS = ["hip", "knee", "ankle", "shoulder", "elbow"]
HP_JOINT_NAMES = JOINTS

class Human:
    def __init__(self, spec=SPEC, tau_scale=None, act_tau=None):
        xml, d = build_xml(spec)
        self.m = mujoco.MjModel.from_xml_string(xml); self.d = mujoco.MjData(self.m)
        self.spec = d
        jj = d["joints"]
        self.tau = np.array([jj[n]["tau_max"]["value"] for n in JOINTS])
        if tau_scale is not None: self.tau = self.tau*np.asarray(tau_scale)
        self.wmax = np.array([jj[n]["omega_max"]["value"] for n in JOINTS])
        self.ta = [(jj[n]["torque_angle"]["peak_at"], jj[n]["torque_angle"]["width"],
                    jj[n]["torque_angle"]["floor"]) for n in JOINTS]
        self.act_tau = act_tau if act_tau is not None else d["actuation"]["activation_tau"]["value"]
        self.jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
        self.qadr = [self.m.jnt_qposadr[i] for i in self.jid]
        self.vadr = [self.m.jnt_dofadr[i] for i in self.jid]
        self.M = float(sum(self.m.body_mass[1:]))
        self.fid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "foot")
        self.qsign = np.array([QSIGN[n] for n in JOINTS]); self._AJmap = {}
        # Anderson, Madigan & Nussbaum 2007 joint-torque surfaces (spec actuation.torque_model)
        self.anderson = d["actuation"].get("torque_model", {}).get("value") == "anderson2007"
        self.tscale = np.ones(5) if tau_scale is None else np.asarray(tau_scale, float)*np.ones(5)
        if self.anderson:
            BWH = d["meta"]["subject"]["body_mass"]["value"]*9.80665*d["meta"]["subject"]["height"]["value"]
            self.AJ = [i for i, n in enumerate(JOINTS) if "anderson" in jj[n]]
            self.AP = {}
            for i in self.AJ:
                a = jj[JOINTS[i]]["anderson"]
                sA = a["angle_sign"]["value"]           # anatomical (flexion/dorsiflexion +) = sA * spec angle
                e = -sA*QSIGN[JOINTS[i]]                # actuator sign of the EXTENSOR torque
                def par(blk):
                    v = {k: blk[k]["value"] for k in ("C1", "C2", "C3", "C4", "C5", "C6")}
                    v["T0"] = 2.0*v["C1"]*BWH           # bilateral lumping of a unilateral measurement
                    return v
                self.AP[i] = dict(sA=sA, e=e, ext=par(a["extensor"]), flex=par(a["flexor"]))
                self.tau[i] = self.AP[i]["ext"]["T0"]*self.tscale[i]   # command normalisation
            self._AJmap = {i: (float(self.qsign[i]*self.AP[i]["sA"]), float(self.AP[i]["e"]), float(self.tscale[i]),
                               tuple(float(self.AP[i]["ext"][c]) for c in ("C2", "C3", "C4", "C5", "C6", "T0")),
                               tuple(float(self.AP[i]["flex"][c]) for c in ("C2", "C3", "C4", "C5", "C6", "T0")))
                           for i in self.AJ}
            # vectorised parameter arrays over the Anderson joints
            ix = np.array(self.AJ); self._ix = ix
            self._k = self.qsign[ix]*np.array([self.AP[i]["sA"] for i in self.AJ])     # thA = k * qpos
            self._e = np.array([self.AP[i]["e"] for i in self.AJ])
            self._V = {key: {c: np.array([self.AP[i][key][c] for i in self.AJ]) for c in ("C2", "C3", "C4", "C5", "C6", "T0")}
                       for key in ("ext", "flex")}

    def rescale_anderson(self, vel=1.0, iso=1.0, ankle_iso=1.0):
        """LABELLED SWEEP ONLY (Phase 10 attribution map, never a banked spec): stretch the velocity axis
        (C4, C5 x vel; C6 / vel keeps the eccentric slope per unit of stretched speed), scale isometric
        magnitude (T0 x iso; plantarflexors additionally x ankle_iso)."""
        for i in self.AJ:
            for key in ("ext", "flex"):
                p = self.AP[i][key]
                p["C4"] *= vel; p["C5"] *= vel; p["C6"] /= vel
                p["T0"] *= iso*(ankle_iso if (HP_JOINT_NAMES[i] == "ankle" and key == "ext") else 1.0)
            self.tau[i] = self.AP[i]["ext"]["T0"]*self.tscale[i]
        self._AJmap = {i: (float(self.qsign[i]*self.AP[i]["sA"]), float(self.AP[i]["e"]), float(self.tscale[i]),
                           tuple(float(self.AP[i]["ext"][c]) for c in ("C2", "C3", "C4", "C5", "C6", "T0")),
                           tuple(float(self.AP[i]["flex"][c]) for c in ("C2", "C3", "C4", "C5", "C6", "T0")))
                       for i in self.AJ}

    @staticmethod
    def _ander(p, thA, w):
        """Anderson 2007 Eq. 9 (concentric, w >= 0) and the eccentric factor (1 - C6*w), w < 0."""
        ang = max(0.0, np.cos(p["C2"]*(thA - p["C3"])))
        if w >= 0:
            c4, c5 = p["C4"], p["C5"]
            g = (2*c4*c5 + w*(c5 - 3*c4))/(2*c4*c5 + w*(2*c5 - 4*c4))
            g = min(1.0, max(0.0, g))
        else:
            g = 1.0 - p["C6"]*w
        return p["T0"]*ang*g

    def limits(self, th, w):
        """Directional torque capacity (lo, hi) >= 0 in actuator sign at qpos th, qvel w:
        the actuator may apply tau in [-lo, hi]. Scalar math (5 joints; numpy per-call overhead
        dominated the vectorised form). Checked against _ander/_surf in tests."""
        from math import cos, pi
        lo = np.empty(5); hi = np.empty(5)
        for i in range(5):
            P = self._AJmap.get(i)
            if P is None:
                p, wd, fl = self.ta[i]
                x = (self.qsign[i]*th[i] - p)/wd; x = 1.0 if x > 1.0 else (-1.0 if x < -1.0 else x)
                c = cos(0.5*pi*x); lim = self.tau[i]*(fl + (1.0 - fl)*c*c)
                wi = w[i]; sp = 1.0 - abs(wi)/self.wmax[i]; sp = 0.0 if sp < 0.0 else (1.0 if sp > 1.0 else sp)
                hi[i] = lim*sp if wi > 0 else lim; lo[i] = lim*sp if wi < 0 else lim
            else:
                k, e, ts, E, F = P; thA = k*th[i]; wE = e*w[i]
                Te = self._sc(E, thA, wE)*ts; Tf = self._sc(F, thA, -wE)*ts
                if e > 0: hi[i] = Te; lo[i] = Tf
                else: hi[i] = Tf; lo[i] = Te
        return lo, hi

    @staticmethod
    def _sc(p, thA, w):
        from math import cos
        C2, C3, C4, C5, C6, T0 = p
        a = cos(C2*(thA - C3)); a = a if a > 0.0 else 0.0
        if w >= 0.0:
            g = (2*C4*C5 + w*(C5 - 3*C4))/(2*C4*C5 + w*(2*C5 - 4*C4)); g = 0.0 if g < 0.0 else (1.0 if g > 1.0 else g)
        else:
            g = 1.0 - C6*w
        return T0*a*g

    @staticmethod
    def _surf(V, thA, w):
        """Vectorised _ander (identical formula; tests/test_human_conventions.py checks it)."""
        ang = np.maximum(0.0, np.cos(V["C2"]*(thA - V["C3"])))
        c4, c5 = V["C4"], V["C5"]; wp = np.maximum(w, 0.0)
        g = np.clip((2*c4*c5 + wp*(c5 - 3*c4))/(2*c4*c5 + wp*(2*c5 - 4*c4)), 0.0, 1.0)
        g = np.where(w < 0, 1.0 - V["C6"]*w, g)
        return V["T0"]*ang*g

    def f_angle(self, th):
        """th in QPOS coordinates; the curve is defined in the spec's anatomical convention."""
        out = np.empty(5)
        for i, (p, w, fl) in enumerate(self.ta):
            x = np.clip((self.qsign[i]*th[i]-p)/w, -1, 1)
            out[i] = fl + (1-fl)*np.cos(0.5*np.pi*x)**2
        return out

    def _geom_low(self, g):
        """True lowest world-z point of a geom. geom_rbound is the CIRCUMSCRIBED sphere
        radius and is wrong for boxes and long capsules; using it placed the trunk capsule's
        bounding sphere on the floor and drove the feet metres underground."""
        t = self.m.geom_type[g]; c = self.d.geom_xpos[g]; sz = self.m.geom_size[g]
        Rm = self.d.geom_xmat[g].reshape(3, 3)
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            ext = abs(Rm[2, 0])*sz[0]+abs(Rm[2, 1])*sz[1]+abs(Rm[2, 2])*sz[2]
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            ext = sz[0]
        elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            import plant as _P
            return _P.geom_low(self.m, self.d, g)          # single audited implementation
        else:
            import plant as _P
            return _P.geom_low(self.m, self.d, g)          # exact (mesh/ellipsoid) or error; never geom_rbound
        return float(c[2]-ext)

    def place(self, qj=None):
        mujoco.mj_resetData(self.m, self.d)
        if qj is not None:
            for i, v in zip(self.qadr, qj): self.d.qpos[i] = v
        mujoco.mj_forward(self.m, self.d)
        lo = min(self._geom_low(g) for g in range(self.m.ngeom) if self.m.geom_bodyid[g] != 0)
        self.d.qpos[1] -= lo; mujoco.mj_forward(self.m, self.d)
        return self.d.qpos.copy()

    def stand(self):
        return self.place(None)

    def com(self):
        return (self.m.body_mass[1:, None]*self.d.xipos[1:]).sum(0)/self.M
