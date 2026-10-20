"""Overarm throw as an engine Task (wraps src/throw.py). Demand: {"target_speed": None|float, "rom_scale": 1.0, "prior": True}.
value = ball speed at release. rom_scale shrinks the arm/trunk/wrist joint ranges about the stance (the form-factor axis)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pickle, mujoco, branch_i as BI, throw as T
from engine import core
from engine.task import Task
PRIOR = f"{core.RES}/phase11/prior_G1.pkl"
ROM_JOINTS = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
              "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint"]


class ThrowTask(Task):
    name = "throw"
    def __init__(self, robot="G1"): self.robot = robot; self._prior = None; self._plants = {}
    def demand_default(self): return dict(target_speed=None, rom_scale=1.0, prior=True, omega=20.0)
    def prior(self):
        if self._prior is None: self._prior = pickle.load(open(PRIOR, "rb"))
        return self._prior
    def plant(self, demand, dt=core.DT_OPT, fresh=False):
        rom = float(demand.get("rom_scale", 1.0)); key = (dt, rom, fresh)
        if key in self._plants: return self._plants[key]
        S0 = T.setup(self.robot)
        S = T.setup(self.robot, start=np.array(S0["q0"]) + (1e-9*(1 + len(self._plants)))) if (fresh or rom != 1.0) else S0   # uncached copy
        if rom != 1.0:
            m = S["m"]; q0 = np.array(S["q0"])
            for jn in ROM_JOINTS:
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn); a = m.jnt_qposadr[j]; lo, hi = m.jnt_range[j]
                m.jnt_range[j] = [q0[a] + rom*(lo - q0[a]), q0[a] + rom*(hi - q0[a])]
                for act in range(m.nu):
                    if m.actuator_trnid[act][0] == j and m.actuator_ctrllimited[act]: m.actuator_ctrlrange[act] = m.jnt_range[j]
        BI.apply_dt(S, dt); BI.settle(S); self._plants[key] = S; return S
    def n_params(self, S, demand): return len(S["types"])*T.NK + 1 + (2 if demand.get("prior", True) else 0)
    def unpack(self, S, demand, z):
        z = np.asarray(z); n = len(S["types"])*T.NK + 1
        p = dict(x=z[:n])
        if demand.get("prior", True): p["amp"] = 0.2 + 0.8*(float(z[n]) + 1)/2; p["tscale"] = 0.87 + 0.3*float(z[n+1]); p["omega"] = float(demand.get("omega", 20.0))
        return p
    def controller(self, S, demand, p):
        if not demand.get("prior", True): return None
        ref = self.prior()
        if demand.get("direction") is not None:
            ref = T.windup_reference(S, direction=demand["direction"])
        return T.tracker(S, T.scaled_ref(ref, p["amp"]), p["omega"], p["tscale"])
    def rollout(self, S, demand, p, ctrl_fn=None, record=False):
        r = T.rollout(np.asarray(p["x"]), S, ctrl_fn=ctrl_fn, record=record, caps=(np.asarray(demand["_caps"]) if demand.get("_caps") else None))
        # the direction cone is a BODY-frame demand: azimuth relative to the base heading at the start of the rollout
        # (after a composer handoff the robot may not face world +x)
        qw, qx, qy, qz = np.asarray(S["q_settled"])[3:7]; yaw0 = float(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
        az_rel = float((r["az"] - yaw0 + np.pi) % (2*np.pi) - np.pi); r["az_world"] = r["az"]; r["az"] = az_rel
        if demand.get("direction") is None:
            elev_range, az_limit = T.ELEV, T.AZ
        else:
            ec, ac = map(float, demand["direction"])
            et = float(demand.get("elev_tol", np.radians(5.0))); az_limit = float(demand.get("az_tol", np.radians(5.0)))
            elev_range = (ec - et, ec + et)
        r["in_cone"] = bool((elev_range[0] <= r["elev"] <= elev_range[1]) and abs(az_rel - (float(demand["direction"][1]) if demand.get("direction") is not None else 0.0)) <= az_limit)
        if demand.get("ignore_cone"): r["in_cone"] = True                    # impact tasks judge the OBJECT's direction, not the effector's
        pen = T.penalty(r, target=None, cone=(elev_range, az_limit)); viol = max(0.0, pen + 0.01*r["speed"] - 100.0) if pen >= 100.0 else 0.0
        reasons = r["reasons"] + ([] if r["in_cone"] else ["OUT_OF_CONE"])
        if "OVERSPEED" in r["reasons"]: viol += 1.0 + 10.0*max(0.0, r["overspeed"] - T.OVERSPEED_TOL)   # a joint past 110% of its rated speed invalidates the throw (graded for the search)
        # RECOVERABLE THROW (J22): the release state must be one a standing controller can recover: COM descent bounded
        # (the speed-only throw dives at -2.45 m/s) and the capture point near the feet centre. Graded violation for the search.
        if demand.get("recoverable"):
            # v2: the whole COM velocity is bounded (a release with the COM rising 1.5 m/s and moving 1.4 m/s sideways passed
            # the descent-only v1 and was still a launch no controller recovered), the capture point stays near the feet
            # centre and BOTH feet carry load at release
            v_max = float(demand.get("vcom_max", 0.6)); cp_max = float(demand.get("cp_max", 0.10)); f_min = float(demand.get("feet_min_bw", 0.10))
            over = max(0.0, r["com_speed"] - v_max) + max(0.0, r["cp_err_rel"] - cp_max)*5.0 + max(0.0, f_min - min(r["feet_bw"]))*5.0
            if over > 0: viol += 1.0 + over; reasons = reasons + ["NOT_RECOVERABLE"]
        return dict(value=float(r["speed"]), valid=(viol <= 0), reasons=reasons, violation=float(viol), trace=r.get("trace"), final_state=r.get("final_state"),
                    cost=float(sum(r["work"].values())), extra=dict(elev=r["elev"], az=r["az"], t_rel=r["t_rel"], work=r["work"], participation=r["participation"], com_speed=r["com_speed"], vcom_z=r["vcom_z"], cp_err_rel=r["cp_err_rel"], com_speed_rel=r["com_speed"], feet_bw_rel=min(r["feet_bw"]), overspeed=r["overspeed"], overspeed_by_joint=r["overspeed_by_joint"], amp=p.get("amp"), tscale=p.get("tscale"), omega=p.get("omega"), final_state=r.get("final_state"), trace=r.get("trace")))
    def objective(self, rec, demand):
        tg = demand.get("target_speed")
        if tg is None: return -rec["value"]
        return abs(rec["value"] - tg)*5.0 + 0.002*rec["cost"]
    def warm(self, S, demand, stored):
        z = list(np.asarray(stored["x"]))
        if demand.get("prior", True): z += [2*(stored["amp"] - 0.2)/0.8 - 1, (stored["tscale"] - 0.87)/0.3]
        return np.array(z)
