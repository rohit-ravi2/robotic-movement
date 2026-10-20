"""Bisection step 1: run the simulator-agnostic controller spec (the exact loop used for Drake)
inside the ORIGINAL MuJoCo plant. If it reproduces BI.rollout's trace, the control-law port is
exact and any Drake difference lies in the model export or the physics."""
import sys, os, pickle, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, branch_i as BI, mocap_baseline as MB, plant as P
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xsim_drake_score import score

def loop(S, sp, act0=None):
    m = S["m"]; d = mujoco.MjData(m); BI.settle(S); BI._set_capacity_limits(m, S, None)
    d.qpos[:] = S["q_settled"]; d.qvel[:] = S["v_settled"]; mujoco.mj_forward(m, d)
    J = S["jarr"]
    for (a_, dof, qad, tau, kp, kd) in S["held"]: d.ctrl[a_] = S["q_settled"][qad]
    kn = sp["kn"]; NK = kn.shape[1]; ti = sp["ti"]; sgn = sp["sgn"]; tau = sp["tau"]; om = sp["om"]
    act = (sp["act0"] if act0 is None else act0).copy(); alpha = sp["alpha"]; dt = sp["dt"]; kd_ = sp["knot_dt"]
    t_, vz_, fz_ = [], [], []; fl = S["floor"]
    for s in range(int(sp["horizon"]/dt)):
        tt = s*dt; k = min(int(tt/kd_ + 1e-9), NK-1); k2 = min(k+1, NK-1)
        f = min(1.0, (tt - k*kd_)/kd_) if k < NK-1 else 0.0
        cmd = (kn[:, k] + (kn[:, k2]-kn[:, k])*f)[ti]*sgn
        q = d.qpos[J["qad"]]; w = d.qvel[J["dof"]]
        i = min(int(round(tt/dt)), len(sp["QR"]) - 1)
        cmd = cmd + (sp["tr_kp"]*(sp["QR"][i] - q) + sp["tr_kd"]*(sp["QDR"][i] - w))/tau
        act += alpha*(cmd - act)
        frac = np.maximum(0.0, 1.0 - np.abs(w)/om)
        hi = np.where(w > 0, tau*frac, tau); lo = np.where(w < 0, tau*frac, tau)
        d.ctrl[J["aid"]] = np.clip(act*tau, -lo, hi)
        mujoco.mj_step(m, d)
        vz = float(P.com_velocity(m, d)[2]); fz = 0.0
        for ci in range(d.ncon):
            c = d.contact[ci]
            if fl in (c.geom1, c.geom2):
                fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv); fz += fv[0]
        t_.append(s*dt); vz_.append(vz); fz_.append(fz)
    return score(np.array(t_), np.array(vz_), np.array(fz_))

if __name__ == "__main__":
    for f in sorted(glob.glob("../results/phase10/xsim/*.pkl"))[::4]:
        Pk = pickle.load(open(f, "rb")); meta = Pk["meta"]
        c = MB.load_clip(meta["clip"], "retargeted_g1_all.pkl")
        S = BI.setup(c["start"]); BI.apply_dt(S, 0.001)
        v, t, n = loop(S, Pk["spec"])
        BI.apply_dt(S, 0.004)
        print(os.path.basename(f)[:-4], "BI.rollout", round(meta["v_mujoco"], 3), "spec-loop in MuJoCo", round(v, 3), flush=True)
