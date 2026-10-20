import numpy as np, mujoco
import plant as P
G = 9.80665
from gate_a2 import NK, HOR, CROUCH
def rollout(h, params, lock=None, record=False):
    m = h.m; d = mujoco.MjData(m)
    q0 = h.place(CROUCH); d.qpos[:] = q0; d.qvel[:] = 0
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    mujoco.mj_forward(m, d)
    th0 = np.array([d.qpos[i] for i in h.qadr]); lim0 = h.tau*h.f_angle(th0)
    a = np.clip(h._hold/h.tau, -1, 1) if h._hold is not None else np.zeros(5)
    kn = np.asarray(params, float).reshape(5, NK).copy()
    if lock is not None:
        for i in lock: kn[i, :] = a[i]           # locked joint: keeps its holding torque
    nst = int(HOR/m.opt.timestep); per = nst//NK; dt = m.opt.timestep; alpha = dt/h.act_tau
    E0 = float(d.energy[0] + d.energy[1]); z0 = float(h.com()[2]); zmin = z0
    W = 0.0; work = np.zeros(5); imp = 0.0; vto = None; tto = None; Eto = None; Wto = None
    wlog = []; peak = 0.0; pen = 0.0
    for s in range(nst):
        k = min(s//per, NK-1); k2 = min(k+1, NK-1); f = (s % per)/per
        a += alpha*(kn[:, k] + (kn[:, k2]-kn[:, k])*f - a)
        th = np.array([d.qpos[i] for i in h.qadr]); w = np.array([d.qvel[i] for i in h.vadr])
        lim = h.tau*h.f_angle(th); sp = np.clip(1.0 - np.abs(w)/h.wmax, 0.0, 1.0)
        tau = np.clip(a*h.tau, -lim*np.where(w < 0, sp, 1.0), lim*np.where(w > 0, sp, 1.0))
        d.ctrl[:] = tau
        qv0 = np.array([d.qvel[i] for i in h.vadr])
        mujoco.mj_step(m, d)
        w = np.array([d.qvel[i] for i in h.vadr])
        W += float(np.dot(tau, 0.5*(qv0 + w)))*dt
        vz = float(P.com_velocity(m, d)[2]); zmin = min(zmin, float(h.com()[2]))
        fz = 0.0
        for ci in range(d.ncon):
            fv = np.zeros(6); mujoco.mj_contactForce(m, d, ci, fv); fz += fv[0]
            pen = max(pen, -float(d.contact[ci].dist))
        if vto is None:
            imp += fz*dt; peak = max(peak, fz); work += np.maximum(tau*w, 0)*dt
            if record: wlog.append(w.copy())
            if fz < 1.0 and vz > 0.0:
                vto = vz; tto = (s+1)*dt; Eto = float(d.energy[0] + d.energy[1]); Wto = W
                break
    if vto is None: vto = 0.0; tto = HOR; Eto = float(d.energy[0] + d.energy[1]); Wto = W
    mom_err = vto - (imp/h.M - G*tto)
    return dict(v=vto, height=max(vto, 0)**2/(2*G), T=tto, dip=z0 - zmin, peak_grf_bw=peak/(h.M*G),
                work=work, energy_residual=(Eto - E0) - Wto, W=Wto, momentum_error=mom_err,
                max_pen=pen, wlog=np.array(wlog) if record else None,
                valid=bool(abs(mom_err) < 0.05 and (Eto - E0) - Wto <= max(1.0, 0.02*abs(Wto)) and vto > 0))


