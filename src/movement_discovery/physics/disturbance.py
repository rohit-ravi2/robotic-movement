"""Scheduled external disturbances applied through d.xfrc_applied during a rollout. Interface for the world factories ():
    apply(disturbances, m, d, t)        disturbance = dict(kind="impulse"|"force"|"torque", t0, t1, vec, body[, local=False])
    apply_power(disturbances, m, d, t)  apply() and return the wrenches' power (W) for external-work accounting
    wrap(ctrl_fn, disturbances, n_out)  a ctrl_fn that applies them then defers to the inner controller (None -> zeros)
"impulse": vec is an impulse (N s) delivered as a constant force over [t0, t1] (default 20 ms); "force": vec in N over [t0, t1];
"torque": vec in N m. Bodies by name (resolved once). Note the throw/reach rollouts read d.qfrc_applied for the energy ledger and
branch_i.backemf_brake OVERWRITES qfrc_applied on the driven dofs each step; xfrc_applied is separate and is accounted in W_pas
only through qfrc_applied, so a disturbed rollout may report ENERGY_CLOSURE: the validity policy for disturbed scenarios should
read `extra["disturbance_work"]` recorded by telemetry rather than the closure gate (documented limitation of the frozen rollouts)."""
from __future__ import annotations
from typing import Callable, List, Optional
import numpy as np

_BODY = {}


def _bid(m, body):
    import mujoco
    k = (id(m), body)
    if k not in _BODY: _BODY[k] = int(body) if isinstance(body, (int, np.integer)) else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    return _BODY[k]


def force_at(dist: dict, t: float, dt: float = 0.001) -> Optional[np.ndarray]:
    """WORLD schema: impulse = vec (N s) as F = vec/(t1 - t0) over [t0, t1), t1 None -> the single step containing t0 (F = vec/dt);
    force/torque = vec (N / N m) over [t0, t1) (t1 None -> 0.1 s)."""
    if dist.get("kind", "force") not in ("impulse", "force", "torque"): return None       # slip_patch (static), platform_motion (mocap)
    t0 = float(dist.get("t0", 0.0)); t1 = dist.get("t1"); imp = dist.get("kind") == "impulse"
    t1 = float(t1) if t1 is not None else (t0 + dt if imp else t0 + 0.1)
    if not (t0 - 1e-9 <= t < t1 - 1e-9): return None
    v = np.asarray(dist["vec"], float)
    return v/max(t1 - t0, 1e-9) if imp else v


def apply(disturbances: List[dict], m, d, t: float) -> None:
    """Zero xfrc_applied and add every active disturbance (world frame unless local=True: rotated by the body's xmat)."""
    d.xfrc_applied[:] = 0.0
    for dist in disturbances or ():
        f = force_at(dist, t, float(m.opt.timestep))
        if f is None: continue
        b = _bid(m, dist.get("body", "torso_link"))
        if b < 0: continue
        if dist.get("local"): f = d.xmat[b].reshape(3, 3) @ f
        if dist.get("kind") == "torque": d.xfrc_applied[b, 3:6] += f
        else: d.xfrc_applied[b, 0:3] += f


def apply_power(disturbances: List[dict], m, d, t: float) -> float:
    """apply() + the wrenches' instantaneous POWER (W) from xfrc_applied and the body twists, so a rollout can integrate the external work
    (throw.rollout's energy gate credits actuator work only: an external wrench is an 'unexplained' residual there by construction;
    physics.validity.rejudge_energy_closure re-judges it with this work). The one disturbance applier of the families (interception)."""
    import mujoco
    apply(disturbances, m, d, t); P_ext = 0.0; v6 = np.zeros(6)
    for b in np.nonzero(np.any(d.xfrc_applied != 0.0, axis=1))[0]:
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, int(b), v6, 0); P_ext += float(d.xfrc_applied[b, :3] @ v6[3:] + d.xfrc_applied[b, 3:] @ v6[:3])
    return P_ext


def wrap(ctrl_fn: Optional[Callable], disturbances: List[dict], n_out: int) -> Callable:
    def fn(t, d, S):
        apply(disturbances, S["m"], d, t)
        return ctrl_fn(t, d, S) if ctrl_fn is not None else np.zeros(n_out)
    fn.disturbances = list(disturbances); return fn


def schedule_summary(disturbances: List[dict]) -> List[str]:
    return [f"{d.get('kind', 'force')} {np.round(d['vec'], 2).tolist()} on {d.get('body', 'torso_link')} [{d.get('t0', 0.0):.3f}, {d.get('t1', '-')}]" for d in disturbances or ()]
