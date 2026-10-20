"""Local helpers for the projection family (2026-10-12). Flight is MODELLED from the release state of the
reference construction (tasks/effect.py: point mass merged at the effector, released at t_rel): drag-free ballistics from the
release position and velocity, a floor/wall bounce with a scenario restitution, and a rolling run-out with a rolling-resistance
coefficient. No free ball is simulated after release (the plant releases nothing). MuJoCo only inside
functions. Never edit world/: this file supplies what the projection programs need until WORLD's factories land."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import ValidityReport, Outcome
from movement_discovery.physics import validity as V
G = 9.80665


def release_state(S, rec):
    """Effector position and velocity (world) at release, recomputed from the recorded final state (exact, not from the cone angles)."""
    import mujoco
    m = S["m"]; d = mujoco.MjData(m); q, v = rec["extra"]["final_state"]; d.qpos[:] = q; d.qvel[:] = v; mujoco.mj_forward(m, d)
    v6 = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, S["site"], v6, 0)
    return dict(pos=np.array(d.site_xpos[S["site"]], float), v=np.array(v6[3:6], float))


def flight_time(z0, vz, z_land=0.0):
    disc = vz*vz + 2*G*max(z0 - z_land, 0.0); return (vz + np.sqrt(max(disc, 0.0)))/G


def landing(pos, v, z_land=0.0):
    """Drag-free landing point on the plane z = z_land, flight time and landing velocity."""
    tf = flight_time(pos[2], v[2], z_land); p = np.array([pos[0] + v[0]*tf, pos[1] + v[1]*tf, z_land]); vl = np.array([v[0], v[1], v[2] - G*tf]); return p, float(tf), vl


def height_at_range(pos, v, x_plane):
    """Height of the drag-free trajectory where it crosses the vertical plane x = x_plane (None if it never gets there)."""
    if v[0] <= 1e-6 or x_plane <= pos[0]: return None
    t = (x_plane - pos[0])/v[0]; return float(pos[2] + v[2]*t - 0.5*G*t*t)


def position_at(pos, v, t): return np.array([pos[0] + v[0]*t, pos[1] + v[1]*t, pos[2] + v[2]*t - 0.5*G*t*t])


def bounce(pos, v, e, mu=0.0, wall_x=None):
    """One floor bounce (or a wall bounce at x = wall_x if reached first) with normal restitution e and a tangential loss mu;
    returns the post-bounce state (pos, v) and where the bounce happened."""
    p_f, t_f, v_f = landing(pos, v, 0.0)
    if wall_x is not None and v[0] > 1e-6 and (wall_x - pos[0])/v[0] < t_f:
        t_w = (wall_x - pos[0])/v[0]; p_w = position_at(pos, v, t_w); v_w = np.array([v[0], v[1], v[2] - G*t_w])
        return p_w, np.array([-e*v_w[0], (1 - mu)*v_w[1], (1 - mu)*v_w[2]]), "wall"
    return p_f, np.array([(1 - mu)*v_f[0], (1 - mu)*v_f[1], -e*v_f[2]]), "floor"


def rollout_distance(v_h, c_rr):
    """Run-out of a rolling ball decelerated by rolling resistance c_rr*g (m)."""
    return float(v_h*v_h/(2.0*max(c_rr, 1e-4)*G))


def effect_validity(rec, extra_reasons=()):
    """physics.validity.classify (the one reason -> flag table); extra reasons appended as given (duplicates kept, as before)."""
    v = V.classify(rec); v.reasons = list(rec["reasons"]) + list(extra_reasons)
    for f, val in V.flags(v.reasons).items(): setattr(v, f, val)
    return v


def outcome(value, rec, measures, extra_reasons=(), extra_violation=0.0):
    v = effect_validity(rec, extra_reasons); v.violation += float(extra_violation)
    return Outcome(value=float(value), achieved=v.ok, measures=measures, validity=v)


def bridge(scenario, **demand):
    """EffectTask bridge: the reference construction with the scenario's mass/support/direction. A wide cone (direction given
    with large tolerances) keeps the body-derived prior pointed at the target while the landing error, not the cone, judges."""
    from tasks.effect import EffectTask
    robot = scenario.params.get("robot", "G1"); t = StartPoseEffectTask(robot) if demand.get("start_key") else EffectTask(robot)
    return t, dict(t.demand_default(), **demand)


class StartPoseEffectTask:
    """EffectTask whose plant starts from a named keyframe of the robot model (constrained initial posture, e.g. the G1 'knees_bent'
    crouch). Built lazily so the module stays MuJoCo-free at import."""
    def __new__(cls, robot="G1"):
        from tasks.effect import EffectTask
        import throw as T, branch_i as BI, mujoco
        class _Start(EffectTask):
            def plant(self, demand, dt=None, fresh=False):
                from engine import core
                dt = core.DT_OPT if dt is None else dt; key = ("start", dt, fresh, demand.get("effector"), float(demand.get("mass", 0.0)), demand.get("support", "both"), demand.get("start_key"))
                if key in self._plants: return self._plants[key]
                kw = dict(site=demand.get("effector"), ball=False, mass=float(demand.get("mass", 0.0)), support=demand.get("support", "both"), driven=demand.get("driven"))
                Sb = T.setup(self.robot, **kw); m = Sb["m"]; kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, demand["start_key"]); assert kid >= 0, demand["start_key"]
                S = T.setup(self.robot, start=np.array(m.key_qpos[kid]) + 1e-9*(1 + len(self._plants)), **kw); BI.apply_dt(S, dt); BI.settle(S); S["_setup_kw"] = kw; self._plants[key] = S; return S
        return _Start(robot)

