"""Shared helpers for the BODY families (locomotion, aerial, balance, posture, negotiation). MuJoCo-free at import.
RECORD FORMATS consumed by the evaluators (what exists today; contact_schedule rollouts from construction/ must emit the same keys):
  A) throw.rollout(record=True):  rec["trace"] = dict(q=[nq per step], qvel, torque, power, foot_load_bw=[[l, r] per step], dt), rec["reasons"], rec["violation"]
  B) JumpTask.rollout(record=True): rec["trace"] = dict(t, z (COM z), fz), rec["extra"]["land_trace"] = [(t, qpos)] every 10 steps after takeoff, rec["extra"]["landing"]
  C) balance bridge (balance/_local.py): rec["trace"] = dict(q, foot_load_bw, dt) sampled every 10 steps
World dicts come from the world/ factories (terrain / obstacles / disturbances; the pre-WORLD dict fallbacks are gone); validity flags
from the one table in physics.validity (classify)."""
from __future__ import annotations
import numpy as np
from movement_discovery.core.contracts import TaskProgram, Capability, Status, EmbodimentSpec, GoalSpec, Scenario, Outcome, ValidityReport
from movement_discovery.physics import validity as V
from movement_discovery.world import terrain as _WT, obstacles as _WO, disturbances as _WD

TORSO_BODY = "torso_link"


# ---------------------------------------------------------------- world dict constructors (thin names over the world/ factories)
def terrain(kind, **p):
    """world.terrain.<kind>(**p): flat, slope, steps, stairs, uneven, gap, platform, narrow_beam, friction_patches."""
    return getattr(_WT, kind)(**p)

def obstacle(kind, **p):
    """world.obstacles.<kind>(**p): barrier, low_clearance, corridor, opening, moving_obstacle (goes in WorldSpec.mechanisms)."""
    return getattr(_WO, kind)(**p)

def impulse(t, vec, body=TORSO_BODY):
    """vec = IMPULSE in N s (world.disturbances schema: applied on the step containing t0; t1 None)."""
    return _WD.impulse(t, vec, body)

def force(t0, t1, vec, body=TORSO_BODY): return _WD.force(t0, t1, vec, body)

def platform_motion(body="platform", axis=(0, 0, 1), amp=0.05, freq=1.0, phase=0.0, t0=0.0, t1=None): return _WD.platform_motion(body, axis, amp, freq, phase, t0, t1)


# ---------------------------------------------------------------- trace readers (pelvis = free joint qpos[0:7])
def q_trace(rec):
    """(q [n, nq], dt) from any of the record formats, or (None, None)."""
    if not rec: return None, None
    tr = rec.get("trace")
    if isinstance(tr, dict) and tr.get("q") is not None and not isinstance(tr["q"], dict):
        return np.asarray(tr["q"], float), float(tr.get("dt", 0.001))
    lt = (rec.get("extra") or {}).get("land_trace")
    if lt: return np.asarray([q for _, q in lt], float), float(lt[1][0] - lt[0][0]) if len(lt) > 1 else 0.02
    return None, None

def foot_loads(rec):
    tr = (rec or {}).get("trace") or {}
    if isinstance(tr, dict) and tr.get("foot_load_bw") is not None: return np.asarray(tr["foot_load_bw"], float)
    if isinstance(tr, dict) and tr.get("fz") is not None: fz = np.asarray(tr["fz"], float); return np.stack([fz, fz], 1)/2.0   # total only
    return None

def quat_to_yaw_tilt(q4):
    w, x, y, z = q4; yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    up_z = 1 - 2*(x*x + y*y)                     # world z component of the body z axis
    return float(yaw), float(np.arccos(np.clip(up_z, -1.0, 1.0)))

def pelvis_kin(q, dt):
    """xyz [n,3], yaw [n], tilt [n] (rad, 0 = body z up), vxy [n,2] from the free-joint part of the trace."""
    xyz = q[:, :3]; yt = np.array([quat_to_yaw_tilt(r[3:7]) for r in q]); yaw = np.unwrap(yt[:, 0]); tilt = yt[:, 1]
    v = np.gradient(xyz, dt, axis=0) if len(q) > 1 else np.zeros_like(xyz)
    return xyz, yaw, tilt, v[:, :2]

def upright(q, dt, T=0.5, tilt_max=np.radians(30), z_frac=0.7):
    """Upright over the last T s: pelvis tilt < tilt_max and pelvis height > z_frac * initial height. Returns (ok, measures)."""
    xyz, yaw, tilt, _ = pelvis_kin(q, dt); n = max(1, int(round(T/dt))); w = slice(max(0, len(q) - n), len(q)); z0 = float(xyz[0, 2])
    ok = bool(np.all(tilt[w] < tilt_max) and np.all(xyz[w, 2] > z_frac*z0))
    return ok, dict(tilt_max_deg=float(np.degrees(tilt.max())), tilt_end_deg=float(np.degrees(tilt[-1])), z_end=float(xyz[-1, 2]), z_min=float(xyz[:, 2].min()), z0=z0)

def flight_phases(loads, dt, confirm=0.02):
    """Count confirmed flights (both feet unloaded for >= confirm s) in a foot-load trace [[l, r], ...]."""
    tot = loads.sum(1) if loads.ndim == 2 else loads; off = tot < 1e-3; n = 0; run = 0.0; counted = False
    for o in off:
        if o:
            run += dt
            if run >= confirm - 1e-12 and not counted: n += 1; counted = True
        else: run = 0.0; counted = False
    return n

def base_validity(rec, **flags):
    """physics.validity.classify over the record (preset flags, then the one reason -> flag table)."""
    return V.classify(rec or {}, **flags)

def no_trace(reason="NO_TRACE"):
    return Outcome(value=float("nan"), achieved=False, validity=ValidityReport(reasons=[reason], violation=1.0))

def fail(v: ValidityReport, reason, amount=1.0):
    v.reasons.append(reason); v.violation += float(amount); return v


# ---------------------------------------------------------------- program base for the BODY families
class BodyProgram(TaskProgram):
    """Defaults: embodiment via the adapter (driven set optional), validity = evaluate().validity, observe = engine.emergence on
    throw-format traces (joint chain of `observe_site`) else a pelvis-kinematics summary, serialize adds the strategy space."""
    driven: list | None = None; observe_site = "right_foot"
    def embodiment(self, robot) -> EmbodimentSpec:
        from movement_discovery.embodiment import resolve; return resolve(robot, self, driven=self.driven)
    def scenario(self, seed, params, **ov):
        params.update(ov); return Scenario(id=f"{self.id}_s{seed}", program_id=self.id, seed=seed, params=params)
    def validity(self, rec, scenario): return self.evaluate(rec, scenario).validity
    def observe(self, rec, scenario):
        tr = (rec or {}).get("trace") or {}
        if isinstance(tr, dict) and tr.get("torque") is not None and hasattr(self, "demand"):
            from engine import emergence as EM; t, d = self.demand(scenario); return EM.analyze(t.plant(d), rec, self.observe_site)
        q, dt = q_trace(rec)
        if q is None: return None
        xyz, yaw, tilt, v = pelvis_kin(q, dt); sp = np.linalg.norm(v, axis=1)
        return dict(kind="pelvis_kinematics", path_m=float(np.sum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))), peak_speed=float(sp.max()), z_range=[float(xyz[:, 2].min()), float(xyz[:, 2].max())], yaw_change_deg=float(np.degrees(yaw[-1] - yaw[0])), tilt_max_deg=float(np.degrees(tilt.max())), duration=float(len(q)*dt))
    def serialize(self):
        sc = self.sample_scenario(0); return super().serialize() | dict(strategy_space=self.strategy_space(sc), goal_kind=self.goal(sc).kind, world=self.world(sc).hash(), driven=self.driven)


CS_NAME = "contact_schedule+knots"      # construction.contact_schedule.ContactSchedule.name

def schedule_demand(scenario, horizon, phases, goal: GoalSpec, notes="", allowed=("both", "left", "right"), effect=None):
    """Demand for the contact_schedule+knots representation (construction/). The PROGRAM states the horizon RANGE, the
    support contracts a phase MAY use, and the goal per declared phase; the actual sequence of contracts and per-phase horizons is
    sampled by `sample_schedule` (random, a search variable of the runner) -- never authored here. `effect` = the base-level effect the
    EffectTask wrapper should chase (effector 'pelvis', direction (elev, az)); the representation needs a base-displacement mode for it."""
    sched = dict(representation=CS_NAME, horizon=[float(horizon[0]), float(horizon[1])], allowed_supports=list(allowed), declared_phases=list(phases), goal=goal, effect=effect, scenario=scenario.id, notes=notes)
    sched["phases"] = sample_schedule(sched, scenario.seed)       # one sampled schedule (the runner's search variable), in the representation's phase format
    return schedule_task(scenario, sched)

def schedule_task(scenario, sched):
    """(reference task, demand) for the runner: the contact_schedule+knots representation around BaseEffectTask
    (construction/base_effect.py: effect on the pelvis/COM, measure displacement or speed, cone on the base motion)."""
    from movement_discovery.construction.base_effect import BaseEffectTask
    t = BaseEffectTask(scenario.params.get("robot", "G1")); eff = sched.get("effect") or {}
    d = dict(t.demand_default(), **sched, effector=eff.get("effector", "pelvis"), direction=eff.get("direction"), measure=eff.get("measure", "displacement"), support="both", prior=None)
    return t, d

def sample_schedule(demand, seed, n_phases=(1, 6), min_phase=0.3):
    """One random schedule inside the demand: n phases, supports drawn from allowed_supports, horizons summing inside the range."""
    rng = np.random.default_rng(seed); H = float(rng.uniform(*demand["horizon"])); n = int(rng.integers(n_phases[0], min(n_phases[1], int(H/min_phase)) + 1))
    n = max(1, min(n, int(H/min_phase))); hs = min_phase + rng.dirichlet(np.ones(n))*(H - n*min_phase)
    return [dict(support=str(rng.choice(demand["allowed_supports"])), horizon=float(h), prior=None) for h in hs]

def phase(support, goal, **extra):
    """support: 'both' | 'left' | 'right' | 'feet_any' (any sequence of the allowed contracts, flight allowed) | 'any' (hands allowed)."""
    return dict(support=support, goal=goal, **extra)
