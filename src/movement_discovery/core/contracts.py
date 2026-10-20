"""Typed contracts shared by every subsystem. Keep these small, explicit and versioned. Nothing here prescribes a movement.
Status ladder (per program and per scenario): DESIGNED -> SCAFFOLDED -> SOURCE_IMPLEMENTED -> INTEGRATED -> EXECUTED -> VALIDATED.
In this build phase nothing new may be marked EXECUTED or VALIDATED; a smoke rollout (plant builds, one tracked rollout, no
optimisation) is an INTEGRATION CHECK and leaves the status at INTEGRATED."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import json, hashlib

SCHEMA_VERSION = "md-1.0"


class Status(str, Enum):
    DESIGNED = "designed"; SCAFFOLDED = "scaffolded"; SOURCE_IMPLEMENTED = "source_implemented"; INTEGRATED = "integrated"; EXECUTED = "executed"; VALIDATED = "validated"


class Family(str, Enum):
    LOCOMOTION = "locomotion"; AERIAL = "aerial"; BALANCE = "balance"; POSTURE = "posture"; REACHING = "reaching"; PROJECTION = "projection"
    IMPACT = "impact"; FORCE_INTERACTION = "force_interaction"; PAYLOAD = "payload"; INTERCEPTION = "interception"; NEGOTIATION = "negotiation"; COMPOUND = "compound"


class Capability(str, Enum):
    """What an embodiment can offer; a program DECLARES what it needs; the adapter RESOLVES or refuses (never approximates silently)."""
    PALM_SITE_RIGHT = "palm_site_right"; PALM_SITE_LEFT = "palm_site_left"; FOOT_SITE_RIGHT = "foot_site_right"; FOOT_SITE_LEFT = "foot_site_left"
    WRIST_3DOF = "wrist_3dof"; WAIST_PITCH = "waist_pitch"; WAIST_YAW = "waist_yaw"; HIP_ROLL = "hip_roll"; ANKLE_ROLL = "ankle_roll"
    FINGER_GRASP = "finger_grasp"; WELD_GRASP = "weld_grasp"; TWO_HANDS = "two_hands"; BODY_COLLISION = "body_collision"; EFFECTOR_COLLIDER = "effector_collider"
    TORQUE_CONTROL = "torque_control"; SERVO_CONTROL = "servo_control"; FLOATING_BASE = "floating_base"


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass
class EmbodimentSpec:
    robot: str                                   # "G1" | "G1H" | "H1" (keys of branch_i.ROBOTS / throw.CFG)
    mjcf: str
    driven: List[str]                            # driven joint names for this program (resolved by the adapter)
    effectors: Dict[str, str] = field(default_factory=dict)      # role -> site name, e.g. {"hand_r": "right_palm", "foot_r": "right_foot"}
    feet: Tuple[str, str] = ("left_foot", "right_foot")
    capabilities: List[Capability] = field(default_factory=list)
    notes: str = ""
    def hash(self) -> str: return _hash(asdict(self))


@dataclass
class WorldSpec:
    """Declarative world: compiled by movement_discovery.world into an MJCF next to the robot assets. Objects/surfaces follow
    engine.world dict conventions (obj(), surface(), brush()); mechanisms/terrain/disturbances are new factories."""
    objects: List[dict] = field(default_factory=list)
    surfaces: List[dict] = field(default_factory=list)
    mechanisms: List[dict] = field(default_factory=list)       # door, drawer, crank, lever: dict(kind=..., **params)
    terrain: Optional[dict] = None                              # dict(kind="flat"|"slope"|"steps"|"uneven"|"platform_moving", **params)
    disturbances: List[dict] = field(default_factory=list)     # dict(kind="impulse"|"force"|"platform", t=..., vec=..., body=...)
    friction: Optional[float] = None
    gravity: Optional[Tuple[float, float, float]] = None
    tag: Optional[str] = None
    def hash(self) -> str: return _hash(asdict(self))


@dataclass
class GoalSpec:
    """The desired EXTERNAL outcome. Never a joint trajectory."""
    kind: str                     # "effector_velocity" | "object_velocity" | "object_displacement" | "com_displacement" | "pose" | "stability" | "contact_force" | "sequence" | ...
    params: Dict[str, Any] = field(default_factory=dict)
    tolerance: Dict[str, Any] = field(default_factory=dict)
    horizon: float = 0.8
    def hash(self) -> str: return _hash(asdict(self))


@dataclass
class Scenario:
    id: str; program_id: str; seed: int; params: Dict[str, Any]; version: str = SCHEMA_VERSION; status: Status = Status.DESIGNED
    def to_json(self) -> str: return json.dumps(asdict(self), sort_keys=True, default=str)


@dataclass
class Candidate:
    """A possible movement construction: representation name + parameters + provenance. The representation decides meaning."""
    representation: str           # "torque_knots+tracked_prior" | "torque_knots" | "servo_knots" | "contact_schedule+knots" | ...
    params: Dict[str, Any]
    duration: float
    provenance: Dict[str, Any] = field(default_factory=dict)     # generator, parent id, optimizer, seed, budget
    id: str = ""
    def __post_init__(self):
        if not self.id: self.id = _hash([self.representation, self.params, self.duration, self.provenance])


@dataclass
class ValidityReport:
    numerical: bool = True; joint_limits: bool = True; actuator_torque: bool = True; actuator_speed: bool = True; envelope: bool = True
    contact: bool = True; support: bool = True; direction: bool = True; object_interaction: bool = True; timestep_robust: Optional[bool] = None
    reasons: List[str] = field(default_factory=list); violation: float = 0.0
    @property
    def ok(self) -> bool: return self.violation <= 0 and not self.reasons


@dataclass
class Outcome:
    value: float                              # the task's scalar achievement (speed, displacement, error, ...); sign convention: higher is better unless program says otherwise
    achieved: bool
    measures: Dict[str, Any] = field(default_factory=dict)
    validity: ValidityReport = field(default_factory=ValidityReport)
    trace_ref: Optional[str] = None          # path of the full-resolution trace, when recorded
    mechanics: Optional[dict] = None         # engine.emergence-style summary (descriptive, never a filter)


@dataclass
class SearchRecord:
    optimizer: str; config: Dict[str, Any]; seed: int; budget_evals: int; best_id: str; history: List[dict] = field(default_factory=list); secs: float = 0.0


@dataclass
class ArchiveEntry:
    program_id: str; scenario_id: str; embodiment_hash: str; world_hash: str; candidate: Candidate; outcome: Outcome; search: Optional[SearchRecord]
    descriptors: Dict[str, float] = field(default_factory=dict); software: Dict[str, str] = field(default_factory=dict); version: str = SCHEMA_VERSION


@dataclass
class Step:
    """One external sub-goal executed by ONE reference task (engine.task.Task-shaped: EffectTask, ImpactTask, ReachTask-derived,
    BaseEffectTask, JumpTask, ...): `demand` is that task's demand dict; `held` the world objects carried INTO the step (the
    runner carries the previous step's held set forward, as engine.compose does); `world` an optional WorldSpec override (None =
    the program's world); `event` the name of the world condition that ends the sub-goal (documentation of the goal, never a
    movement); `representation` an optional construction name for this step (None = the program's strategy_space()[0]);
    `z` stored parameters (None = the representation's prior-only start z0)."""
    task: Any; demand: Dict[str, Any]; held: List[str] = field(default_factory=list); world: Optional[WorldSpec] = None
    event: Optional[str] = None; representation: Optional[str] = None; z: Optional[List[float]] = None; label: str = ""


@dataclass
class Plan:
    """THE demand shape every program returns (directly, or through as_plan from the legacy `(task, demand)` pair): an ordered list of
    Steps executed by the runner from REAL handed-off states (construction.state_map across differing plants, engine.core.handoff),
    `missing` the sub-goals without a bridge today (dict(goal, why): a scaffold names its missing asset instead of passing silently),
    and the between-step treatment of engine.compose (freeze_T stiff servos at the handed posture, settle_T balance settle toward the
    stance, settle_last). A single-step program is a one-step Plan. Nothing here is a movement sequence: steps are goals."""
    steps: List[Step]; robot: str = "G1"; missing: List[dict] = field(default_factory=list)
    settle_T: float = 0.0; freeze_T: float = 0.4; settle_last: bool = True; notes: str = ""
    @property
    def executable(self) -> bool: return bool(self.steps)
    @property
    def single(self) -> bool: return len(self.steps) == 1
    @property
    def task(self): return self.steps[0].task if self.steps else None
    @property
    def demand(self) -> Dict[str, Any]: return self.steps[0].demand if self.steps else {}
    def pairs(self) -> List[Tuple[Any, Dict[str, Any]]]: return [(s.task, s.demand) for s in self.steps]
    def missing_reasons(self) -> List[str]: return [f"MISSING:{m['goal']}" for m in self.missing]


def as_plan(ret, robot: Optional[str] = None) -> Plan:
    """Coerce a program's demand() result into a Plan: a Plan passes through; the legacy `(reference task, demand dict)` pair becomes a
    one-step Plan (held objects read from demand["held"]). Any other shape is a contract error (named, not guessed)."""
    if isinstance(ret, Plan): return ret
    if isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], dict) and ret[0] is not None:
        task, demand = ret; d = dict(demand)
        return Plan(steps=[Step(task=task, demand=d, held=list(d.get("held") or []))], robot=robot or getattr(task, "robot", "G1"))
    raise TypeError(f"demand() must return a Plan or (reference task, demand dict); got {type(ret).__name__}")


class TaskProgram:
    """Contract for one physical challenge. Subclasses own their physics; they must not encode a technique.
    Required class attributes: id, family, title, requires (list[Capability]), status (Status).
    demand(scenario) returns a Plan (core.contracts.Plan) or the legacy (reference task, demand dict) pair; plan(scenario) is what the
    runner reads and always yields a Plan (as_plan)."""
    id: str = ""; family: Family = Family.LOCOMOTION; title: str = ""; requires: List[Capability] = []; status: Status = Status.DESIGNED
    distinct_from: str = ""      # one sentence: why this is a different physical problem from its neighbours (the anti-duplicate rule)
    def embodiment(self, robot: str) -> EmbodimentSpec: raise NotImplementedError
    def world(self, scenario: Scenario) -> WorldSpec: raise NotImplementedError
    def sample_scenario(self, seed: int, **overrides) -> Scenario: raise NotImplementedError
    def goal(self, scenario: Scenario) -> GoalSpec: raise NotImplementedError
    def strategy_space(self, scenario: Scenario) -> List[str]:
        """Representations compatible with this task (names registered in movement_discovery.construction). No technique."""
        return ["torque_knots+tracked_prior", "torque_knots"]
    def evaluate(self, rec: dict, scenario: Scenario) -> Outcome: raise NotImplementedError
    def validity(self, rec: dict, scenario: Scenario) -> ValidityReport: raise NotImplementedError
    def observe(self, rec: dict, scenario: Scenario) -> Optional[dict]: return None
    def demand(self, scenario: Scenario): raise NotImplementedError
    def plan(self, scenario: Scenario) -> Plan:
        """The runner's entry point: the program's demand() coerced to a Plan (single-step programs: one Step)."""
        return as_plan(self.demand(scenario), robot=scenario.params.get("robot") if isinstance(scenario.params, dict) else None)
    def serialize(self) -> dict:
        return dict(id=self.id, family=self.family.value, title=self.title, requires=[c.value for c in self.requires], status=self.status.value, distinct_from=self.distinct_from, cls=f"{type(self).__module__}.{type(self).__name__}", version=SCHEMA_VERSION)
