"""Deterministic scenario generation. For every registered program: `base_seeds` seeded samples on the G1, `alt_seeds` on each
other robot the adapter resolves (embodiment.resolve capability check; a refusal is recorded, never approximated), and explicit
FACTORIAL overrides (RECIPES) applied through program.sample_scenario(seed, **overrides) only when the recipe's keys exist in that
program's sampled params (so an override is never inert by construction). Rows are de-duplicated by (program, robot, params) hash.
Capacity perturbations (tau_scale / jnt_ranges) use the embodiment helpers' plain-dict conventions (EffectTask demand keys) and are
emitted only for EffectTask-backed families (projection, impact); runner.assemble forwards them from scenario.params into every Plan
step's demand (embodiment.capacity.apply; the check line reports which the step's task consumes). MuJoCo-free unless robots other
than G1 are requested (the capability probe loads the models once)."""
from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import math
from movement_discovery.core.contracts import SCHEMA_VERSION, Scenario, Status, _hash
from movement_discovery.core import registry

ROBOTS = ("G1", "G1H", "H1")
BASE_SEEDS = 4; ALT_SEEDS = 2
R = math.radians


def _vec(J: float, a: float) -> List[float]: return [J*math.cos(a), J*math.sin(a), 0.0]


# ---------------------------------------------------------------- factorial recipes: (name, applies(params) -> bool, variants(params) -> [overrides])
def _directions(p):
    return [dict(elev=R(10.0), az=0.0), dict(elev=R(35.0), az=R(-30.0)), dict(elev=R(20.0), az=R(30.0))]

def _direction_pair(p):
    return [dict(direction=[R(10.0), 0.0]), dict(direction=[R(35.0), R(-30.0)]), dict(direction=[R(20.0), R(30.0)])]

def _masses(p): return [dict(mass=round(float(p["mass"])*f, 4)) for f in (0.5, 2.0)]

def _supports(p): return [dict(support=s) for s in ("left", "right") if s != p.get("support")]

def _impulse(p):
    out = []
    for f, da in ((0.5, 0.0), (2.0, 0.0), (1.0, math.pi/2)):
        J = float(p["J"])*f; a = float(p["dir"]) + da; out.append(dict(J=J, dir=a, Jvec=_vec(J, a)))
    return out

def _force(p):
    out = []
    for f in (0.5, 2.0):
        F = float(p["Fmag"])*f; a = float(p["dir"]); out.append(dict(Fmag=F, F=_vec(F, a)))
    return out

def _dist_force(p): return [dict(F=float(p["F"])*f) for f in (0.5, 2.0)]

def _friction(p):
    k = "friction" if "friction" in p else "mu"; return [{k: v} for v in (0.3, 0.9) if abs(float(p[k]) - v) > 0.05]

def _slope(p): return [dict(angle=R(a)) for a in (5.0, 15.0) if abs(float(p["angle"]) - R(a)) > R(1.0)]

def _steps(p): return [dict(rise=0.06, n=3), dict(rise=0.15, n=5)]

def _uneven(p): return [dict(amp=0.02), dict(amp=0.08)]

def _platform(p): return [dict(amp=float(p["amp"])*2.0), dict(axis=[1.0, 0.0, 0.0])]

def _offset(p): return [dict(offset=float(p["offset"])*f) for f in (0.5, 2.0)]

def _payload_side(p): return [dict(site="left_palm")] if p.get("site") == "right_palm" else []

def _object_mass(p):
    o = dict(p["object"]); return [dict(object=dict(o, mass=float(o["mass"])*f)) for f in (0.5, 3.0)]

TAU_SCALES = {"arm_x0.5": dict(shoulder=0.5, elbow=0.5, wrist=0.5), "legs_trunk_x0.5": dict(legs=0.5, trunk=0.5)}
HUMAN_SHOULDER = {"right_shoulder_pitch_joint": (-3.09, 1.05), "right_shoulder_yaw_joint": (-1.57, 1.57), "right_elbow_joint": (-0.17, 2.09)}   # embodiment.adapter.HUMAN_SHOULDER (G1 names; adapter.joint_ranges re-keys for H1)

def _capacity(p):
    from movement_discovery.embodiment import torque_scale
    out = [dict(tau_scale=torque_scale(**g)) for g in TAU_SCALES.values()]
    out.append(dict(jnt_ranges={k: list(v) for k, v in HUMAN_SHOULDER.items()})); return out

RECIPES: List[Tuple[str, Callable[[dict], bool], Callable[[dict], List[dict]]]] = [
    ("direction", lambda p: "elev" in p and "az" in p, _directions),
    ("direction", lambda p: isinstance(p.get("direction"), (list, tuple)) and len(p["direction"]) == 2, _direction_pair),
    ("mass", lambda p: isinstance(p.get("mass"), (int, float)) and p["mass"] > 0, _masses),
    ("support", lambda p: p.get("support") in ("both", "left", "right"), _supports),
    ("impulse", lambda p: "J" in p and "Jvec" in p and "dir" in p, _impulse),
    ("force", lambda p: "Fmag" in p and "F" in p and "dir" in p, _force),
    ("disturbance_force", lambda p: isinstance(p.get("F"), (int, float)), _dist_force),
    ("friction", lambda p: isinstance(p.get("friction"), (int, float)) or isinstance(p.get("mu"), (int, float)), _friction),
    ("slope", lambda p: "angle" in p and "dist" in p, _slope),
    ("steps", lambda p: "rise" in p and "n" in p and "run" in p, _steps),
    ("uneven", lambda p: "terrain_seed" in p and "amp" in p, _uneven),
    ("platform", lambda p: "axis" in p and "amp" in p and "freq" in p, _platform),
    ("payload_asymmetry", lambda p: isinstance(p.get("offset"), (int, float)), _offset),
    ("payload_side", lambda p: p.get("site") == "right_palm" and "mass" in p, _payload_side),
    ("object_mass", lambda p: isinstance(p.get("object"), dict) and "mass" in p["object"], _object_mass),
]
CAPACITY_FAMILIES = ("projection", "impact")


# ---------------------------------------------------------------- robots
_CAPS: Dict[str, object] = {}

def robot_ok(robot: str, program) -> Tuple[bool, str]:
    """(resolves, note) from embodiment.resolve; the model probe (MuJoCo) runs once per robot and is cached."""
    from movement_discovery import embodiment as E
    try: E.resolve(robot, program); return True, ""
    except E.UnsupportedCapability as e: return False, "missing " + ",".join(m.value for m in e.missing)
    except Exception as e: return False, f"{type(e).__name__}: {str(e)[:120]}"


# ---------------------------------------------------------------- rows
def scenario_row(program, sc: Scenario, robot: str, recipe: str = "seed", overrides: Optional[dict] = None, base_id: str = "") -> dict:
    params = dict(sc.params); ph = _hash([program.id, robot, params])
    return dict(id=f"{program.id}_{robot}_{ph}", program_id=program.id, family=program.family.value, robot=robot, seed=int(sc.seed), params=params,
                version=SCHEMA_VERSION, status="defined", recipe=recipe, overrides=dict(overrides or {}), params_hash=ph, base_id=base_id or sc.id)


def _sample(program, seed: int, robot: str, **ov) -> Scenario:
    return program.sample_scenario(int(seed), robot=robot, **ov)


def generate_program(program, robots: Sequence[str] = ("G1",), base_seeds: int = BASE_SEEDS, alt_seeds: int = ALT_SEEDS, factorial: bool = True, capacity: bool = True, robot_check: Optional[Callable] = None) -> Tuple[List[dict], Dict[str, str]]:
    """Rows for one program (deduplicated) and the per-robot refusal notes."""
    rows: List[dict] = []; seen = set(); notes: Dict[str, str] = {}
    def add(row):
        if row["params_hash"] in seen: return
        seen.add(row["params_hash"]); rows.append(row)
    for robot in robots:
        if robot != "G1":
            ok, note = (robot_check or robot_ok)(robot, program)
            if not ok: notes[robot] = note; continue
        n = base_seeds if robot == "G1" else alt_seeds
        for seed in range(n): add(scenario_row(program, _sample(program, seed, robot), robot, "seed"))
        if not factorial or robot != "G1": continue
        p0 = _sample(program, 0, robot).params
        for name, applies, variants in RECIPES:
            if not applies(p0): continue
            for ov in variants(p0): add(scenario_row(program, _sample(program, 0, robot, **ov), robot, name, ov))
        if capacity and program.family.value in CAPACITY_FAMILIES:
            for ov in _capacity(p0): add(scenario_row(program, _sample(program, 0, robot, **ov), robot, "capacity", ov))
    return rows, notes


def generate(programs: Optional[Iterable[str]] = None, robots: Sequence[str] = ROBOTS, base_seeds: int = BASE_SEEDS, alt_seeds: int = ALT_SEEDS, factorial: bool = True, capacity: bool = True, robot_check: Optional[Callable] = None) -> List[dict]:
    """All rows for the selected programs (default: every registered program), in registry order. Deterministic."""
    registry.load_all(); ids = list(programs) if programs else [r["id"] for r in registry.manifest()]; out: List[dict] = []
    for pid in ids:
        rows, _ = generate_program(registry.get(pid), robots, base_seeds, alt_seeds, factorial, capacity, robot_check); out += rows
    return out


def refusals(programs: Optional[Iterable[str]] = None, robots: Sequence[str] = ROBOTS) -> Dict[str, Dict[str, str]]:
    """program id -> {robot: why it does not resolve} (empty dict = resolves everywhere)."""
    registry.load_all(); ids = list(programs) if programs else [r["id"] for r in registry.manifest()]; out = {}
    for pid in ids:
        p = registry.get(pid); out[pid] = {r: robot_ok(r, p)[1] for r in robots if not robot_ok(r, p)[0]}
    return out


def to_scenario(row: dict) -> Scenario:
    """Manifest row -> core.contracts.Scenario (what runner.assemble accepts)."""
    return Scenario(id=row["id"], program_id=row["program_id"], seed=int(row["seed"]), params=dict(row["params"]), version=row.get("version", SCHEMA_VERSION), status=Status.DESIGNED)
