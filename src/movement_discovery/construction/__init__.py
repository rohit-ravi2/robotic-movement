"""Construction representations (registry). A representation = a parameter space + controller glue over a wrapped reference task,
exposing the engine.task.Task interface (see base.Representation) so engine.task.optimize / measure run unchanged.
    get(name)(task, horizon=None) -> Representation      names: torque_knots+tracked_prior | torque_knots | servo_knots | contact_schedule+knots | torque_knots+brace_prior | servo_knots+brace_prior | torque_knots+absorb_prior
    priors: windup / ascent / reach / brace / absorb (options; none prescribed)
Programs declare compatible names in TaskProgram.strategy_space; the runner instantiates by name. MuJoCo-free at import."""
from __future__ import annotations
from typing import Dict, Type
from .base import Representation, bind_horizon
from .torque_knots import TorqueKnotsTrackedPrior, TorqueKnots
from .servo_knots import ServoKnots
from .contact_schedule import ContactSchedule
from .brace import TorqueKnotsBracePrior, ServoKnotsBracePrior
from . import priors   # noqa

REPRESENTATIONS: Dict[str, Type[Representation]] = {c.name: c for c in (TorqueKnotsTrackedPrior, TorqueKnots, ServoKnots, ContactSchedule, TorqueKnotsBracePrior, ServoKnotsBracePrior)}


def register(cls: Type[Representation]) -> Type[Representation]:
    assert cls.name not in REPRESENTATIONS, cls.name; REPRESENTATIONS[cls.name] = cls; return cls


from .absorb import TorqueKnotsAbsorbPrior   # noqa  (ABSORB: receiving prior for the interception regime)
register(TorqueKnotsAbsorbPrior)

from .rise import TorqueKnotsRisePrior   # noqa  (RISE: certified support-transition prior for the non-standing-start regime)
register(TorqueKnotsRisePrior)


def get(name: str) -> Type[Representation]:
    if name not in REPRESENTATIONS: raise KeyError(f"unknown representation {name!r}; known: {sorted(REPRESENTATIONS)}")
    return REPRESENTATIONS[name]


def describe_all() -> dict: return {n: dict(label=c.label, doc=(c.__doc__ or "").strip().split("\n")[0]) for n, c in REPRESENTATIONS.items()}
from .step import StepPrior        # noqa: E402  (: step-as-an-effect prior for the stepping / base regime)
register(StepPrior)
