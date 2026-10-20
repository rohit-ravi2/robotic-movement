"""Program registry and manifest. Importing a family package registers its programs (side-effect free: no MuJoCo at import)."""
from __future__ import annotations
from typing import Dict, List, Type
from .contracts import TaskProgram, Family
_REG: Dict[str, Type[TaskProgram]] = {}
IMPORT_ERRORS: Dict[str, str] = {}     # family -> error text when its package failed to import (see load_all)

def register(cls: Type[TaskProgram]) -> Type[TaskProgram]:
    assert cls.id and cls.id not in _REG, f"program id missing or duplicate: {cls.id!r}"; _REG[cls.id] = cls; return cls

def get(pid: str) -> TaskProgram:
    if pid not in _REG: load_all()
    return _REG[pid]()

def programs(family: Family | None = None) -> List[Type[TaskProgram]]:
    return [c for c in _REG.values() if family is None or c.family == family]

def load_all():
    """Import every family package so their @register decorators run. Keep imports MuJoCo-free."""
    import importlib
    for fam in ("locomotion", "aerial", "balance", "posture", "reaching", "projection", "impact", "force_interaction", "payload", "interception", "negotiation", "compound"):
        try: importlib.import_module(f"movement_discovery.tasks.{fam}")
        except ModuleNotFoundError as e:
            if e.name != f"movement_discovery.tasks.{fam}": IMPORT_ERRORS[fam] = f"{type(e).__name__}: {e}"     # a missing dependency inside an existing family is an error, an absent family is not
        except Exception as e: IMPORT_ERRORS[fam] = f"{type(e).__name__}: {e}"                                   # one broken family must not hide the others

def manifest() -> List[dict]:
    load_all(); return [c().serialize() | dict(module=c.__module__) for c in sorted(_REG.values(), key=lambda c: (c.family.value, c.id))]
