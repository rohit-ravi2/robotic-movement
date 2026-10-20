"""Runner plan path (no rollout), the allowed integration check, and backward compatibility of the reference construction:
the corrected throw (results/atlas/throw_gated_v1.json, 15.995 m/s, valid at 1 and 0.5 ms) replays through ThrowTask, through
EffectTask (~15.95-16.02 m/s measured 2026-10-12) and through the torque_knots+tracked_prior representation over EffectTask."""
import json, os
import numpy as np
import pytest
from .conftest import RUN2

GATED = f"{RUN2}/results/atlas/throw_gated_v1.json"


@pytest.mark.mujoco
def test_plan_only_does_not_execute(registry, mujoco):
    from movement_discovery import runner
    plan = runner.run("P41_max_release_speed", 0, execute=False); assert plan["status"].startswith("planned") and plan["execute"] is False and plan["n_params"] == 173
    assert plan["representation"]["representation"] == "torque_knots+tracked_prior" and plan["budget"] == 2400


@pytest.mark.mujoco
def test_assemble_contract(registry, mujoco):
    from movement_discovery import runner
    asm = runner.assemble("P17_impulse_recovery", 0); assert asm["world_path"] == "" or os.path.exists(asm["world_path"])
    assert asm["embodiment"].robot == "G1" and asm["rep"].name in asm["program"].strategy_space(asm["scenario"]) and "driven" in asm["demand"]


@pytest.mark.execution
@pytest.mark.slow
def test_integration_check_p41(registry, mujoco):
    from movement_discovery import runner
    r = runner.check("P41_max_release_speed", 0, verbose=False); assert r["n_params"] == 173 and r["line"].startswith("[check P41")
    assert r["outcome"].validity.reasons is not None and r["representation"]["native"] is True


@pytest.mark.execution
@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(GATED), reason="reference json missing")
def test_corrected_throw_replays_through_throwtask():
    from tasks.throw import ThrowTask
    from engine.task import measure
    rec = json.load(open(GATED)); t = ThrowTask("G1"); d = dict(t.demand_default(), **rec["demand"]); m = measure(t, d, np.asarray(rec["z"]))
    assert m["value"] == pytest.approx(rec["value"], abs=0.05) and m["ok_all"]


@pytest.mark.execution
@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(GATED), reason="reference json missing")
def test_corrected_throw_replays_through_effecttask_and_representation():
    """~15.95 m/s through EffectTask (recomputed reference differs from the stored pickle by <= 0.014 rad), z layout unchanged."""
    from tasks.effect import EffectTask
    from engine import core
    from movement_discovery import construction as C
    rec = json.load(open(GATED)); z = np.asarray(rec["z"]); t = EffectTask("G1"); d = dict(t.demand_default(), omega=rec["demand"].get("omega", 20.0))
    S = t.plant(d, core.DT_OPT); assert t.n_params(S, d) == len(z) == 173
    r = t.run(S, d, z, record=False); assert r["value"] == pytest.approx(15.95, abs=0.25) and r["valid"]
    rep = C.get("torque_knots+tracked_prior")(EffectTask("G1")); S2 = rep.plant(d, core.DT_OPT); assert rep.n_params(S2, d) == 173
    r2 = rep.run(S2, d, z); assert r2["value"] == pytest.approx(r["value"], abs=0.05)
