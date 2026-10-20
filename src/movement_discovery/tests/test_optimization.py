"""Optimizer interface: ask/tell/run over [-1,1]^n, deterministic by seed, start point always a candidate, budget tiers."""
import numpy as np
import pytest
from movement_discovery import optimization as O


def sphere(z): return float(np.sum((np.asarray(z) - 0.3)**2))


def test_names_and_tiers():
    assert {"cmaes", "random", "sobol", "cross_entropy"} <= set(O.names()); assert O.SCREEN == 2400 and O.EMERGENCE == 38400 and O.TIERS == dict(screen=2400, emergence=38400)


@pytest.mark.parametrize("name", ["random", "sobol", "cross_entropy", "cmaes"])
def test_run_interface(name):
    if name == "cmaes": pytest.importorskip("cma")
    opt = O.get(name)(seed=1, pop=6); r = opt.run(sphere, 3, z0=np.zeros(3), budget=60, sigma=0.3)
    assert set(r) >= {"z", "score", "evals", "secs", "optimizer", "config", "history"} and r["optimizer"] == name and len(r["z"]) == 3
    assert r["evals"] >= 1 and r["score"] <= sphere(np.zeros(3)) + 1e-12       # z0 is always a candidate: never worse than the start
    assert all(-1 - 1e-9 <= v <= 1 + 1e-9 for v in r["z"]) and r["history"][0]["evals"] == 1


@pytest.mark.parametrize("name", ["random", "sobol", "cross_entropy"])
def test_deterministic_by_seed(name):
    a = O.get(name)(seed=7, pop=4).run(sphere, 2, budget=20); b = O.get(name)(seed=7, pop=4).run(sphere, 2, budget=20); c = O.get(name)(seed=8, pop=4).run(sphere, 2, budget=20)
    assert a["z"] == b["z"] and a["score"] == b["score"] and (c["z"] != a["z"] or c["score"] != a["score"])


def test_ask_tell_bookkeeping():
    opt = O.get("random")(seed=0, pop=5); opt.start(2, np.zeros(2)); X = opt.ask(); assert len(X) == 5 and all(x.shape == (2,) for x in X)
    opt.tell(X, [sphere(x) for x in X]); assert opt.n_evals == 5 and opt.best[0] == min(sphere(x) for x in X) and len(opt.history) == 1


def test_cross_entropy_contracts_and_stops():
    opt = O.get("cross_entropy")(seed=0, pop=8, sigma_min=0.05); r = opt.run(sphere, 2, budget=400, sigma=0.5)
    assert r["evals"] <= 400 + 8 and r["score"] < 0.05
