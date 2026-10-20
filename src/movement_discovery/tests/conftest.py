"""Markers: `mujoco` (loads robot models / compiles worlds, no rollout), `execution` (rolls out: skipped unless --run-execution),
`slow`. The src directory (throw.py, engine/, tasks/) is put on sys.path so the reference implementation imports resolve."""
from __future__ import annotations
import os, sys
import pytest

SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC not in sys.path: sys.path.insert(0, SRC)
RUN2 = os.path.dirname(SRC)


def pytest_addoption(parser):
    parser.addoption("--run-execution", action="store_true", default=False, help="run tests that roll out the simulator (marker execution)")


def pytest_configure(config):
    config.addinivalue_line("markers", "execution: rolls the simulator out (ONE rollout per test, never a search); skipped unless --run-execution")
    config.addinivalue_line("markers", "mujoco: loads MuJoCo models or compiles worlds (no rollout)")
    config.addinivalue_line("markers", "slow: takes more than ~30 s")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-execution"): return
    skip = pytest.mark.skip(reason="execution test: pass --run-execution")
    for it in items:
        if "execution" in it.keywords: it.add_marker(skip)


@pytest.fixture(scope="session")
def registry():
    from movement_discovery.core import registry as R
    R.load_all(); assert not R.IMPORT_ERRORS, R.IMPORT_ERRORS; return R


@pytest.fixture(scope="session")
def mujoco():
    return pytest.importorskip("mujoco")


@pytest.fixture
def tmp_archive(tmp_path):
    from movement_discovery.archive import Archive
    return Archive(str(tmp_path / "archive"))
