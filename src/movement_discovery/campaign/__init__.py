"""Resumable campaign driver for movement_discovery (2026-10-13). Four phases over the scenario bank:
    SCREEN          every program x nominal scenario (seed 0) x 2 optimizer seeds x first strategy_space representation, 2,400 evals
    BREADTH         every manifest scenario of every SCREEN passer, 1 seed, 2,400
    EMERGENCE       every SCREEN passer, nominal scenario, 3 seeds, 38,400
    COUNTERFACTUAL  every EMERGENCE passer x {tau_arm_0.5, tau_legs_0.5, human_rom, mirror}, 1 seed, 38,400
Each phase reads the previous phase's results from disk (phases.passers), so a phase can be regenerated at any time.
A job runs in its own subprocess (worker._exec) through runner.run(..., execute=True); the driver only claims, launches, waits.
Layout (never under results/construct*):
    results/md_campaign/<phase>/queue.jsonl      the queue (states queued | running | done | failed | skipped)
    results/md_campaign/<phase>/<job>.log|.json  per-job stdout+stderr and result (ledger entry + emergence summary + validity + pass)
    results/md_campaign/<phase>/archive/         the runner's Archive for that phase
    results/md_campaign/<phase>/REPORT.md        `report <phase>`
    results/md_campaign/LEDGER.md                `ledger` (append-only rows, construct/LEDGER.md style)
A smoke run (`run <phase> --budget-override N`) writes under results/md_campaign/_smoke/<phase>/ so the real phase starts clean.
PASS (rules.judge) = valid at both timesteps AND the program's achieved flag; regime labels are descriptive, never a filter."""
import os

RUN2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROOT = os.path.join(RUN2, "results", "md_campaign")
PHASES = ("screen", "breadth", "emergence", "counterfactual")
STATES = ("queued", "running", "done", "failed", "skipped")


def phase_dir(phase: str, smoke: bool = False) -> str:
    assert phase in PHASES, f"unknown phase {phase!r}; one of {PHASES}"
    d = os.path.join(ROOT, "_smoke", phase) if smoke else os.path.join(ROOT, phase); os.makedirs(d, exist_ok=True); return d
