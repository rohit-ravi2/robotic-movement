"""Command line (). `python -m movement_discovery.cli <command>` from run2/src:
    list [--family F]                               registered programs (MuJoCo-free)
    inspect <program_id> [--robots]                 serialize + requirements + world description + strategy space + sample scenario
    scenarios <program_id|all> [--n N] [--robot R] [--write]   scenario rows (the bank; --write regenerates scenarios/manifest.jsonl)
    check <program_id> [--seed S] [--robot R] [--representation NAME]   runner.check: plant + ONE prior-only rollout (the allowed integration check)
    run <program_id> <seed> [--representation NAME] [--optimizer cmaes] [--budget 2400] [--robot R] [--execute]   plan only unless --execute
    replay <candidate_id|scenario_id|program_id> [--archive ROOT] [--rollout]   print an archive entry; --rollout re-runs its ONE rollout
    manifest [--no-robots] [--out DIR]              regenerate MANIFEST.md / MANIFEST.json
argparse only; execution paths (run --execute, replay --rollout) are off by default and say so."""
from .main import main   # noqa
