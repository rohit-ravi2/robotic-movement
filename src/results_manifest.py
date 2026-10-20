"""Phase 10 D1: manifest of result files that are NOT in git (raw campaign JSON, logs, pickles).
Decision: raw outputs stay out of git (they are regenerable from committed, seeded scripts and are
~150 MB); a copy is archived offline and the sha256 of every file is recorded here, in git."""
import os, json, hashlib, subprocess, shutil, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); ARCH = "<archive>"
PRODUCERS = {"results/phase8/": "src/phase8_campaign.py, src/baseline_campaign.py, src/gate_a2.py (Phase 8; superseded by Phase 9 faithful replays)",
             "results/phase9/baseline9": "src/baseline9.py (NS-02, NS-04, DEV-01)",
             "results/phase9/aprior_standard": "src/aprior_standard.py (NS-04 standard start)",
             "results/phase10/xsim/": "src/xsim_export.py, src/xsim_drake.py (NS-04 second simulator)",
             "results/phase10/": "src/ns04_sens.py, src/gate_a10.py (Phase 10)",
             "results/phase11/": "src/throw.py, src/throw_campaign.py, src/throw_demand.py, src/throw_demand2.py (Phase 11 throw)",
             "results/phase12/": "src/throw_counterfactual.py (Phase 12 pre-registered counterfactual)",
             "results/phase13/": "src/jump13.py (Phase 13 ladder / margin / squat-vs-CMJ)",
             "results/phase14/": "src/tune_balance.py, src/jump14.py (Phase 14 landing controller tuning; jump + landing; robust variants)",
             "results/pose_bank_certified_v1_P11flawed": "src/rebuild_bank.py (pose bank v1, invalid: P11)",
             "results/rebuild_v2.log": "src/rebuild_bank.py (pose bank v2 log)",
             "results/construct_full/": "src/engine/construct.py, src/engine/battery.py with CONSTRUCT_OUT (full tier 38,400 evals, n=3, 21 runs + *_ref.pkl)",
             "results/construct/": "src/engine/construct.py, src/engine/battery.py, src/engine/ledger.py (pilot tier 2,400 evals)",
             "results/md_campaign/": "python -m movement_discovery.campaign plan|run|report <phase> (src/movement_discovery/campaign/)",
             "results/md_build/": "movement_discovery build integration checks (runner.check, world_check.py), 2026-10-12",
             "results/atlas/": "src/engine/build_atlas.py, src/engine/g1h_*.py, src/engine/stress_wrap.py, src/throw*.py (J3-J27 atlas, grasp, throw gating)",
             "results/demo/": "src/engine/demo.py, src/engine/g1h_wrap_wipe.py, src/engine/render_wrap.py (demo scenes and clips)",
             "results/generalize/": "src/engine/generalize.py (held-out pick-place harness, freeze-gen tags)",
             "results/viz/": "src/progression.py, src/render*.py (progression page clips)"}
files = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "results"], cwd=ROOT, capture_output=True, text=True).stdout.split()
rows = []
for f in sorted(files):
    p = os.path.join(ROOT, f); h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    prod = next((v for k, v in PRODUCERS.items() if f.startswith(k)), "see phase docs")
    dst = os.path.join(ARCH, f); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(p, dst)
    assert hashlib.sha256(open(dst, "rb").read()).hexdigest() == h
    rows.append(dict(path=f, bytes=os.path.getsize(p), sha256=h, producer=prod, archived=dst))
json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M"), archive=ARCH, files=rows), open(f"{ROOT}/results/manifest.json", "w"), indent=1)
with open(f"{ROOT}/results/MANIFEST.md", "w") as o:
    o.write(f"# Untracked result files ({len(rows)}, {sum(r['bytes'] for r in rows)/1e6:.1f} MB)\n\n"
            f"Raw campaign outputs are kept out of git: regenerable from committed seeded scripts. Archived copy "
            f"(sha256-verified) at `{ARCH}`. Full list with hashes: `results/manifest.json`. Regenerate this file "
            f"with `src/results_manifest.py` after each campaign.\n\n| file | MB | sha256 (first 12) | producer |\n|---|---|---|---|\n")
    for r in rows: o.write(f"| {r['path']} | {r['bytes']/1e6:.2f} | {r['sha256'][:12]} | {r['producer']} |\n")
print(len(rows), "files,", round(sum(r['bytes'] for r in rows)/1e6, 1), "MB archived")
