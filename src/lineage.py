"""ResultLineage: every result carries git commit, asset hash, contract/plant hash, pose-bank
hash, optimizer config, seed, and its own hash, so a future plant bug can be traced to exactly
the results it touches."""
import hashlib, json, subprocess, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def git_commit():
    r = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip()
    return r.stdout.strip() + ("+dirty" if dirty else "")


def file_hash(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] if os.path.exists(p) else None


def stamp(result, **ctx):
    lin = dict(git=git_commit(), **ctx)
    body = json.dumps(result, sort_keys=True, default=float)
    lin["result_hash"] = hashlib.sha256(body.encode()).hexdigest()[:16]
    return dict(result, lineage=lin)


def manifest(path, **fields):
    """Experiment manifest written BEFORE running: question, robot, hashes, optimizer, seeds,
    metric, expected interpretation, acceptance criteria."""
    fields["git"] = git_commit()
    json.dump(fields, open(path, "w"), indent=1, default=str)
    return fields
