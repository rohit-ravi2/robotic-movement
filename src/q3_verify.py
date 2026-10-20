"""Q3 verify (2026-10-17): replay stored G1 sprint z on FRESH plants (one new process per job: env set before import, plant built from
the Menagerie model at the job's dt; nothing cached or re-timed) at 1 ms and 0.5 ms, under the continuous (URDF effort) and the peak
(MJCF) rating sets. Also records RMS torque per rated-differing joint (thermal reading). Output results/overnight/Q3/verify_q3.json."""
import os, sys, json, subprocess
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); OUT = f"{ROOT}/results/overnight/Q3"
CHILD = r'''
import os, sys, json; sys.path.insert(0, "src")
import numpy as np
from engine import sprint as SP
pth, ix, dt = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]); R = json.load(open(pth))["rungs"][ix]
st = None
if R.get("stride_extrapolated"):   # the ladder used the EXACT stride of the rung it extrapolated from (the stored stride is rounded)
    st = dict(SP.select_stride(SP.body("G1"), float(R["stride"]["extrapolated_from"])), extrapolated_from=R["stride"]["extrapolated_from"])
P = SP.Sprint("G1", R["v"], stride=st)
r = P.run(np.asarray(R["z"]), dt, record=True); s = SP.summarize(r); S = P.S[dt]; tq = np.asarray(r["trace"]["tq"])
J = {t: dict(tau=float(S["jarr"]["tau"][i]), max=round(float(np.abs(tq[:, i]).max()), 2), rms=round(float(np.sqrt(np.mean(tq[:, i]**2))), 2))
     for i, t in enumerate(S["types"]) if any(k in t for k in ("ankle", "waist_roll", "waist_pitch", "hip_roll", "knee", "hip_pitch"))}
print("JSON" + json.dumps(dict(s, ratings=SP.RATINGS, act_tau=SP.ACT_TAU, timestep=float(S["m"].opt.timestep), joints=J), default=float))
'''
def job(pth, ix, lag, ratings, dt):
    env = dict(os.environ, SPRINT_LAG=str(lag), SPRINT_RATINGS=ratings, OMP_NUM_THREADS="1")
    o = subprocess.run(["nice", "-n", "10", sys.executable, "-c", CHILD, pth, str(ix), str(dt)], cwd=ROOT, env=env, capture_output=True, text=True)
    line = [l for l in o.stdout.splitlines() if l.startswith("JSON")]
    return {**(json.loads(line[0][4:]) if line else dict(error=o.stderr[-800:])), **dict(src=pth, rung=ix, lag=lag, ratings_job=ratings, dt=dt)}
if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    specs = json.loads(sys.argv[1])  # [[path, idx, lag], ...]
    jobs = [(p, i, l, rt, dt) for p, i, l in specs for rt in ("continuous", "peak") for dt in (0.001, 0.0005)]
    with ThreadPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 3) as ex: res = list(ex.map(lambda a: job(*a), jobs))
    out = f"{OUT}/verify_q3{sys.argv[3] if len(sys.argv) > 3 else ''}.json"; json.dump(res, open(out, "w"), indent=1)
    for r in res: print(os.path.basename(r["src"]), r["rung"], r["lag"], r["ratings_job"], r["dt"], r.get("value"), r.get("valid"), r.get("reasons"), r.get("periodic"), r.get("flight_mean"), r.get("slip_max"), r.get("overspeed"), r.get("error", "")[:200])
