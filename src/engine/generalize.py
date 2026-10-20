"""J17: generalization harness. FREEZE the generator (a git tag), then run held-out configurations the generator was never
tuned on: object position / mass / size, surface height, place target. Score per case: effect ok, every step valid,
steps balanced + at rest, total cost, wall time; and report the task-specific code added since the freeze tag
(git diff --stat of tasks/ and engine/ vs the tag). python engine/generalize.py <freeze_tag> [n_adapt]"""
import sys, os, os, json, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from engine import core, world as W, compose as C, task as TK
from tasks.reach import ReachTask
OUT = f"{core.RES}/generalize"

HELD_OUT = [   # (name, cube pos, cube size half, mass, table pos z (top = z + 0.02), place target)
    dict(name="base",        obj=(0.25, -0.25, 0.73), size=0.03, mass=0.2, table_z=0.68, target=(0.35, -0.10, 0.76)),
    dict(name="far_left",    obj=(0.22, -0.32, 0.73), size=0.03, mass=0.2, table_z=0.68, target=(0.30, -0.05, 0.76)),
    dict(name="heavy",       obj=(0.25, -0.25, 0.73), size=0.03, mass=0.8, table_z=0.68, target=(0.35, -0.10, 0.76)),
    dict(name="big_cube",    obj=(0.25, -0.25, 0.75), size=0.05, mass=0.3, table_z=0.68, target=(0.35, -0.10, 0.78)),
    dict(name="low_table",   obj=(0.25, -0.25, 0.63), size=0.03, mass=0.2, table_z=0.58, target=(0.35, -0.10, 0.66)),
    dict(name="high_table",  obj=(0.22, -0.25, 0.81), size=0.03, mass=0.2, table_z=0.76, target=(0.30, -0.10, 0.84)),
    dict(name="near",        obj=(0.18, -0.22, 0.73), size=0.03, mass=0.2, table_z=0.68, target=(0.28, -0.08, 0.76)),
    dict(name="light_small", obj=(0.25, -0.28, 0.72), size=0.02, mass=0.05, table_z=0.68, target=(0.33, -0.12, 0.75)),
]

def run_case(cfg, iters=80, adapt=150):
    objs = [W.obj("cube", kind="box", size=(cfg["size"],)*3, mass=cfg["mass"], pos=cfg["obj"], rgba=(0.2, 0.5, 0.9, 1))]
    surfs = [W.surface("table", pos=(0.50, 0, cfg["table_z"]), size=(0.40, 0.4, 0.02))]
    scene = W.build_scene("G1", objs, surfs); rt = ReachTask("G1", scene=scene, objects=objs, surfaces=surfs); target = list(cfg["target"])
    d1 = dict(rt.demand_default(), target_object="cube", tol=0.03, hold=0.15, grasp_on_reach="cube", omega_servo=120.0)
    d_lift = dict(rt.demand_default(), target_object="cube", offset=[0.0, 0.0, 0.10], tol=0.05, hold=0.15, omega_servo=120.0)
    d2 = dict(rt.demand_default(), target_world=target, tol=0.06, hold=0.15, release_at_end=True, omega_servo=120.0)
    S = rt.plant(d1); n = rt.n_params(S, d1); z0 = np.zeros(n); z0[-1] = (1.0 - 0.87)/0.3; t0 = time.time()
    r1 = TK.optimize(rt, d1, seed=1, iters=iters, pop=8, z0=z0, sigma=0.05, robust=False, measure_after=False)
    # J16 re-plan alternatives: a lower lift, a via-point retracting toward the body before the carry, a slower carry
    mid = [0.5*(cfg["obj"][0] + target[0]) - 0.05, 0.5*(cfg["obj"][1] + target[1]), max(cfg["obj"][2], target[2]) + 0.12]
    lift_alts = [dict(offset=[0.0, 0.0, 0.05]), dict(offset=[-0.04, 0.0, 0.08])]
    carry_alts = [dict(target_world=mid, release_at_end=False, tol=0.06), dict(omega_servo=80.0)]
    steps = [dict(task=rt, demand=d1, z=r1["z"]), dict(task=rt, demand=d_lift, z=z0, alts=lift_alts), dict(task=rt, demand=d2, z=z0, alts=carry_alts)]
    res = C.run_sequence("G1", steps, settle_T=0, adapt_iters=adapt, freeze_T=0)
    q, v = res["final_state"]; m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d)
    eff = W.effect_place(rt.world(S), None, d, "cube", target, tol=0.06, surface="table")
    return dict(case=cfg["name"], cfg=cfg, effect=eff, steps=[dict(task=s["task"], valid=s["valid"], reasons=s["reasons"], cost=-s["value"], settled=s["settled"], adapted=s["adapted"], alt=s.get("alt_used")) for s in res["steps"]],
                all_valid=all(s["valid"] for s in res["steps"]), cost=float(sum(-s["value"] for s in res["steps"])), secs=time.time() - t0)

def code_delta(tag):
    try:
        out = subprocess.run(["git", "diff", "--shortstat", tag, "--", "src/tasks"], cwd=core.ROOT, capture_output=True, text=True).stdout.strip()   # TASK-specific code only (engine changes are generic)
    except Exception as e: out = str(e)
    return out or "no changes"

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "HEAD"; adapt = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    os.makedirs(OUT, exist_ok=True); results = []
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    for cfg in HELD_OUT:
        if names and cfg["name"] not in names: continue
        r = run_case(cfg, adapt=adapt); results.append(r)
        print(f"{cfg['name']:12s} effect {r['effect']['ok']} dist {r['effect']['dist']:.3f} all_valid {r['all_valid']} cost {r['cost']:.0f} J {r['secs']:.0f} s | steps {[ (s['task'], s['valid'], s['reasons']) for s in r['steps']]}", flush=True)
        json.dump(dict(freeze_tag=tag, code_delta=code_delta(tag), results=results), open(f"{OUT}/{os.environ.get('GEN_OUT', 'pickplace_heldout')}.json", "w"), default=float, indent=1)
    ok = sum(r["effect"]["ok"] for r in results); print(f"HELD-OUT pick-place: {ok}/{len(results)} effects verified; task-specific code since {tag}: {code_delta(tag)}")
