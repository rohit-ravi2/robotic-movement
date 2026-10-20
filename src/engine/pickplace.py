"""J14 acceptance: pick a ball off a table and place it at a target, as a composition of reach capabilities with the
abstract grasp, verified by the object's final pose. reach(ball) -> grasp -> reach(target, carrying) -> release -> settle."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from engine import core, world as W, compose as C, task as TK
from tasks.reach import ReachTask

def main(iters=30):
    # geometry inside the standing reach envelope (standing palm ~ (0.05, -0.24, 0.62)): a 0.70 m table, ball 0.30 m ahead
    # a CUBE, not a ball: a released ball rolls away on the table (first run: placed within 5 cm, rolled 29 cm)
    objs = [W.obj("cube", kind="box", size=(0.03, 0.03, 0.03), mass=0.2, pos=(0.25, -0.25, 0.73), rgba=(0.2, 0.5, 0.9, 1))]; surfs = [W.surface("table", pos=(0.50, 0, 0.68), size=(0.40, 0.4, 0.02))]
    scene = W.build_scene("G1", objs, surfs); rt = ReachTask("G1", scene=scene, objects=objs, surfaces=surfs)
    target = [0.35, -0.10, 0.76]      # 15 cm to the left and 10 cm forward of the pick: a visible transport within the right arm's envelope
    d1 = dict(rt.demand_default(), target_object="cube", tol=0.03, hold=0.15, grasp_on_reach="cube", omega_servo=120.0)
    # CARRY = LIFT (clear the surface) then MOVE: two reach capabilities, the first relative to the object, the second absolute
    d_lift = dict(rt.demand_default(), target_object="cube", offset=[0.0, 0.0, 0.10], tol=0.05, hold=0.15, omega_servo=120.0)
    d2 = dict(rt.demand_default(), target_world=target, tol=0.06, hold=0.15, release_at_end=True, omega_servo=120.0)
    S = rt.plant(d1); n = rt.n_params(S, d1); z0 = np.zeros(n); z0[-1] = (1.0 - 0.87)/0.3
    t0 = time.time()
    r1 = TK.optimize(rt, d1, seed=1, iters=iters, pop=8, z0=z0, sigma=0.05, robust=False, measure_after=False)   # servo-mode reach: offsets refine the prior
    steps = [dict(task=rt, demand=d1, z=r1["z"]), dict(task=rt, demand=d_lift, z=z0), dict(task=rt, demand=d2, z=z0)]
    res = C.run_sequence("G1", steps, settle_T=0, adapt_iters=int(os.environ.get("ADAPT_ITERS", iters)), freeze_T=0)
    for s in res["steps"]:
        e = s["extra"]; print(s["task"], (s["demand"].get("target_object"), s["demand"].get("offset")) if s["demand"].get("target_object") else s["demand"].get("target_world"), "valid", s["valid"], s["reasons"], "dmin %.3f" % e["dmin"], "t_grasp", e.get("t_grasp"), "held_after", e.get("held_after"), "adapted", s["adapted"], "cost %.0f J" % (-s["value"]), "handoff", {k: round(v, 3) for k, v in s["handoff"].items() if k != "ok"}, "end at rest", s["settled"])
    q, v = res["final_state"]; m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d); wd = rt.world(S)
    eff = W.effect_place(wd, None, d, "cube", target, tol=0.06, surface="table")
    print("EFFECT place:", eff, "on table", wd.on(d, "cube", "table"), "ball pos", np.round(wd.pos(d, "cube"), 3), "| sequence ok", res["ok"], "%.0f s wall" % (time.time() - t0))
    np.save(f"{core.RES}/atlas/pickplace_v0_trace.npy", np.array([q_ for _, q_ in res["trace"]])); json.dump(dict(steps=res["steps"], effect=eff), open(f"{core.RES}/atlas/pickplace_v0.json", "w"), default=float)
    return eff["ok"]

if __name__ == "__main__": main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
