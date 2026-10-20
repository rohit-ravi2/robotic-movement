"""J18: demo runner. GOAL + SCENE in -> plan over capabilities -> engine -> verified effect + clip out, from one entry point.
Goals (v0 planner = capability contracts matched by goal type):
  {"goal": "place", "object": "cube", "target": [x,y,z]}         -> reach+grasp, lift, carry+release  (effect: object pose)
  {"goal": "wipe", "tool": "sponge", "surface": "counter", "path": [[x,y],...]}  -> (tool in hand) wipe  (effect: coverage)
  {"goal": "fold", "object": "towel"}                              -> NO CAPABILITY: the planner names what is missing
Scenes are engine.world descriptions. python engine/demo.py scene.json [more.json ...]  (writes results/demo/<name>.json + .mp4 per scene)
Demo script (J18 v1): scene_kitchen_v1.json (place cube; fold towel -> missing capability) + scene_counter_v1.json (pick sponge, wipe counter)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from engine import core, world as W, compose as C, task as TK
OUT = f"{core.RES}/demo"
CAPABILITIES = {"place": ["reach", "grasp(abstract weld)", "carry", "release"], "wipe": ["reach", "tool path (geometric contact; force reported)"],
                "jump_throw": ["deep-crouch jump + landing (Phase 14 controller)", "settle to stance", "whole-body throw (Phase 11 wind-up)"],
                "fold": None, "pour": None, "open_drawer": None}


def plan(goal, scene):
    g = goal["goal"]
    if CAPABILITIES.get(g) is None:
        missing = {"fold": "deformable-object capability (cloth state, two-handed grasp, fold effect verifier)", "pour": "fluid/material effect model", "open_drawer": "articulated-object capability"}.get(g, "unknown goal")
        return dict(ok=False, missing=missing, steps=[])
    return dict(ok=True, steps=CAPABILITIES[g])


def run_place(goal, scene, adapt=150, iters=80, record_clip=True):
    from tasks.reach import ReachTask
    objs = [(W.brush(**{k: v for k, v in o.items() if k != "kind"}) if o.get("kind") == "brush" else W.obj(**o)) for o in scene["objects"]]; surfs = [W.surface(**s) for s in scene["surfaces"]]
    robot = scene.get("robot", "G1")
    path = W.build_scene(robot, objs, surfs); rt = ReachTask(robot, scene=path, objects=objs, surfaces=surfs); name = goal["object"]; target = list(goal["target"])
    surf = scene.get("place_surface", surfs[0]["name"] if surfs else None)
    # robot-specific GRASP GEOMETRY (labelled): H1's palm site sits at the forearm's end sphere (no wrist, no hand), so the
    # palm aims 4 cm above the object, approaches from 12 cm above and welds within 8 cm; G1 keeps the defaults it was built with
    gk = dict(offset=[0.0, 0.0, 0.04], clearance=0.12, grasp_tol=0.08) if robot == "H1" else {}
    d1 = dict(rt.demand_default(), target_object=name, tol=0.03, hold=0.15, grasp_on_reach=name, omega_servo=120.0, **gk)
    d_lift = dict(rt.demand_default(), target_object=name, offset=[0.0, 0.0, float(goal.get("lift", 0.10)) + gk.get("offset", [0, 0, 0])[2]], tol=0.05, hold=0.15, omega_servo=120.0)
    d2 = dict(rt.demand_default(), target_world=target, tol=0.06, hold=0.15, release_at_end=True, omega_servo=120.0)
    S = rt.plant(d1); n = rt.n_params(S, d1); z0 = np.zeros(n); z0[-1] = (1.0 - 0.87)/0.3; t0 = time.time()
    r1 = TK.optimize(rt, d1, seed=1, iters=iters, pop=8, z0=z0, sigma=0.05, robust=False, measure_after=False) if iters > 0 else dict(z=z0)   # iters 0: pure tracking of the analytic reference (the wrap pick is a precise approach; a first CMA generation of +-0.05 rad knots breaks it)
    res = C.run_sequence(robot, [dict(task=rt, demand=d1, z=r1["z"]), dict(task=rt, demand=d_lift, z=z0), dict(task=rt, demand=d2, z=z0)], settle_T=0, adapt_iters=adapt, freeze_T=0)
    q, v = res["final_state"]; m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d)
    eff = W.effect_place(rt.world(S), None, d, name, target, tol=0.06, surface=surf)
    out = dict(goal=goal, scene=scene, plan=plan(goal, scene), effect=eff, steps=[dict(task=s["task"], valid=s["valid"], reasons=s["reasons"], cost=-s["value"], alt=s.get("alt_used"), extra={k: v for k, v in s.get("extra", {}).items() if isinstance(v, (int, float, str, bool)) or v is None}) for s in res["steps"]], secs=time.time() - t0)
    if record_clip and res["trace"]:
        import progression as PG
        Q = np.array([q_ for _, q_ in res["trace"]]); T = np.linspace(0, len(Q)*0.001, len(Q)); PG.SLOW = 1.0
        tag = scene.get("name", "scene"); PG.render(m, list(zip(T, Q)), dict(z=0.9 if robot == "G1" else 1.1, dist=3.0 if robot == "G1" else 3.6, az=135, el=-12), f"Goal: place {name} ({robot})", f"{tag} · effect {'verified' if eff['ok'] else 'NOT verified'} ({eff['dist']*100:.0f} cm)", f"{OUT}/{tag}_place.mp4")
        out["clip"] = f"{OUT}/{tag}_place.mp4"
    return out


def run_wipe(goal, scene, adapt=150, iters=80, record_clip=True):
    """pick the tool (reach + abstract grasp), then wipe: the SAME quasi-static reach generator drives the tool along the path
    on the surface (squat on demand when the surface is low). Effect = path coverage measured on the simulator state."""
    from tasks.reach import ReachTask; from tasks.wipe import WipeTask
    objs = [(W.brush(**{k: v for k, v in o.items() if k != "kind"}) if o.get("kind") == "brush" else W.obj(**o)) for o in scene["objects"]]; surfs = [W.surface(**s) for s in scene["surfaces"]]
    robot = scene.get("robot", "G1")
    path = W.build_scene(robot, objs, surfs); rt = ReachTask(robot, scene=path, objects=objs, surfaces=surfs); wt = WipeTask(robot, scene=path, objects=objs, surfaces=surfs)
    tool = goal["tool"]; surf = goal["surface"]; pts = [list(map(float, p_)) for p_ in goal["path"]]
    gk = dict(offset=[0.0, 0.0, 0.04], clearance=0.12, grasp_tol=0.08) if robot == "H1" else {}   # robot-specific grasp geometry (see run_place)
    top = max(s_["pos"][2] + s_["size"][2] for s_ in surfs)
    if robot == "G1H":
        # PALM-PRESS grasp with real fingers (J20): palm down onto the sponge, fingers forward, fingers curl over the front edge
        # TOP-DOWN PINCH with real fingers (J20): fingers point down, the thumb (hand +y) at world +y of the sponge, pinch across the
        # sponge's <= 5 cm width; palm site 10 cm above the sponge centre (the fingers reach ~12 cm below the site), fingers close on arrival
        tool_obj = {o["name"]: o for o in objs}[tool]; grip = float(tool_obj.get("grip_top", 0.10)) - 0.03     # pinch space is at the palm: palm 3 cm below the handle top
        gk = dict(offset=[0.0, 0.03, grip], clearance=0.06, grasp_tol=0.07, tol=0.04, hold=1.0, horizon=9.0, orient_min_z=top + 0.09, refine=2, aim_z=True,
                  orient=[dict(body="palm", axis=[1, 0, 0], target=[0, 0, -1], k=10.0, max=1.5), dict(body="palm", axis=[0, 1, 0], target=[0, -1, 0], k=5.0, max=1.5)])   # thumb OUTWARD (world -y): the orientation with wrist margins at both hover and grasp; handle 3 cm to the thumb side of the palm, palm 3 cm below the handle top
    d1 = dict(rt.demand_default(), target_object=tool, tol=0.03, hold=0.15, grasp_on_reach=tool, omega_servo=120.0, trunk_cap=0.02, horizon=2.5); d1.update(gk)   # a tool on a low surface: reach down slowly with the waist yaw/roll frozen (approach-from-above is automatic)
    dw = dict(wt.demand_default(), tool=tool, surface=surf, held=[tool], path=pts, duration=float(goal.get("duration", 0.8*(len(pts) - 1))), press=float(goal.get("press", 0.05)), approach=0.6, clearance=0.04, refine=int(goal.get("refine", 2)))   # xy aim-off: 2 replays close the systematic waypoint offset after a handoff
    if robot == "G1H":   # real contact: rigid, no press depth, force by admittance, tool kept level through the wrist
        dw.update(press=0.0, force_control=True, adm_k=float(goal.get("adm_k", 0.003)), F_ref=float(goal.get("F_ref", 4.0)), clearance=0.03, refine=0,
                  orient=[dict(body="palm", axis=[1, 0, 0], target=[0, 0, -1], k=10.0, max=1.5), dict(body="palm", axis=[0, 1, 0], target=[0, -1, 0], k=5.0, max=1.5), dict(body=tool, axis=[0, 0, 1], target=[0, 0, 1], k=4.0, max=1.0)], omega_legs=150.0)
    S = rt.plant(d1); n = rt.n_params(S, d1); z0 = np.zeros(n); z0[-1] = (1.0 - 0.87)/0.3; t0 = time.time()
    r1 = TK.optimize(rt, d1, seed=1, iters=iters, pop=8, z0=z0, sigma=0.05, robust=False, measure_after=False) if iters > 0 else dict(z=z0)   # iters 0: pure tracking of the analytic reference (the wrap pick is a precise approach; a first CMA generation of +-0.05 rad knots breaks it)
    res = C.run_sequence(robot, [dict(task=rt, demand=d1, z=r1["z"]), dict(task=wt, demand=dw, z=z0)], settle_T=0, adapt_iters=adapt, freeze_T=0)
    sw = res["steps"][-1]; ex = sw.get("extra", {}); cov = float(ex.get("coverage", 0.0))
    eff = dict(ok=bool(sw["valid"] and cov >= 0.999), coverage=cov, gap_mean=ex.get("gap_mean"), force_mean=ex.get("force_mean"), note="coverage = tool passed within tol of every path point while on the surface (simulator state); contact force reported, not regulated")
    out = dict(goal=goal, scene=scene, plan=plan(goal, scene), effect=eff, steps=[dict(task=s["task"], valid=s["valid"], reasons=s["reasons"], cost=-s["value"], alt=s.get("alt_used"), extra={k: v for k, v in s.get("extra", {}).items() if isinstance(v, (int, float, str, bool)) or v is None}) for s in res["steps"]], secs=time.time() - t0)
    if record_clip and res["trace"]:
        import progression as PG; m = S["m"]
        Q = np.array([q_ for _, q_ in res["trace"]]); T = np.linspace(0, len(Q)*0.001, len(Q)); PG.SLOW = 1.0
        tag = scene.get("name", "scene"); PG.render(m, list(zip(T, Q)), dict(z=0.8, dist=2.8, az=120, el=-15), f"Goal: wipe {surf}", f"{tag} · coverage {100*cov:.0f}% · contact {float(ex.get('force_mean', 0) or 0):.1f} N (band open)", f"{OUT}/{tag}_wipe.mp4")
        out["clip"] = f"{OUT}/{tag}_wipe.mp4"
    return out


def run_jump_throw(goal, scene, record_clip=True):
    """DYNAMIC goal through the same runner: landed deep jump -> settle -> whole-body throw from the post-landing stance (one
    continuous physical state, no reset), warm-started from the atlas (jump: Phase 14 robust4 seed 1; throw: best atlas throw)."""
    from tasks.jump import JumpTask; from tasks.throw import ThrowTask
    R4 = {r["seed"]: r for r in json.load(open(f"{core.RES}/phase14/jump14_robust4.json"))}; Wt = json.load(open(f"{core.RES}/atlas/warm_throw_best.json"))[0]
    jt, tt = JumpTask("G1"), ThrowTask("G1"); dj = jt.demand_default(); dt_ = dict(tt.demand_default(), omega=Wt["omega"]); t0 = time.time()
    steps = [dict(task=jt, demand=dj, z=jt.warm(jt.plant(dj), dj, R4[int(goal.get("jump_seed", 1))])), dict(task=tt, demand=dt_, z=tt.warm(tt.plant(dt_), dt_, Wt))]
    res = C.run_sequence("G1", steps, settle_T=float(goal.get("settle", 2.0)), adapt_iters=int(goal.get("adapt", 20)), settle_last=bool(goal.get("recover", False)))   # no settle after the throw: the speed-only throw is not recoverable (J22); the clip ends at release and says so
    sj, st = res["steps"]; ej = sj.get("extra", {}); et = st.get("extra", {})
    eff = dict(ok=bool(sj["valid"] and st["valid"]), jump_takeoff=ej.get("v_takeoff", ej.get("takeoff_speed")), landed=ej.get("landed"), throw_speed=st.get("value"),
               note="effect = jump landed (controller) then the ball released from the same continuous state; speeds are simulator measurements")
    out = dict(goal=goal, scene=scene, plan=plan(goal, scene), effect=eff, steps=[dict(task=s_["task"], valid=s_["valid"], reasons=s_["reasons"], value=s_["value"], extra={k: v for k, v in s_.get("extra", {}).items() if isinstance(v, (int, float, str, bool)) or v is None}) for s_ in res["steps"]], secs=time.time() - t0)
    if record_clip and res["trace"]:
        import progression as PG; m = jt.plant(dj)["m"]
        Q = np.array([q_ for _, q_ in res["trace"]]); T = np.array([t_ for t_, _ in res["trace"]]); PG.SLOW = 1.0
        tag = scene.get("name", "scene"); PG.render(m, list(zip(T, Q)), dict(z=0.9, dist=3.4, az=120, el=-10), "Goal: jump, then throw", f"{tag} · jump {'landed' if sj['valid'] else 'NOT valid'} · throw {st['value']:.1f} m/s from the landing stance · ends at release: follow-through not yet controlled (J22)", f"{OUT}/{tag}_jump_throw.mp4")
        out["clip"] = f"{OUT}/{tag}_jump_throw.mp4"
    return out


def main(scene_file):
    os.makedirs(OUT, exist_ok=True); scene = json.load(open(scene_file)); results = []
    for goal in scene["goals"]:
        p = plan(goal, scene)
        if not p["ok"]:
            results.append(dict(goal=goal, plan=p, effect=dict(ok=False), note=f"missing capability: {p['missing']}")); print(goal["goal"], "-> MISSING:", p["missing"], flush=True); continue
        if goal["goal"] == "place":
            r = run_place(goal, scene, record_clip=os.environ.get("MUJOCO_GL") is not None, iters=int(os.environ.get("DEMO_ITERS", 80)), adapt=int(os.environ.get("DEMO_ADAPT", 150))); results.append(r)
            print(goal["goal"], goal["object"], "-> effect", r["effect"]["ok"], "dist %.3f" % r["effect"]["dist"], "steps", [(s["valid"], s["reasons"]) for s in r["steps"]], "%.0f s" % r["secs"], flush=True)
        elif goal["goal"] == "jump_throw":
            r = run_jump_throw(goal, scene, record_clip=os.environ.get("MUJOCO_GL") is not None); results.append(r)
            print(goal["goal"], "-> effect", r["effect"]["ok"], "throw %.1f m/s" % (r["effect"]["throw_speed"] or 0), "steps", [(s["task"], s["valid"], s["reasons"]) for s in r["steps"]], "%.0f s" % r["secs"], flush=True)
        elif goal["goal"] == "wipe":
            r = run_wipe(goal, scene, record_clip=os.environ.get("MUJOCO_GL") is not None, iters=int(os.environ.get("DEMO_ITERS", 80)), adapt=int(os.environ.get("DEMO_ADAPT", 150))); results.append(r)
            print(goal["goal"], goal["surface"], "-> effect", r["effect"]["ok"], "coverage %.2f" % r["effect"]["coverage"], "steps", [(s["valid"], s["reasons"]) for s in r["steps"]], "%.0f s" % r["secs"], flush=True)
        else:
            results.append(dict(goal=goal, plan=p, effect=dict(ok=False), note="runner not wired for this goal in v0")); print(goal["goal"], "-> planner ok, runner v0 not wired", flush=True)
    json.dump(results, open(f"{OUT}/{scene.get('name', 'scene')}.json", "w"), default=float, indent=1)

if __name__ == "__main__":
    for f in sys.argv[1:]: main(f)          # several scenes from one entry point: python engine/demo.py kitchen.json counter.json
