"""CONSTRUCT (2026-10-11): the one entry point of the movement-construction procedure recovered from g1t_3.
  input  = robot + physical effect + support contract            (no demonstration, no movement script)
  prior  = body-derived wind-up -> release reference for the effect (capacity-weighted constrained Jacobian)
  search = the reference tracked at a searched amplitude + torque knots refined on the true dynamics (CMA-ES)
  gates  = support feet loaded, direction cone, actuator envelope, energy closure, valid at 1 ms AND 0.5 ms
  ledger = what was SUPPLIED (effect, structure, budget) and what EMERGED (engine.emergence)
usage: construct.py <scenario> <seed> [iters] [pop]      scenarios in SCENARIOS; results/construct/<scenario>_s<seed>.json"""
import sys, os, json, time; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import numpy as np
from engine import core, task as TK, emergence as EM
import throw as T
R = np.radians
LEGS_FULL = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
KICK_DRIVEN = LEGS_FULL + ["waist_yaw_joint", "waist_pitch_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint"]   # a kick needs hip/ankle roll (weight transfer); 21 joints
SCENARIOS = {
    # name: effect spec. direction = (elev, az) rad in the body frame; None = the Phase 11 cone (0-45 deg elev, +-20 deg az)
    "throw":            dict(effector="right_palm", mass=T.BALL, support="both", direction=None),
    "throw_flat":       dict(effector="right_palm", mass=T.BALL, support="both", direction=(R(10), 0.0)),
    "throw_high":       dict(effector="right_palm", mass=T.BALL, support="both", direction=(R(35), 0.0)),
    "throw_left30":     dict(effector="right_palm", mass=T.BALL, support="both", direction=(R(20), R(30))),
    "throw_right30":    dict(effector="right_palm", mass=T.BALL, support="both", direction=(R(20), R(-30))),
    "strike_fwd":       dict(effector="right_palm", mass=0.0,    support="both", direction=(0.0, 0.0)),        # empty-hand forward strike: max palm speed, horizontal
    "strike_down":      dict(effector="right_palm", mass=0.0,    support="both", direction=(R(-60), 0.0)),     # downward blow
    "strike_backhand":  dict(effector="right_palm", mass=0.0,    support="both", direction=(0.0, R(90))),      # sideways across the body (+y = the robot's left)
    # COUNTERFACTUALS (Jobs F/G): the throw re-optimised under a modified body; the prior, the search and the gates all see the modification
    "throw_humanROM":   dict(effector="right_palm", mass=T.BALL, support="both", direction=None, jnt_ranges={"right_shoulder_pitch_joint": (-3.09, 1.05), "right_shoulder_yaw_joint": (-1.57, 1.57), "right_elbow_joint": (-0.17, 2.09)}),   # shoulder extension <= 60 deg, rotation <= 90 deg, elbow hyperextension <= 10 deg
    "throw_armonly":    dict(effector="right_palm", mass=T.BALL, support="both", direction=None, driven=["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_pitch_joint"]),
    "throw_weakarm":    dict(effector="right_palm", mass=T.BALL, support="both", direction=None, tau_scale={"shoulder": 0.5, "elbow": 0.5, "wrist": 0.5}),
    "throw_weaklegs":   dict(effector="right_palm", mass=T.BALL, support="both", direction=None, tau_scale={"legs": 0.5, "trunk": 0.5}),
    "throw_lefthand":   dict(effector="left_palm", mass=T.BALL, support="both", direction=None, driven=[j.replace("right_", "L_").replace("left_", "right_").replace("L_", "left_") for j in T.CFG["G1"]["driven"]]),   # mirrored arm set: left arm throws, right shoulder counters
    "kick_left30":      dict(effector="right_foot", mass=0.0, support="left", direction=(R(15), R(30)), driven=KICK_DRIVEN, omega=40.0, tscale_range=(0.9, 1.45)),
    "kick_right30":     dict(effector="right_foot", mass=0.0, support="left", direction=(R(15), R(-30)), driven=KICK_DRIVEN, omega=40.0, tscale_range=(0.9, 1.45)),
    "kick_high":        dict(effector="right_foot", mass=0.0, support="left", direction=(R(45), 0.0), driven=KICK_DRIVEN, omega=40.0, tscale_range=(0.9, 1.45)),
    # IMPACT effects: the effect is on an OBJECT (ball speed after the hit); the prior does not know where the ball is
    "strike_ball":      dict(type="impact", effector="right_palm", support="both", direction=(0.0, 0.0), object=dict(pos=[0.30, -0.25, 0.70], radius=0.05, mass=0.1, tee=True)),   # on the stance palm's forward line, hip height: where a forward sweep passes
    "kick_ball":        dict(type="impact", effector="right_foot", support="left", direction=(R(15), 0.0), object=dict(pos=[0.30, -0.17, 0.11], radius=0.11, mass=0.43, tee=False), driven=KICK_DRIVEN, omega=40.0, tscale_range=(0.9, 1.45)),
    "push_box":         dict(type="impact", effector="right_palm", support="both", direction=(0.0, 0.0), measure="displacement", hit_min=0.02, elev_tol=R(89), az_tol=R(30), object=dict(kind="box", pos=[0.50, -0.20, 0.30], size=[0.15, 0.15, 0.30], mass=6.0)),   # NEGATIVE CONTROL for the wind-up: a 6 kg box must be moved 2+ cm; a slap may or may not do it
    "push_box_sustained": dict(type="impact", effector="right_palm", support="both", direction=(0.0, 0.0), measure="displacement", hit_min=0.02, elev_tol=R(89), az_tol=R(30), f_max=300.0, t_contact_min=0.20, object=dict(kind="box", pos=[0.50, -0.20, 0.30], size=[0.15, 0.15, 0.30], mass=6.0)),   # the slap is forbidden (<= 300 N, >= 0.2 s of contact): does the same construction find a push?
    "throw_H1":         dict(robot="H1", effector="right_palm", mass=T.BALL, support="both", direction=None),      # second morphology, same procedure
    "kick_fwd":         dict(effector="right_foot", mass=0.0,    support="left", direction=(R(15), 0.0), driven=KICK_DRIVEN, omega=40.0, tscale_range=(0.9, 1.45)),      # kick: right foot, left-leg support; stiffer tracker, slower reference allowed
    "kick_fwd_left":    dict(effector="left_foot",  mass=0.0,    support="right", direction=(R(15), 0.0), driven=KICK_DRIVEN, omega=40.0, tscale_range=(0.9, 1.45)),
}
def make_task(robot, effect):
    from tasks.effect import EffectTask; from tasks.impact import ImpactTask
    robot = effect.get("robot", robot); t = (ImpactTask if effect.get("type") == "impact" else EffectTask)(robot); d = dict(t.demand_default(), **{k: v for k, v in effect.items() if k not in ("type", "robot")}); return t, d
def construct(robot, scenario, seed, iters=150, pop=16, out_dir=None):
    effect = SCENARIOS[scenario]; task, demand = make_task(robot, effect); S = task.plant(demand); ref = task.ref(S, demand)
    n = task.n_params(S, demand); z0 = np.zeros(n); z0[len(S["types"])*T.NK] = 2*(ref["takeoff_ref"] - T.T_REL_MIN)/(T.HOR - T.T_REL_MIN) - 1   # release at the reference's release time; amp mid, tscale mid-range
    t0 = time.time(); out = TK.optimize(task, demand, seed=seed, iters=iters, pop=pop, z0=z0, sigma=0.10, robust=True); secs = time.time() - t0
    z = np.asarray(out["z"]); p = task.unpack(S, demand, z); rec = task.rollout(S, demand, p, ctrl_fn=task.controller(S, demand, p), record=True)
    em = EM.analyze(S, rec, effect["effector"]) if rec.get("trace") else None
    robot = effect.get("robot", robot); led = dict(scenario=scenario, robot=robot, seed=seed, supplied=dict(effect=effect, structure="windup->release reference (t_w 0.40, t_ref 0.55, leg cap 0.30 on support legs, 95% joint range), tracker omega 20, knots 17x10 + release + amp + tscale", budget=dict(iters=iters, pop=pop, evals=iters*pop), prior_steps=dict(windup=ref["windup_steps"], path=ref["path_steps"], prep=ref.get("prep_steps"), prep_err=ref.get("prep_err"))),
               result=dict(value=float(out["value"]) if out.get("value") is not None else float(rec["value"]), valid_both_dt=bool(out["ok_all"]), reasons=rec["reasons"], meas=[dict(dt=m_.get("dt"), value=round(float(m_["value"]), 3), valid=bool(m_["valid"]), reasons=m_["reasons"]) for m_ in out.get("meas", [])], extra={k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v) for k, v in rec["extra"].items() if k in ("elev", "az", "t_rel", "work", "com_speed", "overspeed", "feet_bw", "cp_err_rel")}, amp=p.get("amp"), tscale=p.get("tscale"), secs=round(secs)),
               emerged=em, z=z.tolist())
    od = out_dir or os.environ.get("CONSTRUCT_OUT") or f"{core.RES}/construct"; os.makedirs(od, exist_ok=True); json.dump(led, open(f"{od}/{scenario}_s{seed}.json", "w"), indent=1, default=float)
    import pickle; pickle.dump(ref, open(f"{od}/{scenario}_s{seed}_ref.pkl", "wb"))      # the reference the search tracked: replays must not depend on the reference builder's later edits
    e = em or {}; print(f"[{scenario} s{seed}] {led['result']['value']:.2f} m/s valid(both dt) {led['result']['valid_both_dt']} {rec['reasons']} | peaks {e.get('peak_omega')} order {e.get('peak_order')} | own {e.get('ownership_group')} base {round(e.get('ownership_base', 0), 2)} | beyond-human {e.get('beyond_human')} | W {{{', '.join(f'{k} {round(v)}' for k, v in (e.get('W_pos') or {}).items())}}} | {secs:.0f} s", flush=True)
    return led
def replay(path, record=True):
    """Replay a ledger from its RECORDED parameters (amp, tscale, omega) so a change of the z -> parameter mapping cannot change the result."""
    led = json.load(open(path)); task, demand = make_task(led["robot"], SCENARIOS[led["scenario"]]); S = task.plant(demand); z = np.asarray(led["z"]); p = task.unpack(S, demand, z)
    rp = path.replace(".json", "_ref.pkl")
    if os.path.exists(rp):                                   # pin the recorded reference into the task's cache
        import pickle; dr = demand.get("direction"); key = (id(S["m"]), None if dr is None else tuple(np.round(dr, 4)), demand.get("capacity_mode", "power")); task._refs[key] = pickle.load(open(rp, "rb"))
    if led["result"].get("amp") is not None: p["amp"] = float(led["result"]["amp"]); p["tscale"] = float(led["result"]["tscale"])
    rec = task.rollout(S, demand, p, ctrl_fn=task.controller(S, demand, p), record=record); return task, demand, S, p, rec
if __name__ == "__main__":
    sc = sys.argv[1]; seed = int(sys.argv[2]); iters = int(sys.argv[3]) if len(sys.argv) > 3 else 150; pop = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    construct(os.environ.get("ROBOT", "G1"), sc, seed, iters, pop)
