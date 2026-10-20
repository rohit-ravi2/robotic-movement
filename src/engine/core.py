"""Frozen core wrappers: plant construction per robot, timestep, landing controller, faithful replay helpers.
Nothing here changes physics; it only names the pieces the tasks reuse. Touch only for bugs."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, branch_i as BI, balance as B

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); RES = f"{ROOT}/results"
DT_OPT = 0.001            # optimise at the faithful timestep (Phase 11 lesson)
DT_CHECK = 0.0005         # and require validity at the finer one (Phase 14 lesson)
LANDING_PAR = {"G1": f"{RES}/phase14/balance_tuning.json", "G1H": f"{RES}/phase14/balance_tuning.json", "H1": f"{RES}/phase14/balance_tuning_h1.json"}
H1_START = f"{RES}/phase9/h1_start_soleflat.npy"


def plant(robot="G1", start=None, fresh=False):
    """The settled plant S (branch_i setup dict) at DT_OPT. fresh=True builds an uncached copy that a task may modify
    (joint ranges for a range-of-motion axis, friction, masses) without touching the shared cache."""
    if robot == "H1" and start is None: start = np.load(H1_START)
    if fresh: S = BI._build(None if start is None else np.asarray(start, float), {}, robot)
    else: S = BI.setup(start=start, robot=robot) if (robot != "G1" or start is not None) else BI.setup()
    BI.apply_dt(S, DT_OPT); BI.settle(S)
    return S


def at_dt(S, dt):
    """Context: run at dt (re-settled, consistent holds), restore afterwards."""
    class _C:
        def __enter__(self_): self_.dt0 = S["m"].opt.timestep; BI.apply_dt(S, dt); BI.settle(S); return S
        def __exit__(self_, *a): BI.apply_dt(S, self_.dt0); BI.settle(S)
    return _C()


def landing_controller(S, robot="G1"):
    par = json.load(open(LANDING_PAR[robot]))["best"]
    return B.BalanceController(S, np.array(S["q_settled"]), params=par), par


def standard_jump_clip(robot="G1"):
    """The standard start as a 'clip' (knees_bent on G1, certified sole-flat on H1), as Phases 9-14 used."""
    if robot == "G1":
        import aprior_standard as AS; return AS.clip0()
    import aprior_h1 as AH; return AH.clip0()


class handoff:
    """Run a task's rollout from a HANDED-OFF state instead of the canonical settled stance, without touching the frozen
    rollouts: they read S["q_settled"] / S["v_settled"], which this context swaps and restores. Holding torques and
    held-joint targets stay those of the canonical stance, so the composer must settle near it first; the validity
    gates (START_AT_REST etc.) then judge the start honestly."""
    def __init__(self, S, q, v, hold="keep"): self.S, self.q, self.v, self.hold = S, np.asarray(q, float).copy(), np.asarray(v, float).copy(), hold
    def __enter__(self):
        import mujoco
        S = self.S; self.q0, self.v0, self.a0 = S["q_settled"], S["v_settled"], S["a_hold"]
        S["q_settled"], S["v_settled"] = self.q, self.v
        # hold="keep" (default): the task's torque baseline a_hold stays the canonical stance's. A torque-knot primitive
        # (jump, throw) is optimised RELATIVE to that baseline; recomputing it at the landed stance shifted the whole throw
        # policy and the composed throw fell from 22-23.6 m/s valid to 9-15 m/s with the stance lost (bisected to the
        # commit that introduced the recomputation, 2026-10-10). Servo-mode tasks (reach/wipe) never read a_hold.
        # hold="recompute": gravity torques of the handed posture (for a torque-mode primitive started from a bent posture).
        if self.hold != "recompute": return S
        # HOLDING TORQUES FOR THE HANDED POSTURE: the rollouts start every driven actuator at S["a_hold"] (normalised torque that
        # holds the canonical stance). A bent-over handoff needs different gravity torques; without this the robot sags and
        # falls at the start of the next primitive. a_hold = qfrc_bias(q, qdot=0) / tau, per type (mean over its actuators).
        m = S["m"]; d = mujoco.MjData(m); d.qpos[:] = self.q; d.qvel[:] = 0; mujoco.mj_forward(m, d)
        a = {}
        for t in S["types"]:
            vals = [float(d.qfrc_bias[dof]/tau) for (jid, dof, qad, tau, om, sgn, aid) in S["tmap"][t]]
            a[t] = float(np.clip(np.mean(vals), -1.0, 1.0))
        S["a_hold"] = a
        return S
    def __exit__(self, *a): self.S["q_settled"], self.S["v_settled"], self.S["a_hold"] = self.q0, self.v0, self.a0
