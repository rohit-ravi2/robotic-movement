"""Grasp recipes shared by the pipeline, the stress test and the demo runner (J27, 2026-10-11).

DEX3 CYLINDRICAL WRAP (G1H right hand). Measured in the fixed-base grasp lab (engine/grasp_lab.py): a 3 cm handle closed on by
the Dex3 fingers settles into a diagonal groove of the fist; in the STANDARD hand pose (fingers forward = world +x, palm z up) the
handle axis settles along U_EQ and its mid point sits at C_EQ (palm frame). A brush whose handle leans along U_EQ (angled dish-brush
handle, head flat on the counter) is therefore held at equilibrium: 4-5 contacts, 50-120 N, 100% held through a 0.25/0.30 rad
shoulder/wrist shake, <= 2 cm slip. The pick slides the open hand onto the handle from VIA (behind and slightly above, in the hand's
frame) with the fingers forward; the fingers close on arrival; the wipe keeps the tool rigid in the hand and places the TOOL through
the hand (IK on the palm-frame tool point)."""
import numpy as np
U_EQ = (np.array([0.69, 0.32, 0.65])/np.linalg.norm([0.69, 0.32, 0.65])).tolist()   # settled handle axis, palm frame (standard pose)
C_EQ = [0.054, 0.056, 0.0]                                                            # mid-handle grasp point, palm frame
VIA = [-0.14, 0.0, 0.03]                                                              # side via relative to the palm target (hand frame = world under the standard pose): 14 cm behind, 3 cm up

def rotz(v, yaw):
    c, s = np.cos(yaw), np.sin(yaw); return [float(c*v[0] - s*v[1]), float(s*v[0] + c*v[1]), float(v[2])]

def dex3_wrap_pick(rt, br, yaw=0.0, via=None, **kw):
    """ReachTask demand: side-approach wrap of a leaning-handle brush dict `br` (world.brush with lean=U_EQ, yawed by `yaw`).
    The grasp frame is fixed in the HAND, so the hand yaw follows the brush yaw (palm x -> yawed +x; offset and via rotated)."""
    via = VIA if via is None else via
    ori = [dict(body="palm", axis=[0, 0, 1], target=[0, 0, 1], k=10.0, max=1.5), dict(body="palm", axis=[1, 0, 0], target=rotz([1, 0, 0], yaw), k=3.0, max=1.5)]
    d = dict(rt.demand_default(), target_object=br["name"], object_point=br["grip_point"], offset=rotz([-C_EQ[0], -C_EQ[1], -C_EQ[2]], yaw), approach="side", via_offset=rotz(via, yaw), tol=0.03, hold=1.0,
             grasp_on_reach=br["name"], grasp_tol=0.05, omega_servo=120.0, trunk_cap=0.02, horizon=9.0, orient=ori, orient_min_z=-1.0, refine=2, aim_z=True)
    d.update(kw); return d

def wrap_wipe(wt, tool, surface, path, tool_k=10.0, **kw):
    """WipeTask demand for a tool held in a Dex3 wrap: force by admittance (adm_k 0.03, F_ref 4 N), tool axis kept vertical through
    the wrist (gain tool_k), the balance controller tracks the reference torso lean (tilt_ref), 0.2 s end dwell, rigid-in-hand tool."""
    ori = [dict(body=tool, axis=[0, 0, 1], target=[0, 0, 1], k=float(tool_k), max=1.0), dict(body="palm", axis=[0, 0, 1], target=[0, 0, 1], k=2.0, max=1.5), dict(body="palm", axis=[1, 0, 0], target=[1, 0, 0], k=2.0, max=1.5)]
    d = dict(wt.demand_default(), tool=tool, surface=surface, held=[tool], path=path, duration=3.2, approach=0.6, clearance=0.03, press=0.0, force_control=True, adm_k=0.03, F_ref=4.0, refine=0, orient=ori, omega_legs=150.0, tilt_ref=True)
    d.update(kw); return d
