"""Grip probe (J27): run the wipe from the saved grasped state under named variants and report when/if the pinch is lost.
usage: grip_probe.py <variant>[,<variant>...]   variants: base | notool (drop the tool-axis orient term) | cond4 (torsional friction on hand+handle) | slow (qd_max 0.4) | lowk (palm orient gains /2)"""
import sys, time; sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), 'src')); import numpy as np, mujoco
from engine import world as W, core; from tasks.wipe import WipeTask
top = 0.64; sp = (0.30, -0.30, top + 0.015)
PATH = [[0.30, -0.30], [0.42, -0.30], [0.42, -0.18], [0.30, -0.18]]
def demand(wt, variants):
    ori = [dict(body="palm", axis=[1, 0, 0], target=[0, 0, -1], k=10.0, max=1.5), dict(body="palm", axis=[0, 1, 0], target=[0, -1, 0], k=5.0, max=1.5)]
    if "lowk" in variants: ori = [dict(o, k=o["k"]/2) for o in ori]
    if "notool" not in variants: ori.append(dict(body="sponge", axis=[0, 0, 1], target=[0, 0, 1], k=4.0, max=1.0))
    dw = dict(wt.demand_default(), tool='sponge', surface='counter', held=['sponge'], path=PATH, duration=3.2, approach=0.6, clearance=0.03, press=0.0, force_control=True, adm_k=0.003, F_ref=4.0, refine=0, orient=ori, omega_legs=150.0)
    if "slow" in variants: dw["qd_max"] = 0.4
    if "noyaw" in variants: dw["orient_joints"] = [1.0, 1.0, 0.0]      # wrist (pitch, roll, yaw): yaw spins the handle in the pinch while the head is on the counter
    if "noori" in variants: dw["orient"] = None
    if "lift" in variants: dw["clearance"] = 0.06
    return dw
def apply_model_variants(m, wd, variants):
    if "cond4" in variants:
        gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] in wd.hand_bodies and m.geom_contype[g] > 0] + [g for g in range(m.ngeom) if m.geom_bodyid[g] == wd.bid['sponge']]
        for g in gs: m.geom_condim[g] = 4; m.geom_friction[g][1] = 0.01
        m.opt.cone = 1   # elliptic (torsional friction is only meaningful with the elliptic cone)
    if "cond6" in variants:
        gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] in wd.hand_bodies and m.geom_contype[g] > 0] + [g for g in range(m.ngeom) if m.geom_bodyid[g] == wd.bid['sponge']]
        for g in gs: m.geom_condim[g] = 6; m.geom_friction[g][1] = 0.01; m.geom_friction[g][2] = 0.002
        m.opt.cone = 1
def run(variants):
    br = W.brush(name="sponge", pos=sp); objs = [br]; surfs = [W.surface("counter", pos=(0.55, 0.0, top - 0.02), size=(0.35, 0.4, 0.02))]
    scene = W.build_scene("G1H", objs, surfs); wt = WipeTask('G1H', scene=scene, objects=objs, surfaces=surfs)
    q = np.load(f'{core.RES}/atlas/g1h_grasp_state_q.npy'); v = np.load(f'{core.RES}/atlas/g1h_grasp_state_v.npy')
    dw = demand(wt, variants); S = wt.plant(dw); m = S['m']; J = S['jarr']; wd = wt.world(S); sid = S['site']; bb = wd.bid['sponge']
    apply_model_variants(m, wd, variants)
    names = [t.replace('_joint', '').replace('right_', '') for t in S['types']]; Fs = []
    with core.handoff(S, q, v):
        def obs(t, d, S_):
            links, F = wd.hand_contacts(d, 'sponge'); Fs.append((t, len(links), F, d.site_xpos[sid][2], d.xpos[bb][2], float(np.max(np.abs(d.qvel[J['dof']])))))
        n = wt.n_params(S, dw); t0 = time.time(); rec = wt.rollout(S, dw, wt.unpack(S, dw, np.zeros(n)), ctrl_fn=obs)
    F = np.array([f[2] for f in Fs]); t = np.array([f[0] for f in Fs]); L = np.array([f[1] for f in Fs]); qv = np.array([f[5] for f in Fs])
    lost = np.where((L == 0))[0]; run0 = -1
    # grip LOST = no hand link touching the brush for 50 consecutive ms
    for i in lost:
        if i + 50 < len(L) and np.all(L[i:i+50] == 0): run0 = t[i]; break
    e = rec.get('extra', {})
    print(f"[{'+'.join(variants)}] valid {rec.get('valid')} {rec.get('reasons')} | grip lost at t={run0:.2f} | F<5N frac {np.mean(F<5):.2f} | max qvel {qv.max():.2f} | coverage {e.get('coverage')} force_mean {e.get('force_mean')} held_after {e.get('held_after')} ori_err_max {e.get('ori_err_max')} ({time.time()-t0:.0f} s)", flush=True)
    return rec
if __name__ == "__main__":
    for vs in sys.argv[1:]: run(vs.split('+'))
