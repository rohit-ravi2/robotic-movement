"""J27 WRAP pipeline: G1H picks an ANGLED-handle scrub brush from a 0.76 m counter with a cylindrical wrap (side approach, fingers
forward, palm z up; the handle leans along the fist's settled groove axis so it is held at equilibrium with the head flat), then wipes.
usage: g1h_wrap_wipe.py [full <iters> <adapt>]"""
import sys, os, time, json; sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), 'src')); import numpy as np, mujoco
from engine import world as W, core, task as TK, compose as C; from tasks.reach import ReachTask; from tasks.wipe import WipeTask
from engine.grasps import U_EQ, C_EQ, VIA as VIA_DEFAULT, rotz, dex3_wrap_pick, wrap_wipe
TOP = 0.76; BXY = (0.32, -0.30); CX = float(os.environ.get("CX", "0.60"))
PATH = json.loads(os.environ["PATH_XY"]) if os.environ.get("PATH_XY") else [[0.32, -0.30], [0.42, -0.30], [0.42, -0.20], [0.32, -0.20]]   # 10 x 10 cm stroke; the 12 cm version needed a 7 deg torso lean at the far side (lean transients lifted the head)
HH = float(os.environ.get("HH", "0.20"))      # handle length: 0.20 m (dish-brush); a 0.16 m handle grasped at mid-height put the lower finger 2 cm above the head top and the approach clipped the head corner
VIA = [float(x) for x in os.environ["VIA"].split(",")] if os.environ.get("VIA") else VIA_DEFAULT
TOOLK = float(os.environ.get("TOOLK", "10.0"))
def scene(top=TOP, bxy=BXY, yaw=0.0):
    br = W.brush(name="sponge", pos=(bxy[0], bxy[1], top + 0.015), lean=U_EQ, handle_h=HH, bristle=float(os.environ.get("BRISTLE", "0.02")), quat=[float(np.cos(yaw/2)), 0.0, 0.0, float(np.sin(yaw/2))])   # bristle compliance 0.02 s (a few mm of give under 5-10 N; rigid head chattered 0-24 N)
    objs = [br]; surfs = [W.surface("counter", pos=(CX, 0.0, top - 0.02), size=(0.35, 0.4, 0.02))]
    sc = W.build_scene("G1H", objs, surfs); return sc, objs, surfs, br
def pick_demand(rt, br, yaw=0.0): return dex3_wrap_pick(rt, br, yaw, via=VIA)
def wipe_demand(wt, path=PATH): return wrap_wipe(wt, 'sponge', 'counter', path, tool_k=TOOLK)
if __name__ == "__main__":
    sc, objs, surfs, br = scene(); rt = ReachTask('G1H', scene=sc, objects=objs, surfaces=surfs); wt = WipeTask('G1H', scene=sc, objects=objs, surfaces=surfs)
    d1 = pick_demand(rt, br); S = rt.plant(d1); wd = rt.world(S); m = S['m']; sid = S['site']; t0 = time.time(); ref = rt.ref(S, d1)
    print(f'ref built in {time.time()-t0:.0f} s: via {np.round(ref.get("via"), 3)} misses {np.round(ref.get("misses"), 3)} aim {np.round(ref.get("aim"), 3)} ik err {ref.get("ik_pos_err", -1):.3f}/{ref.get("ik_ori_err", -1):.3f}', flush=True)
    n = rt.n_params(S, d1); z0 = np.zeros(n); z0[-1] = (1.0 - 0.87)/0.3; log = []
    def obs(t, d, S_):
        if abs(t - int(round(t/0.5, 6))*0.5) < 1e-6:
            R = d.site_xmat[sid].reshape(3, 3); links, F = wd.hand_contacts(d, 'sponge'); log.append((t, 'palm', np.round(d.site_xpos[sid], 3), 'z-axis', R[:, 2].round(2), 'brush@palm', (R.T @ (wd.pos(d, 'sponge') - d.site_xpos[sid])).round(3), len(links), round(F, 1), 'closing' if 'sponge' in wd.closing else 'open'))
    rec = rt.rollout(S, d1, rt.unpack(S, d1, z0), ctrl_fn=obs); e = rec['extra']
    print(f'pick pure: {rec["valid"]} {rec["reasons"]} dmin {e["dmin"]:.3f} t_grasp {e.get("t_grasp")} held {e.get("held_after")} cp {e["cp_err_max"]:.3f} ori_err_max {e.get("ori_err_max", -1):.3f} palm_end {np.round(e.get("palm_end"), 3)}', flush=True)
    for r_ in log[1::2]: print('   ', r_, flush=True)
    qf, vf = e["final_state"]; np.save(f'{core.RES}/atlas/g1h_wrap_state_q.npy', qf); np.save(f'{core.RES}/atlas/g1h_wrap_state_v.npy', vf)
    if len(sys.argv) > 1 and sys.argv[1] == 'full':
        iters = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        if iters > 0:
            t = time.time(); out = TK.optimize(rt, d1, seed=1, iters=iters, pop=8, z0=z0, sigma=0.05, robust=False); e = out['meas'][0]['extra']; z1 = np.array(out['z'])
            print(f'pick {iters}-it: ok {out["ok_all"]} {out["meas"][0]["reasons"]} dmin {e["dmin"]:.3f} t_grasp {e.get("t_grasp")} held {e.get("held_after")} ({time.time()-t:.0f} s)', flush=True)
        else: z1 = z0
        json.dump(dict(demand=d1, z=z1.tolist(), scene=dict(top=TOP, sponge=BXY, lean=U_EQ)), open(f'{core.RES}/atlas/g1h_wrap_v0.json', 'w'))
        dw = wipe_demand(wt)
        res = C.run_sequence('G1H', [dict(task=rt, demand=d1, z=z1), dict(task=wt, demand=dw, z=np.zeros(wt.n_params(wt.plant(dw), dw)))], settle_T=0, adapt_iters=int(sys.argv[3]) if len(sys.argv) > 3 else 0, freeze_T=0)
        for s_ in res['steps']:
            ex = s_.get('extra', {}); print(s_['task'], s_['valid'], s_['reasons'], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ex.items() if k in ('coverage', 'gap_mean', 'force_mean', 'force_in_band', 'force_max', 'held_after', 'cp_err_max', 'ori_err_max', 't_grasp', 'dmin', 'pen_max', 'tilt_mean')}, flush=True)
        np.save(f'{core.RES}/atlas/g1h_wrapwipe_trace.npy', np.array([q for _, q in res['trace']]))
