"""Top-down PINCH (fingers down, thumb at +y) of a 10x6x4 cm sponge across its 6 cm width on a 0.64 m counter, then the wipe."""
import sys, time, json; sys.path.insert(0,os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), 'src')); import numpy as np, mujoco
from engine import world as W, core, task as TK, compose as C; from tasks.reach import ReachTask; from tasks.wipe import WipeTask
top=0.64; sp=(0.30,-0.30,top+0.015)
br=W.brush(name="sponge", pos=sp); objs=[br]; surfs=[W.surface("counter", pos=(0.55,0.0,top-0.02), size=(0.35,0.4,0.02))]
scene=W.build_scene("G1H", objs, surfs); rt=ReachTask('G1H', scene=scene, objects=objs, surfaces=surfs); wt=WipeTask('G1H', scene=scene, objects=objs, surfaces=surfs)
ori=[dict(body="palm", axis=[1,0,0], target=[0,0,-1], k=10.0, max=1.5), dict(body="palm", axis=[0,1,0], target=[0,-1,0], k=5.0, max=1.5)]
d1=dict(rt.demand_default(), target_object='sponge', offset=[0.0,0.03,br['grip_top']-0.03], tol=0.04, hold=1.0, grasp_on_reach='sponge', grasp_tol=0.07, clearance=0.08, omega_servo=120.0, trunk_cap=0.02, horizon=9.0, orient=ori, orient_min_z=top+0.09, refine=2, aim_z=True)
S=rt.plant(d1); wd=rt.world(S); m=S['m']; sid=S['site']; ref=rt.ref(S,d1)
n=rt.n_params(S,d1); z0=np.zeros(n); z0[-1]=(1.0-0.87)/0.3; log=[]
def obs(t,d,S_):
    if abs(t-int(round(t/0.5,6))*0.5)<1e-6:
        R=d.xmat[m.site_bodyid[sid]].reshape(3,3); links,F=wd.hand_contacts(d,'sponge'); log.append((t, R[:,0].round(2), (R.T@(wd.pos(d,'sponge')-d.site_xpos[sid])).round(3), len(links), round(F,1), 'closing' if 'sponge' in wd.closing else 'open'))
rec=rt.rollout(S,d1,rt.unpack(S,d1,z0),ctrl_fn=obs); e=rec['extra']
print(f'pinch pure: aim {np.round(ref.get("aim"),3)} ik err {ref.get("ik_pos_err",-1):.3f}/{ref.get("ik_ori_err",-1):.3f} | {rec["reasons"]} dmin {e["dmin"]:.3f} t_grasp {e.get("t_grasp")} held {e.get("held_after")} cp {e["cp_err_max"]:.3f}', flush=True)
for r_ in log[1::2]: print('   ', r_, flush=True)
if len(sys.argv)>1 and sys.argv[1]=='full':
    iters=int(sys.argv[2]) if len(sys.argv)>2 else 0
    if iters>0:
        t=time.time(); out=TK.optimize(rt, d1, seed=1, iters=iters, pop=8, z0=z0, sigma=0.05, robust=False); e=out['meas'][0]['extra']; z1=np.array(out['z'])
        print(f'pinch {iters}-it: ok {out["ok_all"]} {out["meas"][0]["reasons"]} dmin {e["dmin"]:.3f} t_grasp {e.get("t_grasp")} held {e.get("held_after")} ({time.time()-t:.0f} s)', flush=True)
    else: z1=z0
    json.dump(dict(demand=d1, z=z1.tolist(), scene=dict(top=top, sponge=sp)), open(f'{core.RES}/atlas/g1h_pinch_v0.json','w'))
    dw=dict(wt.demand_default(), tool='sponge', surface='counter', held=['sponge'], path=[[0.30,-0.30],[0.42,-0.30],[0.42,-0.18],[0.30,-0.18]], duration=3.2, approach=0.6, clearance=0.03, press=0.0, force_control=True, adm_k=0.003, F_ref=4.0, refine=0, orient=[dict(body="palm", axis=[1,0,0], target=[0,0,-1], k=10.0, max=1.5), dict(body="palm", axis=[0,1,0], target=[0,-1,0], k=5.0, max=1.5), dict(body="sponge", axis=[0,0,1], target=[0,0,1], k=4.0, max=1.0)], omega_legs=150.0)
    res=C.run_sequence('G1H', [dict(task=rt, demand=d1, z=z1), dict(task=wt, demand=dw, z=np.zeros(wt.n_params(wt.plant(dw), dw)))], settle_T=0, adapt_iters=int(sys.argv[3]) if len(sys.argv)>3 else 0, freeze_T=0)
    for s_ in res['steps']:
        ex=s_.get('extra',{}); print(s_['task'], s_['valid'], s_['reasons'], {k:(round(v,3) if isinstance(v,float) else v) for k,v in ex.items() if k in ('coverage','gap_mean','force_mean','force_in_band','force_max','held_after','cp_err_max','ori_err_max','t_grasp','dmin','pen_max','tilt_mean')}, flush=True)
    np.save(f'{core.RES}/atlas/g1h_pinchwipe_trace.npy', np.array([q for _,q in res['trace']]))
