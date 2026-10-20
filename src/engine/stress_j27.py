"""J27 stress test: the palm-press pick -> wipe on G1H across sponge pose perturbations and both timesteps; every run must be valid
with no PENETRATION / TOOL_TILTED / unrealistic event. Usage: j27_stress.py <pick z json> <n> """
import sys, json, time; sys.path.insert(0,os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), 'src')); import numpy as np
from engine import world as W, core, task as TK, compose as C; from tasks.reach import ReachTask; from tasks.wipe import WipeTask
sol=json.load(open(sys.argv[1])); n=int(sys.argv[2]) if len(sys.argv)>2 else 6; rng=np.random.default_rng(0); out=[]
for k in range(n):
    dx,dy,yaw=rng.uniform(-0.03,0.03), rng.uniform(-0.03,0.03), rng.uniform(-0.3,0.3); dt=core.DT_OPT if k%2==0 else 0.0005
    q=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
    br=W.brush(name="sponge", pos=(0.30+dx,-0.30+dy,0.655)); br["quat"]=q; objs=[br]; surfs=[W.surface("counter", pos=(0.55,0.0,0.62), size=(0.35,0.4,0.02))]
    scene=W.build_scene("G1H", objs, surfs); rt=ReachTask('G1H', scene=scene, objects=objs, surfaces=surfs); wt=WipeTask('G1H', scene=scene, objects=objs, surfaces=surfs)
    d1=dict(sol['demand']); d1['target_object']='sponge'; d1['grasp_on_reach']='sponge'; dw=dict(wt.demand_default(), tool='sponge', surface='counter', held=['sponge'], path=[[0.30+dx,-0.30+dy],[0.42+dx,-0.30+dy],[0.42+dx,-0.18+dy],[0.30+dx,-0.18+dy]], duration=3.2, approach=0.6, clearance=0.03, press=0.0, force_control=True, adm_k=0.003, F_ref=4.0, refine=0, orient=[dict(body="palm", axis=[1,0,0], target=[0,0,-1], k=10.0, max=1.5), dict(body="palm", axis=[0,1,0], target=[0,-1,0], k=5.0, max=1.5), dict(body="sponge", axis=[0,0,1], target=[0,0,1], k=4.0, max=1.0)], omega_legs=150.0)
    t=time.time(); res=C.run_sequence('G1H', [dict(task=rt, demand=d1, z=np.array(sol['z'])), dict(task=wt, demand=dw, z=np.zeros(wt.n_params(wt.plant(dw, dt), dw)))], settle_T=0, adapt_iters=20, freeze_T=0)
    st=res['steps']; ex=st[-1].get('extra',{}); rec=dict(k=k, dx=dx, dy=dy, yaw=yaw, dt=dt, pick=(st[0]['valid'], st[0]['reasons']), wipe=(st[-1]['valid'], st[-1]['reasons']), coverage=ex.get('coverage'), pen_max=ex.get('pen_max'), tilt_mean=ex.get('tilt_mean'), force_mean=ex.get('force_mean'), secs=time.time()-t); out.append(rec)
    print(rec, flush=True); json.dump(out, open(f'{core.RES}/atlas/j27_stress.json','w'), default=float)
print('J27:', sum(1 for r in out if r['pick'][0] and r['wipe'][0]), '/', len(out), 'fully valid')
