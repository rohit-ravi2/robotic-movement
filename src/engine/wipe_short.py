import sys, time, json; sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), 'src')); import numpy as np
from engine import world as W, task as TK, core; from tasks.wipe import WipeTask
objs=[W.obj("sponge", kind="box", size=(0.05,0.04,0.02), mass=0.1, pos=(0.053,-0.244,0.621), rgba=(0.9,0.8,0.2,1))]; surfs=[W.surface("counter", pos=(0.45,0,0.56), size=(0.35,0.4,0.02))]
scene=W.build_scene("G1", objs, surfs); wt=WipeTask("G1", scene=scene, objects=objs, surfaces=surfs)
for path in ([[0.14,-0.30],[0.22,-0.30]], [[0.14,-0.30],[0.22,-0.30],[0.22,-0.22]]):
    d=dict(wt.demand_default(), surface="counter", held=["sponge"], path=path, duration=1.6*(len(path)-1), press=0.05, approach=0.6, clearance=0.04); S=wt.plant(d); n=wt.n_params(S,d); z=np.zeros(n); z[-1]=(1.0-0.87)/0.3
    rec=wt.run(S,d,z); e=rec['extra']; print('path %d pts pure: %s cov %.2f cov_xy %.2f gap %.3f fmean %.1f N cost %.0f J'%(len(path), rec['reasons'], e['coverage'], e['coverage_xy'], e['gap_mean'], e['force_mean'], rec['cost']), flush=True)
    t=time.time(); out=TK.optimize(wt, d, seed=1, iters=100, pop=8, z0=z, sigma=0.05, robust=False); e=out['meas'][0]['extra']
    print('path %d pts 100-it: ok %s %s cov %.2f cov_xy %.2f gap %.3f fmean %.1f N fmax %.1f cost %.0f J (%.0f s)'%(len(path), out['ok_all'], out['meas'][0]['reasons'], e['coverage'], e['coverage_xy'], e['gap_mean'], e['force_mean'], e['force_max'], out['meas'][0]['cost'], time.time()-t), flush=True)
    if out['ok_all']: json.dump(dict(demand=d, z=list(map(float,out['z']))), open(f'{core.RES}/atlas/wipe_ok_{len(path)}pt.json','w'))
