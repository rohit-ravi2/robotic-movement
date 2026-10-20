import sys, time, json; sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), 'src')); import numpy as np
from engine import world as W, task as TK, core; from tasks.wipe import WipeTask
objs=[W.obj("sponge", kind="box", size=(0.05,0.04,0.02), mass=0.1, pos=(0.053,-0.244,0.621), rgba=(0.9,0.8,0.2,1))]; surfs=[W.surface("counter", pos=(0.45,0,0.56), size=(0.35,0.4,0.02))]
scene=W.build_scene("G1", objs, surfs); wt=WipeTask("G1", scene=scene, objects=objs, surfaces=surfs)
path=[[0.12,-0.30],[0.24,-0.30],[0.24,-0.18],[0.12,-0.18]]
for press in (0.04,):
    d=dict(wt.demand_default(), surface="counter", held=["sponge"], path=path, duration=3.2, press=press, approach=0.6, clearance=0.04); S=wt.plant(d)
    n=wt.n_params(S,d); z=np.zeros(n); z[-1]=(1.0-0.87)/0.3; rec=wt.run(S,d,z); e=rec['extra']
    print('press %.2f pure: %s cov %.2f cov_xy %.2f gap %.3f in band %.2f fmean %.1f fmax %.1f N contacts %d cost %.0f J cp %.3f'%(press, rec['reasons'], e['coverage'], e['coverage_xy'], e['gap_mean'], e['force_in_band'], e['force_mean'], e['force_max'], e['n_contact'], rec['cost'], e['cp_err_max']), flush=True)
d=dict(wt.demand_default(), surface="counter", held=["sponge"], path=path, duration=3.2, press=0.04, approach=0.6, clearance=0.04); S=wt.plant(d); n=wt.n_params(S,d); z=np.zeros(n); z[-1]=(1.0-0.87)/0.3
t=time.time(); out=TK.optimize(wt, d, seed=1, iters=int(sys.argv[1]) if len(sys.argv)>1 else 60, pop=8, z0=z, sigma=0.05, robust=False); e=out['meas'][0]['extra']
print('%d-it press 0.05: ok %s %s cov %.2f cov_xy %.2f gap %.3f in band %.2f fmean %.1f N fmax %.1f cost %.0f J (%.0f s)'%(out['iters'], out['ok_all'], out['meas'][0]['reasons'], e['coverage'], e['coverage_xy'], e['gap_mean'], e['force_in_band'], e['force_mean'], e['force_max'], out['meas'][0]['cost'], time.time()-t), flush=True)
json.dump(dict(demand=d, z=list(map(float, out['z'])), value=out['value'], ok_all=out['ok_all']), open(f'{core.RES}/atlas/wipe_v0.json','w'))
