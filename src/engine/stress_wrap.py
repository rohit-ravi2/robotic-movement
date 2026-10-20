"""J27 stress (wrap): G1H side-approach wrap pick -> wipe across brush pose perturbations (dx, dy +-3 cm, yaw +-0.3 rad) and both
timesteps; every run must be fully valid (pick grasped+held, wipe covered/on-surface/no penetration/tilt, brush held after).
usage: stress_wrap.py <n> [adapt_iters]"""
import sys, json, time; sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), 'src')); import numpy as np
from engine import core, compose as C; from tasks.reach import ReachTask; from tasks.wipe import WipeTask; import engine.g1h_wrap_wipe as P
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6; adapt = int(sys.argv[2]) if len(sys.argv) > 2 else 0; rng = np.random.default_rng(0); out = []
for k in range(n):
    dx, dy, yaw = rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), rng.uniform(-0.3, 0.3); dt = core.DT_OPT if k % 2 == 0 else 0.0005
    sc, objs, surfs, br = P.scene(bxy=(P.BXY[0] + dx, P.BXY[1] + dy), yaw=yaw); rt = ReachTask('G1H', scene=sc, objects=objs, surfaces=surfs); wt = WipeTask('G1H', scene=sc, objects=objs, surfaces=surfs)
    d1 = P.pick_demand(rt, br, yaw); path = [[p_[0] + dx, p_[1] + dy] for p_ in P.PATH]; dw = P.wipe_demand(wt, path)
    S1 = rt.plant(d1, dt) if 'dt' in rt.plant.__code__.co_varnames else rt.plant(d1); n1 = rt.n_params(S1, d1); z1 = np.zeros(n1); z1[-1] = (1.0 - 0.87)/0.3
    t = time.time()
    try:
        res = C.run_sequence('G1H', [dict(task=rt, demand=d1, z=z1), dict(task=wt, demand=dw, z=np.zeros(wt.n_params(wt.plant(dw), dw)))], settle_T=0, adapt_iters=adapt, freeze_T=0, dt=dt) if 'dt' in C.run_sequence.__code__.co_varnames else C.run_sequence('G1H', [dict(task=rt, demand=d1, z=z1), dict(task=wt, demand=dw, z=np.zeros(wt.n_params(wt.plant(dw), dw)))], settle_T=0, adapt_iters=adapt, freeze_T=0)
        st = res['steps']; e0 = st[0].get('extra', {}); ex = st[-1].get('extra', {})
        rec = dict(k=k, dx=round(dx, 3), dy=round(dy, 3), yaw=round(yaw, 3), dt=dt, pick=(bool(st[0]['valid']), st[0]['reasons'], e0.get('held_after'), (round(float(e0['t_grasp']), 2) if e0.get('t_grasp') else None)), wipe=(bool(st[-1]['valid']), st[-1]['reasons']),
                   coverage=ex.get('coverage'), gap_mean=(round(float(ex['gap_mean']), 4) if ex.get('gap_mean') is not None else None), force_mean=(round(float(ex['force_mean']), 2) if ex.get('force_mean') is not None else None), in_band=(round(float(ex['force_in_band']), 2) if ex.get('force_in_band') is not None else None),
                   pen_max=(round(float(ex['pen_max']), 4) if ex.get('pen_max') is not None else None), tilt_mean=(round(float(ex['tilt_mean']), 1) if ex.get('tilt_mean') is not None else None), held_after=ex.get('held_after'), secs=round(time.time() - t))
    except Exception as ex_:
        rec = dict(k=k, dx=round(dx, 3), dy=round(dy, 3), yaw=round(yaw, 3), dt=dt, error=repr(ex_)[:200], secs=round(time.time() - t))
    out.append(rec); print(rec, flush=True); json.dump(out, open(f'{core.RES}/atlas/j27_stress_wrap.json', 'w'), default=float)
ok = sum(1 for r in out if r.get('pick', (False,))[0] and r.get('wipe', (False,))[0] and r.get('held_after')); print(f'J27 wrap stress: {ok}/{len(out)} fully valid (pick grasped + wipe valid + brush held after)')
