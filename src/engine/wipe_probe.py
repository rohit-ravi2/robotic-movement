"""J27 wipe probe: run the wipe from the saved WRAP grasped state (g1h_wrap_state_*.npy) and print validity, coverage, force,
head-bottom gap and grip over time.  usage: wipe_probe.py [key=value ...]  (overrides on the wipe demand, e.g. F_ref=6 clearance=0.02)"""
import sys; sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), 'src')); import numpy as np, mujoco, json
from engine import core; from tasks.wipe import WipeTask; import engine.g1h_wrap_wipe as P
def run(overrides=None, verbose=True):
    sc, objs, surfs, br = P.scene(); wt = WipeTask('G1H', scene=sc, objects=objs, surfaces=surfs); dw = P.wipe_demand(wt); dw.update(overrides or {})
    S = wt.plant(dw); m = S['m']; sid = S['site']; wd = wt.world(S); bb = wd.bid['sponge']; top = wt.surface_top(dw); half = wt.tool_half(dw)
    q = np.load(f'{core.RES}/atlas/g1h_wrap_state_q.npy'); v = np.load(f'{core.RES}/atlas/g1h_wrap_state_v.npy'); log = []
    with core.handoff(S, q, v):
        ref = wt.ref(S, dw); Q = np.array(ref['Q']); fps = ref['fps']; d_ = mujoco.MjData(m)
        def obs(t, d, S_):
            if abs(t - int(round(t/0.2, 6))*0.2) < 1e-6:
                i = min(int(t*fps), len(Q) - 1); d_.qpos[:] = Q[i]; mujoco.mj_forward(m, d_); Rb = d.xmat[bb].reshape(3, 3); links, F = wd.hand_contacts(d, 'sponge')
                c = np.array([[sx, sy, -half] for sx in (-0.04, 0.04) for sy in (-0.025, 0.025)]); zmin = min((d.xpos[bb] + Rb @ ci)[2] for ci in c)
                J = S['jarr']; names = [t_.replace('_joint', '').replace('right_', '') for t_ in S['types']]; arm = [i for i, g in enumerate(S['groups']) if g in ('shoulder', 'elbow', 'wrist', 'trunk')]
                se = {names[i]: (round(float(d.ctrl[int(J['aid'][i])] - d.qpos[int(J['qad'][i])]), 3), round(float(d.ctrl[int(J['aid'][i])] - Q[i_][int(J['qad'][i])]), 3), round(float(abs(d.actuator_force[int(J['aid'][i])])/J['tau'][i]), 2)) for i_ in [i] for i in arm}
                top3 = sorted(se.items(), key=lambda kv: -abs(kv[1][0]))[:3]
                log.append((round(t, 1), d.site_xpos[sid].round(3).tolist(), (d.site_xpos[sid] - d_.site_xpos[sid]).round(3).tolist(), round(zmin - top, 4), round(float(np.degrees(np.arccos(np.clip(Rb[2, 2], -1, 1)))), 1), len(links), round(F), 'servo(err,loop_off,util)', top3))
        n = wt.n_params(S, dw); rec = wt.rollout(S, dw, wt.unpack(S, dw, np.zeros(n)), ctrl_fn=obs); e = rec['extra']
    # per-waypoint: tool xy error and head-bottom gap at the END of each pressing segment (from the 0.2 s log, nearest sample)
    ends = np.asarray(ref['bounds'])[3:3 + len(dw['path'])]; wp = []
    for k_, te in enumerate(ends):
        j = int(np.argmin([abs(r_[0] - te) for r_ in log])); wp.append((k_, round(float(te), 2), 'gap %.3f tilt %.1f' % (log[j][3], log[j][4])))
    print('   waypoint ends:', wp)
    summ = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items() if k in ('coverage', 'coverage_xy', 'gap_mean', 'force_mean', 'force_in_band', 'force_max', 'held_after', 'cp_err_max', 'ori_err_max', 'pen_max', 'tilt_mean', 'tilt_max', 'n_contact')}
    print('WIPE', rec['valid'], rec['reasons'], summ, 'misses', np.round(ref['misses'], 3).tolist(), 'horizon %.2f press %.2f-%.2f' % (ref['horizon'], ref['t_app'], ref['t_press_end']), flush=True)
    at = e.get('adm_trace'); 
    if at is not None and len(at): at = np.asarray(at); print('   adm_trace n %d: F mean %.1f dz final %.4f pen max %.4f tilt mean %.1f' % (len(at), at[:, 1].mean(), at[-1, 2], at[:, 3].max(), at[:, 4].mean()))
    if verbose:
        print('   t  palm  palm-ref  head_gap  tilt  links F')
        for r_ in log: print('  ', r_)
    return rec
if __name__ == "__main__":
    ov = {}
    for a in sys.argv[1:]:
        k, v_ = a.split('=', 1); ov[k] = json.loads(v_)
    run(ov)
