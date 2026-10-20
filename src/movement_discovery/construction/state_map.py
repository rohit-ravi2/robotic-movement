"""State mapping across plants whose models differ (a bare throw plant -> a world plant with objects, a jump plant -> a catch plant):
qpos / qvel are copied JOINT BY NAME (free joints 7/6, ball 4/3, hinge/slide 1/1); joints absent from the source keep the target
plant's settled values (objects not present before start where the target's settle left them), joints absent from the target are
dropped. Also pads/strips object dofs when only the layout differs. Used by ContactSchedule at every phase handoff and offered to
compose chains (engine/compose.py is not edited: it raises on layout mismatch; call map_state first)."""
from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple
import numpy as np

_QW = {0: 7, 1: 4, 2: 1, 3: 1}; _VW = {0: 6, 1: 3, 2: 1, 3: 1}       # mjtJoint: FREE, BALL, SLIDE, HINGE


def joint_layout(m) -> Dict[str, Tuple[int, int, int, int]]:
    """name -> (qpos address, qpos width, dof address, dof width)."""
    import mujoco
    out = {}
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"; t = int(m.jnt_type[j])
        out[n] = (int(m.jnt_qposadr[j]), _QW[t], int(m.jnt_dofadr[j]), _VW[t])
    return out


def map_state(m_from, q: Sequence[float], v: Sequence[float], m_to, q_default: Optional[Sequence[float]] = None, v_default: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    """(qpos, qvel, info) for m_to from the (q, v) of m_from. Defaults = m_to's keyframe 0 (or zeros) unless given (use the target
    plant's q_settled / v_settled). info lists the joints copied, kept from the default, and dropped."""
    q = np.asarray(q, float); v = np.asarray(v, float)
    if m_from is m_to or (m_from.nq == m_to.nq and m_from.nv == m_to.nv and joint_layout(m_from) == joint_layout(m_to)): return q.copy(), v.copy(), dict(copied="all", kept=[], dropped=[])
    qo = np.array(q_default, float) if q_default is not None else (np.array(m_to.key_qpos[0]) if m_to.nkey else np.zeros(m_to.nq)); vo = np.array(v_default, float) if v_default is not None else np.zeros(m_to.nv)
    lf, lt = joint_layout(m_from), joint_layout(m_to); copied, kept = [], []
    for n, (qa, qw, da, dw) in lt.items():
        if n in lf and lf[n][1] == qw:
            fa, _, fd, _ = lf[n]; qo[qa:qa + qw] = q[fa:fa + qw]; vo[da:da + dw] = v[fd:fd + dw]; copied.append(n)
        else: kept.append(n)
    return qo, vo, dict(copied=copied, kept=kept, dropped=[n for n in lf if n not in lt])


def map_to_plant(S_from, state, S_to) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Plant-level convenience: defaults are S_to's settled state."""
    return map_state(S_from["m"], state[0], state[1], S_to["m"], S_to.get("q_settled"), S_to.get("v_settled"))
