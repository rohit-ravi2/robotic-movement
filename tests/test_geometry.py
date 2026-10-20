"""Every lowest-point computation must agree with MuJoCo's own signed distance to the floor, for
every geom type at random orientations (the geom_rbound/capsule-radius bug family: G-a, G-b, P3,
P4, P11, and the Phase 9 cylinder formula)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, mujoco, plant as P
XML = """<mujoco><asset><mesh name='tet' vertex='0 0 0  .1 0 0  0 .08 0  0 0 .12'/></asset><worldbody>
<geom name='floor' type='plane' size='2 2 .1'/>
<body name='b' pos='0 0 1'><freejoint/>
  <geom name='g' type='{t}' size='{s}' {extra}/></body></worldbody></mujoco>"""
CASES = [("box", ".05 .08 .02", ""), ("sphere", ".04", ""), ("capsule", ".02 .06", ""),
         ("cylinder", ".03 .05", ""), ("ellipsoid", ".05 .02 .03", ""), ("mesh", "", "mesh='tet'")]
rng = np.random.default_rng(0)
for t, s, extra in CASES:
    m = mujoco.MjModel.from_xml_string(XML.format(t=t, s=s, extra=extra)); d = mujoco.MjData(m)
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "g")
    for _ in range(20):
        qv = rng.normal(size=4); qv /= np.linalg.norm(qv); d.qpos[3:7] = qv; d.qpos[2] = 1.0
        mujoco.mj_forward(m, d)
        lo = P.geom_low(m, d, g)
        dist = mujoco.mj_geomDistance(m, d, 0, g, 5.0, None)
        assert abs(lo - dist) < 2e-4, (t, lo, dist)
    print("PASS", t)
print("geometry: all lowest-point computations match MuJoCo's signed distance")
