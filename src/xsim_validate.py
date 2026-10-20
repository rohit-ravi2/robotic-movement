"""Validate the Drake port before any comparison: mass, start COM (forward kinematics), and a
0.5 s HOLD (knot command 0, tracker off, holding torques via the lag's initial state) in both sims."""
import sys, pickle, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "src"))
import xsim_drake as XD
P = pickle.load(open(sys.argv[1], "rb")); sp = P["spec"]
sp_hold = dict(sp, kn=np.zeros_like(sp["kn"]), tr_kp=0*sp["tr_kp"], tr_kd=0*sp["tr_kd"], horizon=0.5)
P2 = dict(P, spec=sp_hold); pickle.dump(P2, open("/tmp/hold.pkl", "wb"))
from pydrake.all import Simulator
diagram, plant = XD.build(sp_hold); sim = Simulator(diagram); ctx = sim.get_mutable_context(); pc = plant.GetMyMutableContextFromRoot(ctx)
print("Drake mass", plant.CalcTotalMass(pc), "MuJoCo mass", sp["M"])
r = XD.run("/tmp/hold.pkl")
vz = np.array(r["trace"]["vz"]); fz = np.array(r["trace"]["fz"])
print("hold: max|vz| %.4f  mean fz/BW over last 0.2 s %.3f  qdev %.4f" % (np.abs(vz).max(), fz[-200:].mean()/(r["M_drake"]*9.80665), r["q_rms_dev"]))
