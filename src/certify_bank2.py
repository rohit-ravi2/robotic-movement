"""Gate 1b: re-certify the certified bank with the P11-corrected support geometry, and verify
the certificate DYNAMICALLY: settle the corrected G1 plant in a stratified
sample of certified crouches and in previously-rejected crouches; certified ones must stay put."""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco, robot_bound as R, plant as P, branch_i as BI, rebuild_bank as RB
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
out = {}
for lab in ("G1", "H1"):
    rb, cert = RB.make_body(lab)
    for bankfile in ("pose_bank_certified.pkl", "pose_bank.pkl"):
        D = pickle.load(open(f"{ROOT}/results/{bankfile}", "rb"))[lab]
        rows = []
        for rec in D["bank"]:
            rb.set_pose(rec["t0"]); rb.drop_to_ground(); q = np.array(rb.d.qpos)
            c = cert(q); rows.append(dict(groups=list(rec["groups"]), status=c["status"],
                                          support=c.get("support_margin"), torque_margin=c.get("torque_margin"), q=q.tolist()))
        n = sum(r["status"] == "STATIC_FEASIBLE" for r in rows)
        print(f"{lab} {bankfile}: {n}/{len(rows)} crouches STATIC_FEASIBLE under P11-corrected geometry", flush=True)
        out[f"{lab}/{bankfile}"] = rows
# dynamic verification on G1
def settle_check(q):
    S = BI.setup(start=np.array(q)); BI.settle(S)
    d0 = mujoco.MjData(S["m"]); d0.qpos[:] = q; mujoco.mj_forward(S["m"], d0)
    d1 = mujoco.MjData(S["m"]); d1.qpos[:] = S["q_settled"]; mujoco.mj_forward(S["m"], d1)
    return dict(dz=float(d1.subtree_com[0][2] - d0.subtree_com[0][2]), dup=float(d1.xmat[1][8] - d0.xmat[1][8]),
                qdot=S["settle_qdot_max"])
rng = np.random.default_rng(0)
cert_new = [r for r in out["G1/pose_bank_certified.pkl"] if r["status"] == "STATIC_FEASIBLE"]
rej_old = [r for r in out["G1/pose_bank.pkl"] if r["status"] != "STATIC_FEASIBLE"]
sample_c = [cert_new[i] for i in rng.choice(len(cert_new), min(16, len(cert_new)), replace=False)]
sample_r = [rej_old[i] for i in rng.choice(len(rej_old), min(10, len(rej_old)), replace=False)]
dyn = []
for tag, S_ in (("certified", sample_c), ("rejected", sample_r)):
    for r in S_:
        chk = settle_check(r["q"]); held = abs(chk["dz"]) < 0.01 and abs(chk["dup"]) < 0.05
        dyn.append(dict(tag=tag, groups=r["groups"], held=held, **chk))
        print(f"  {tag:<9} {'+'.join(g[:3] for g in r['groups']):<28} held={held} dz {chk['dz']*1000:+.1f} mm", flush=True)
for tag in ("certified", "rejected"):
    g = [x for x in dyn if x["tag"] == tag]; print(f"{tag}: {sum(x['held'] for x in g)}/{len(g)} dynamically held")
out["dynamic"] = dyn
json.dump(out, open(f"{ROOT}/results/phase8/gate1b_recertify.json", "w"), indent=1, default=float)
