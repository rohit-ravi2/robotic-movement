"""The vectorised Gate A rollout must reproduce the Phase 8 implementation exactly."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, gate_a2 as GA, gate_a2_phase8_rollout as OLD
h = GA.make_h()
for seed in range(5):
    x = np.random.default_rng(seed).uniform(-1, 1, 5*GA.NK)
    a, b = GA.rollout(h, x, flight_confirm=0.0), OLD.rollout(h, x)   # legacy rule; P17 sustained rule is the default
    for k in ("v", "T", "energy_residual", "momentum_error"):   # dip: Phase 8 read h.d (defect P13), not comparable
        assert abs(a[k]-b[k]) < 1e-9, (seed, k, a[k], b[k])
print("gate A fast rollout matches Phase 8 implementation on 5 random controls")


def test_p17_momentary_unload_is_not_takeoff():
    """A solution whose first unloading re-contacts must not be scored at the unloading velocity."""
    import json
    R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "phase10", "gate_a10_main.json")))
    r = max([x for x in R if x["extra"]["spec"] == "invented_fixed"], key=lambda x: x["height"] - x.get("height_p17", x["height"]))
    h = GA.make_h(); legacy = GA.rollout(h, np.array(r["x"]), flight_confirm=0.0); fixed = GA.rollout(h, np.array(r["x"]))
    assert fixed["T"] > legacy["T"] or fixed["v"] < legacy["v"], (legacy["T"], fixed["T"])

if __name__ == "__main__":
    test_p17_momentary_unload_is_not_takeoff(); print("PASS P17 momentary unload is not a takeoff")
