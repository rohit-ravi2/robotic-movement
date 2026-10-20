"""Extract mechanism diagnostics from the corrected envelope-valid throw."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tasks.throw import ThrowTask  # noqa: E402
from measurement import summarize_trace  # noqa: E402


def main() -> None:
    rec = json.loads((ROOT / "results/atlas/throw_gated_v1.json").read_text())
    task = ThrowTask("G1")
    S = task.plant(rec["demand"])
    p = task.unpack(S, rec["demand"], np.asarray(rec["z"], dtype=float))
    r = task.rollout(S, rec["demand"], p, ctrl_fn=task.controller(S, rec["demand"], p), record=True)
    tr = r.get("extra", {}).get("trace")
    if tr is None:
        raise RuntimeError("throw task did not return a recorded trace")
    out = {
        "source": "results/atlas/throw_gated_v1.json",
        "speed": r["value"], "valid": r["valid"], "reasons": r["reasons"],
        "release": {k: r["extra"].get(k) for k in ("t_rel", "elev", "az", "overspeed", "feet_bw_rel", "cp_err_rel")},
        "mechanism": summarize_trace(tr, r["extra"]["t_rel"]),
    }
    path = ROOT / "results/mechanism_corrected_throw.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    try:
        import matplotlib.pyplot as plt
        seg = np.asarray(tr["seg_w"], dtype=float); power = np.asarray(tr["power"], dtype=float)
        t = np.arange(len(seg)) * float(tr["dt"])
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for i in range(seg.shape[1]): ax[0].plot(t, seg[:, i], label=f"segment {i}")
        ax[0].axvline(r["extra"]["t_rel"], color="k", ls="--", lw=1, label="release")
        ax[0].set_ylabel("segment speed")
        ax[0].legend(ncol=3, fontsize=8)
        ax[1].plot(t, np.sum(power, axis=1), color="tab:red", label="total joint power")
        ax[1].axhline(0, color="k", lw=0.5); ax[1].axvline(r["extra"]["t_rel"], color="k", ls="--", lw=1)
        ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("power (W)")
        fig.tight_layout(); fig.savefig(ROOT / "results/mechanism_corrected_throw.png", dpi=150); plt.close(fig)
    except Exception as e:
        out["plot_error"] = repr(e); path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
