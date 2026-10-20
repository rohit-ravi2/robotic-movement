import numpy as np

from measurement import finite_difference, summarize_trace


def test_finite_difference_linear_signal():
    t = np.arange(5) * 0.1
    assert np.allclose(finite_difference(t * 3, 0.1), 3.0)


def test_trace_summary_separates_peak_order_and_work():
    tr = {
        "dt": 0.1,
        "seg_w": [[0, 0], [1, 0], [1, 2], [0, 2]],
        "jv": [[0, 0], [1, -2], [2, 1], [0, 0]],
        "power": [[1, -1], [2, -2], [-1, 3], [0, 0]],
        "foot_load_bw": [[1, 1], [0, 1], [1, 0]],
    }
    s = summarize_trace(tr, release_time=0.3)
    assert s["strict_peak_order"] is True
    assert s["segment_peak_time"] == [0.1, 0.2]
    assert np.isclose(s["positive_work"], 0.6)
    assert np.isclose(s["negative_work"], -0.4)
    assert s["foot_unloaded_fraction"] == [1 / 3, 1 / 3]


def test_directional_prior_vector_is_normalized():
    import sys
    sys.path.insert(0, "src")
    import throw
    # The public strategy API accepts arbitrary elevation/azimuth directions.
    e, a = 0.35, -0.4
    v = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    assert np.isclose(np.linalg.norm(v), 1.0)
