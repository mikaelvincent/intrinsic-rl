import numpy as np

from irl.intrinsic.riac import simulate_lp_emas


def test_riac_lp_positive_on_decreasing_error():
    # A monotonically decreasing error stream should produce LP > 0 eventually
    errors = np.linspace(1.0, 0.0, num=200)
    el, es, lp = simulate_lp_emas(errors, beta_long=0.995, beta_short=0.90)

    assert np.isfinite(el).all() and np.isfinite(es).all() and np.isfinite(lp).all()
    # LP should be non-negative by definition and strictly positive near the end
    assert all(x >= 0.0 for x in lp)
    assert lp[-1] > 0.0
    # Short-term EMA should undershoot long-term EMA when errors trend downward
    assert es[-1] < el[-1]
