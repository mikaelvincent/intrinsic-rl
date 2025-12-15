import numpy as np

from irl.intrinsic.riac import simulate_lp_emas


def test_riac_lp_positive_on_decreasing_error():
    errors = np.linspace(1.0, 0.0, num=200)
    el, es, lp = simulate_lp_emas(errors, beta_long=0.995, beta_short=0.90)

    assert np.isfinite(el).all() and np.isfinite(es).all() and np.isfinite(lp).all()
    assert all(x >= 0.0 for x in lp)
    assert lp[-1] > 0.0
    assert es[-1] < el[-1]
