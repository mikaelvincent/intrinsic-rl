import numpy as np

from irl.stats.mannwhitney import mannwhitney_u


def test_mannwhitney_u_respects_direction():
    x = np.arange(50, dtype=np.float64) + 100.0
    y = np.arange(50, dtype=np.float64)

    res = mannwhitney_u(x, y, alternative="greater")
    assert res.p_value < 1e-12
    assert res.cliffs_delta > 0.9

    res_swap = mannwhitney_u(y, x, alternative="greater")
    assert res_swap.cliffs_delta < -0.9

    res_less = mannwhitney_u(x, y, alternative="less")
    assert res_less.p_value > 1.0 - 1e-12
