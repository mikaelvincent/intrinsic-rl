import numpy as np

from irl.stats.mannwhitney import mannwhitney_u


def test_mannwhitney_u_alternatives_and_effect_sizes():
    x = np.arange(50, dtype=np.float64) + 100.0
    y = np.arange(50, dtype=np.float64)

    res_greater = mannwhitney_u(x, y, alternative="greater")
    assert res_greater.p_value < 1e-12
    assert res_greater.cles > 0.99
    assert abs(res_greater.cliffs_delta - (2.0 * res_greater.cles - 1.0)) < 1e-12

    res_less = mannwhitney_u(x, y, alternative="less")
    assert res_less.p_value > 1.0 - 1e-12

    res_swap = mannwhitney_u(y, x, alternative="greater")
    assert abs(res_swap.cliffs_delta + res_greater.cliffs_delta) < 1e-12
    assert abs(res_swap.cles + res_greater.cles - 1.0) < 1e-12
