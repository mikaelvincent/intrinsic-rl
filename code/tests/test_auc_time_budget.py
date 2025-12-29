from __future__ import annotations

import numpy as np
import pytest

from irl.visualization.paper.auc_plots import _auc_from_time_curve


def test_auc_time_curve_truncates_to_budget() -> None:
    times = np.asarray([0.0, 10.0, 20.0], dtype=np.float64)
    returns = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)

    auc, tmax = _auc_from_time_curve(times, returns, budget_s=15.0)
    assert tmax == pytest.approx(15.0)
    assert auc == pytest.approx(11.25)


def test_auc_time_curve_extends_flat_to_budget() -> None:
    times = np.asarray([0.0, 10.0, 20.0], dtype=np.float64)
    returns = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)

    auc, tmax = _auc_from_time_curve(times, returns, budget_s=25.0)
    assert tmax == pytest.approx(25.0)
    assert auc == pytest.approx(30.0)


def test_auc_time_curve_infers_t0_anchor() -> None:
    times = np.asarray([5.0, 10.0], dtype=np.float64)
    returns = np.asarray([1.0, 2.0], dtype=np.float64)

    auc, tmax = _auc_from_time_curve(times, returns, budget_s=10.0)
    assert tmax == pytest.approx(10.0)
    assert auc == pytest.approx(12.5)
