from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from irl.visualization.paper.auc_plots import _auc_from_curve, plot_eval_auc_bars_by_env


def test_auc_from_curve_prepends_step0_anchor() -> None:
    steps = np.asarray([5, 10], dtype=np.float64)
    mean = np.asarray([2.0, 2.0], dtype=np.float64)

    auc, lo, hi, max_step = _auc_from_curve(steps, mean, None, None)

    assert lo is None and hi is None
    assert int(max_step) == 10
    assert float(auc) == pytest.approx(20.0, rel=0.0, abs=1e-12)


def test_plot_eval_auc_bars_runs_without_ci(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "env_id": ["DummyEnv-v0"] * 6,
            "method": ["vanilla"] * 3 + ["glpe"] * 3,
            "ckpt_step": [0, 10, 20] * 2,
            "mean_return_mean": [0.0, 1.0, 1.0, 0.0, 2.0, 2.0],
            "n_seeds": [2] * 6,
        }
    )

    written = plot_eval_auc_bars_by_env(
        df,
        plots_root=tmp_path,
        methods_to_plot=["vanilla", "glpe"],
        title="AUC test",
        filename_suffix="test",
    )

    assert len(written) == 1
    assert written[0].exists()
