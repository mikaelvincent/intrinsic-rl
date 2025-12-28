from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from irl.benchmarks.plots import write_benchmark_plots
from irl.visualization.paper.auc_plots import _auc_from_curve


def test_auc_from_curve_prepends_step0_anchor() -> None:
    steps = np.asarray([5, 10], dtype=np.float64)
    mean = np.asarray([2.0, 2.0], dtype=np.float64)

    auc, lo, hi, max_step = _auc_from_curve(steps, mean, None, None)

    assert lo is None and hi is None
    assert int(max_step) == 10
    assert float(auc) == pytest.approx(20.0, abs=1e-12)


def test_write_benchmark_plots_speedup_smoke(tmp_path: Path) -> None:
    payload = {
        "run": {"device": "cpu", "python": "3.x", "torch": "2.x", "torch_num_threads": 1},
        "results": [
            {
                "name": "glpe.gate_median_cache.baseline",
                "metric": "transitions_per_s",
                "unit": "transitions/s",
                "params": {"cache_interval": 1},
                "values": [100.0, 110.0, 105.0],
                "durations_s": [1.0, 1.0, 1.0],
            },
            {
                "name": "glpe.gate_median_cache.cached",
                "metric": "transitions_per_s",
                "unit": "transitions/s",
                "params": {"cache_interval": 64},
                "values": [180.0, 190.0, 185.0],
                "durations_s": [1.0, 1.0, 1.0],
            },
        ],
    }

    created = write_benchmark_plots(payload, out_dir=tmp_path)

    assert "speedup" in created
    assert Path(created["speedup"]).exists()
