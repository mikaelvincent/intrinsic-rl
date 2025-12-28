from __future__ import annotations

from pathlib import Path

from irl.benchmarks.plots import write_benchmark_plots


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
