from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .plot_speedup import plot_speedup as _plot_speedup
from .plot_throughput import plot_throughput as _plot_throughput


def write_benchmark_plots(payload: Mapping[str, Any], *, out_dir: Path) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        return {}
    results = payload.get("results")
    if not isinstance(results, list):
        return {}

    run_meta = payload.get("run") if isinstance(payload.get("run"), Mapping) else None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    created: dict[str, str] = {}

    throughput_path = out_dir / "bench-throughput.png"
    if _plot_throughput(results, throughput_path, run_meta=run_meta):
        created["throughput"] = str(throughput_path)

    speedup_path = out_dir / "bench-speedup.png"
    if _plot_speedup(results, speedup_path, run_meta=run_meta):
        created["speedup"] = str(speedup_path)

    return created
