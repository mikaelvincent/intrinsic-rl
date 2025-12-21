from __future__ import annotations

import csv
from pathlib import Path

from irl.visualization.timing_figures import plot_intrinsic_taper_weight


def _write_scalars(run_dir: Path, rows: list[tuple[int, float]]) -> None:
    path = Path(run_dir) / "logs" / "scalars.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "intrinsic_taper_weight"])
        for step, wt in rows:
            w.writerow([int(step), float(wt)])


def test_plot_intrinsic_taper_weight_writes_png(tmp_path: Path) -> None:
    run_a = tmp_path / "runs" / "glpe__DummyEnv-v0__seed1__cfgA"
    run_b = tmp_path / "runs" / "glpe__DummyEnv-v0__seed2__cfgA"
    _write_scalars(run_a, [(0, 1.0), (1, 0.6), (2, 0.0)])
    _write_scalars(run_b, [(0, 1.0), (1, 0.5), (2, 0.0)])

    groups = {"DummyEnv-v0": {"glpe": [run_a, run_b]}}
    plots_root = tmp_path / "plots"

    written = plot_intrinsic_taper_weight(groups, plots_root=plots_root, smooth=1, shade=False)

    out_path = plots_root / "DummyEnv-v0__glpe_intrinsic_taper.png"
    assert out_path.exists()
    assert any(Path(p).name == out_path.name for p in written)


def test_plot_intrinsic_taper_weight_skips_inactive(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "glpe__ConstEnv-v0__seed1__cfgA"
    _write_scalars(run, [(0, 1.0), (1, 1.0), (2, 1.0)])

    groups = {"ConstEnv-v0": {"glpe": [run]}}
    plots_root = tmp_path / "plots"

    written = plot_intrinsic_taper_weight(groups, plots_root=plots_root, smooth=1, shade=False)

    out_path = plots_root / "ConstEnv-v0__glpe_intrinsic_taper.png"
    assert written == []
    assert not out_path.exists()
