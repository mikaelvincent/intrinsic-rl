from __future__ import annotations

from pathlib import Path

from irl.visualization.suite_figures import _generate_comparison_plot


def _write_scalars(run_dir: Path, *, vals: list[tuple[int, float]]) -> None:
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    p = logs / "scalars.csv"
    lines = ["step,episode_return_mean,reward_total_mean,reward_mean\n"]
    for step, v in vals:
        lines.append(f"{int(step)},{float(v)},{float(v)},{float(v)}\n")
    p.write_text("".join(lines), encoding="utf-8")


def test_suite_comparison_plot_writes_file(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    plots_root = tmp_path / "results" / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    r_v = runs_root / "vanilla__DummyEval-v0__seed1__cfgA"
    r_g = runs_root / "glpe__DummyEval-v0__seed1__cfgA"

    _write_scalars(r_v, vals=[(0, 0.0), (100, 1.0)])
    _write_scalars(r_g, vals=[(0, 0.0), (100, 2.0)])

    groups = {"DummyEval-v0": {"vanilla": [r_v], "glpe": [r_g]}}

    _generate_comparison_plot(
        groups_by_env=groups,
        methods_to_plot=["vanilla", "glpe"],
        metric="episode_return_mean",
        smooth=1,
        shade=False,
        title="Task Performance (Episode Return)",
        filename_suffix="perf_extrinsic",
        plots_root=plots_root,
    )

    out = plots_root / "DummyEval-v0__perf_extrinsic.png"
    assert out.exists()
    assert not out.with_suffix(out.suffix + ".tmp").exists()
