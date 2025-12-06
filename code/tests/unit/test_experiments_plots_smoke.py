from __future__ import annotations

from pathlib import Path

from irl.experiments import run_plots_suite


def test_run_plots_suite_with_synthetic_scalars(tmp_path: Path) -> None:
    """Smoke-test run_plots_suite using a tiny synthetic scalars.csv.

    This avoids invoking the full trainer and instead fabricates a minimal
    run directory that matches the expected layout:

      runs_root/
        <method>__<env>__seed<seed>__anything/
          logs/
            scalars.csv

    The test verifies that a plot image is produced in results_dir/plots.
    """
    runs_root = tmp_path / "runs_suite"
    results_dir = tmp_path / "results_suite"

    # Run directory name must follow the parser convention in irl.plot._parse_run_name:
    #   <method>__<env>__seed<NUM>__<suffix>
    run_dir = runs_root / "proposed__MountainCar-v0__seed1__unit-test"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Minimal scalars.csv with the metric we will request.
    csv_path = logs_dir / "scalars.csv"
    csv_path.write_text(
        "step,reward_total_mean\n"
        "0,0.0\n"
        "10,1.0\n",
        encoding="utf-8",
    )

    # Exercise the plotting suite: it should discover the run dir,
    # aggregate the metric, and write an overlay plot per env.
    run_plots_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        metric="reward_total_mean",
        smooth=1,
        shade=True,
    )

    plots_dir = results_dir / "plots"
    # Filename convention: <env_tag>__overlay_<metric>.png
    expected_plot = plots_dir / "MountainCar-v0__overlay_reward_total_mean.png"
    assert expected_plot.exists(), "Expected overlay plot was not created"
    assert expected_plot.stat().st_size > 0, "Overlay plot file is empty"
