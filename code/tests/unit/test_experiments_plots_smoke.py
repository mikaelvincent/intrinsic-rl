from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from irl.experiments import run_plots_suite
from irl.plot import plot_normalized_summary, plot_trajectory_heatmap


def _create_run_dir(root: Path, method: str, env: str, seed: int, steps: int = 10) -> Path:
    run_dir = root / f"{method}__{env}__seed{seed}__timestamp"
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    data = {
        "step": np.arange(steps),
        "reward_mean": np.linspace(0, 10, steps),
        "reward_total_mean": np.linspace(0, 15, steps),
    }
    if method == "proposed":
        data["gate_rate"] = np.linspace(1.0, 0.0, steps)
        data["impact_rms"] = np.ones(steps) * 0.5
        data["lp_rms"] = np.ones(steps) * 0.2

    pd.DataFrame(data).to_csv(logs / "scalars.csv", index=False)
    return run_dir


def _create_summary_csv(results_dir: Path, env_ids: list[str], methods: list[str]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "env_id": env,
            "method": m,
            "mean_return_mean": np.random.rand() * 100,
            "mean_return_std": 5.0,
        }
        for env in env_ids
        for m in methods
    ]
    pd.DataFrame(rows).to_csv(results_dir / "summary.csv", index=False)


def _create_trajectory_npz(results_dir: Path, env_id: str) -> None:
    traj_dir = results_dir / "plots" / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    obs = np.random.randn(100, 2).astype(np.float32)
    gates = np.random.randint(0, 2, size=(100,))
    intrinsic = np.random.rand(100).astype(np.float32)

    np.savez(traj_dir / f"{env_id}_trajectory.npz", obs=obs, gates=gates, intrinsic=intrinsic)


def test_paper_plots_full_suite(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs_suite"
    results_dir = tmp_path / "results_suite"
    env_id = "MountainCar-v0"
    env_tag = "MountainCar-v0"

    methods = [
        "vanilla",
        "icm",
        "rnd",
        "ride",
        "proposed",
        "proposed_lp_only",
        "proposed_impact_only",
        "proposed_nogate",
    ]
    for m in methods:
        _create_run_dir(runs_root, m, env_id, seed=1)
        _create_run_dir(runs_root, m, env_id, seed=2)

    _create_summary_csv(results_dir, [env_id, "CarRacing-v3"], methods)
    _create_trajectory_npz(results_dir, env_tag)

    run_plots_suite(runs_root=runs_root, results_dir=results_dir, metric=None, smooth=1, shade=True)

    plots_dir = results_dir / "plots"
    assert plots_dir.exists()
    assert (plots_dir / f"{env_tag}__perf_extrinsic.png").exists()
    assert (plots_dir / f"{env_tag}__perf_total.png").exists()
    assert (plots_dir / f"{env_tag}__ablations.png").exists()
    assert (plots_dir / f"{env_tag}__ablations_total.png").exists()
    assert (plots_dir / f"{env_tag}__gating_dynamics.png").exists()
    assert (plots_dir / f"{env_tag}__component_evolution.png").exists()
    assert (plots_dir / "summary_normalized_bars.png").exists()
    assert (plots_dir / f"{env_tag}__state_heatmap.png").exists()


def test_plot_normalized_summary_logic(tmp_path: Path) -> None:
    csv_path = tmp_path / "summary.csv"
    out_path = tmp_path / "normalized.png"

    df = pd.DataFrame(
        [
            {"env_id": "E1", "method": "A", "mean_return_mean": 10.0, "mean_return_std": 1.0},
            {"env_id": "E1", "method": "B", "mean_return_mean": 20.0, "mean_return_std": 1.0},
            {"env_id": "E2", "method": "A", "mean_return_mean": 50.0, "mean_return_std": 1.0},
            {"env_id": "E2", "method": "B", "mean_return_mean": 50.0, "mean_return_std": 1.0},
        ]
    )
    df.to_csv(csv_path, index=False)

    plot_normalized_summary(csv_path, out_path)
    assert out_path.exists()


def test_trajectory_heatmap_logic(tmp_path: Path) -> None:
    npz_path = tmp_path / "traj.npz"
    out_path = tmp_path / "heatmap.png"

    obs = np.random.randn(50, 2).astype(np.float32)
    gates = np.ones(50, dtype=int)
    np.savez(npz_path, obs=obs, gates=gates)

    plot_trajectory_heatmap(npz_path, out_path)
    assert out_path.exists()
