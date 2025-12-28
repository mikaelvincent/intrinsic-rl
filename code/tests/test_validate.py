from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from irl.experiments.validation import run_validate_results


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def test_suite_validate_allows_glpe_eta_zero_without_components(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    results_dir = tmp_path / "results"
    runs_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    env_id = "DummyEnv-v0"
    run_name = "glpe__DummyEnv-v0__seed1__cfgA"
    run_dir = runs_root / run_name
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    scalars_path = run_dir / "logs" / "scalars.csv"
    scalars_header = ["step", "reward_mean", "reward_total_mean", "episode_return_mean"]
    _write_csv(
        scalars_path,
        scalars_header,
        [{"step": 0, "reward_mean": 0.0, "reward_total_mean": 0.0, "episode_return_mean": 0.0}],
    )

    cfg = {
        "method": "glpe",
        "seed": 1,
        "env": {"id": env_id},
        "intrinsic": {"eta": 0.0},
        "cfg_hash": "test",
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ckpt_latest.pt"
    torch.save(
        {
            "step": 0,
            "cfg": {"env": {"id": env_id}, "method": "glpe", "seed": 1},
        },
        ckpt_path,
    )

    _write_csv(
        results_dir / "summary_raw.csv",
        ["method", "env_id", "seed", "ckpt_path"],
        [{"method": "glpe", "env_id": env_id, "seed": 1, "ckpt_path": str(ckpt_path)}],
    )

    _write_csv(
        results_dir / "summary.csv",
        [
            "method",
            "env_id",
            "n_runs",
            "n_seeds",
            "seeds",
            "mean_return_mean",
            "mean_return_std",
            "mean_return_se",
            "mean_return_ci95_lo",
            "mean_return_ci95_hi",
            "mean_length_mean",
            "mean_length_std",
            "mean_length_se",
            "mean_length_ci95_lo",
            "mean_length_ci95_hi",
            "step_min",
            "step_max",
            "step_mean",
        ],
        [
            {
                "method": "glpe",
                "env_id": env_id,
                "n_runs": 1,
                "n_seeds": 1,
                "seeds": "1",
                "mean_return_mean": 0.0,
                "mean_return_std": 0.0,
                "mean_return_se": 0.0,
                "mean_return_ci95_lo": 0.0,
                "mean_return_ci95_hi": 0.0,
                "mean_length_mean": 1.0,
                "mean_length_std": 0.0,
                "mean_length_se": 0.0,
                "mean_length_ci95_lo": 1.0,
                "mean_length_ci95_hi": 1.0,
                "step_min": 0,
                "step_max": 0,
                "step_mean": 0,
            }
        ],
    )

    traj_dir = results_dir / "plots" / "trajectories" / run_name / "ckpt_latest"
    traj_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        traj_dir / f"{env_id}_trajectory.npz",
        obs=np.zeros((1, 4), dtype=np.float32),
        rewards_ext=np.zeros((1,), dtype=np.float32),
        gates=np.ones((1,), dtype=np.int8),
        intrinsic=np.zeros((1,), dtype=np.float32),
        env_id=np.asarray([env_id], dtype=np.str_),
        method=np.asarray(["glpe"], dtype=np.str_),
        gate_source=np.asarray(["n/a"], dtype=np.str_),
        intrinsic_semantics=np.asarray(["none"], dtype=np.str_),
    )

    assert run_validate_results(runs_root=runs_root, results_dir=results_dir, strict=True) is True
