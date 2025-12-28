from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from irl.experiments.plotting import run_plots_suite
from irl.visualization.paper.glpe_plots import _select_latest_glpe_trajectories


def _write_min_summary_csv(path: Path) -> None:
    path.write_text(
        (
            "method,env_id,mean_return_mean,mean_return_ci95_lo,mean_return_ci95_hi,n_seeds\n"
            "vanilla,DummyEval-v0,1.0,0.8,1.2,1\n"
        ),
        encoding="utf-8",
    )


def _write_traj(path: Path, *, env_id: str, method: str) -> None:
    obs = np.zeros((5, 2), dtype=np.float32)
    gates = np.ones((5,), dtype=np.int8)
    rewards_ext = np.zeros((5,), dtype=np.float32)
    intrinsic = np.zeros((5,), dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=obs,
        rewards_ext=rewards_ext,
        gates=gates,
        intrinsic=intrinsic,
        env_id=np.asarray([str(env_id)], dtype=np.str_),
        method=np.asarray([str(method)], dtype=np.str_),
        gate_source=np.asarray(["checkpoint"], dtype=np.str_),
        intrinsic_semantics=np.asarray(["frozen_checkpoint"], dtype=np.str_),
    )


def test_plots_manifest_created_and_nonempty(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs_root"
    results_dir = tmp_path / "results_dir"
    runs_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    _write_min_summary_csv(results_dir / "summary.csv")

    run_plots_suite(runs_root=runs_root, results_dir=results_dir)

    manifest = results_dir / "plots" / "plots_manifest.json"
    assert manifest.exists()

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert isinstance(data, dict)

    assert int(data.get("n_written", 0)) > 0
    written = data.get("written")
    assert isinstance(written, list)
    assert written

    p0 = Path(str(written[0]))
    if not p0.is_absolute():
        p0 = results_dir / p0
    assert p0.exists()


def test_select_latest_glpe_trajectories_prefers_max_step(tmp_path: Path) -> None:
    traj_root = tmp_path / "trajectories"
    env_id = "DummyEval-v0"
    env_tag = env_id.replace("/", "-")
    run_name = "glpe__DummyEval-v0__seed1__cfgA"

    p0 = traj_root / run_name / "ckpt_step_0" / f"{env_tag}_trajectory.npz"
    p100 = traj_root / run_name / "ckpt_step_100" / f"{env_tag}_trajectory.npz"

    _write_traj(p0, env_id=env_id, method="glpe")
    _write_traj(p100, env_id=env_id, method="glpe")

    selected = _select_latest_glpe_trajectories(traj_root)
    assert len(selected) == 1

    sel_env, sel_run, sel_step, sel_path = selected[0]
    assert sel_env == env_id
    assert sel_run == run_name
    assert int(sel_step) == 100
    assert Path(sel_path).resolve() == p100.resolve()
