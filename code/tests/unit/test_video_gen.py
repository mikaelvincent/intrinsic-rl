from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock

from irl.experiments import run_video_suite


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()


def _write_summary_raw(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["env_id", "method", "seed", "mean_return", "ckpt_path"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_video_suite_renders_each_shared_step(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    results_dir = tmp_path / "results"
    runs_root.mkdir()
    results_dir.mkdir()

    env_id = "Env-v0"

    run_l = runs_root / "vanilla__Env-v0__seed1__A"
    run_r = runs_root / "proposed__Env-v0__seed2__B"
    ckpt_dir_l = run_l / "checkpoints"
    ckpt_dir_r = run_r / "checkpoints"

    for step in (10, 20):
        _touch(ckpt_dir_l / f"ckpt_step_{step}.pt")
        _touch(ckpt_dir_r / f"ckpt_step_{step}.pt")
    _touch(ckpt_dir_l / "ckpt_latest.pt")
    _touch(ckpt_dir_r / "ckpt_latest.pt")

    _write_summary_raw(
        results_dir / "summary_raw.csv",
        [
            {
                "env_id": env_id,
                "method": "vanilla",
                "seed": 1,
                "mean_return": 10.0,
                "ckpt_path": str(ckpt_dir_l / "ckpt_step_20.pt"),
            },
            {
                "env_id": env_id,
                "method": "proposed",
                "seed": 2,
                "mean_return": 100.0,
                "ckpt_path": str(ckpt_dir_r / "ckpt_step_20.pt"),
            },
        ],
    )

    mock_render = MagicMock()
    monkeypatch.setattr("irl.experiments.videos.render_side_by_side", mock_render)

    run_video_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        device="cpu",
        baseline="vanilla",
        method="proposed",
    )

    assert mock_render.call_count == 2

    calls: dict[str, dict] = {}
    for call in mock_render.call_args_list:
        _, kwargs = call
        calls[Path(kwargs["out_path"]).name] = kwargs

    assert any("step000000010" in name for name in calls)
    assert any("step000000020" in name for name in calls)

    for name, kwargs in calls.items():
        assert kwargs["env_id"] == env_id
        ckpt_left = Path(kwargs["ckpt_left"])
        ckpt_right = Path(kwargs["ckpt_right"])
        if "step000000010" in name:
            assert ckpt_left.name == "ckpt_step_10.pt"
            assert ckpt_right.name == "ckpt_step_10.pt"
        elif "step000000020" in name:
            assert ckpt_left.name == "ckpt_step_20.pt"
            assert ckpt_right.name == "ckpt_step_20.pt"
