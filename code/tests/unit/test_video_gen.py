from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
import pandas as pd
import torch

from irl.experiments import run_video_suite
import irl.video


def test_render_video_frame_composition(tmp_path, monkeypatch):
    """Test that render_side_by_side correctly stitches frames from two envs."""

    # Mock dependencies to avoid real env/policy overhead
    mock_pol = MagicMock()
    mock_pol.act.return_value = (torch.tensor([0]), torch.tensor([0.0]))  # dummy action

    # Mock _load_policy_for_eval to return our mock policy and a dummy config
    def mock_load(path, device):
        return mock_pol, {"env": {"frame_skip": 1, "discrete_actions": True}}

    monkeypatch.setattr(irl.video, "_load_policy_for_eval", mock_load)

    # Mock EnvManager to return a dummy env that produces predictable frames
    class MockEnv:
        def __init__(self, *args, **kwargs):
            self.action_space = MagicMock()
            self.action_space.shape = ()  # discrete

        def reset(self, **kwargs):
            return np.zeros(2), {}

        def step(self, action):
            return np.zeros(2), 1.0, False, False, {}

        def render(self):
            # Return a 10x10 white square
            return np.full((10, 10, 3), 255, dtype=np.uint8)

        def close(self):
            pass

    mock_manager = MagicMock()
    mock_manager.make.return_value = MockEnv()

    # Patch the EnvManager constructor in irl.video to return our mock manager
    monkeypatch.setattr(irl.video, "EnvManager", MagicMock(return_value=mock_manager))

    # Mock imageio to intercept save
    mock_mimsave = MagicMock()
    monkeypatch.setattr(irl.video.imageio, "mimsave", mock_mimsave)

    out_path = tmp_path / "test_video.mp4"

    # Run
    irl.video.render_side_by_side(
        env_id="Dummy-v0",
        ckpt_left=Path("dummy_left.pt"),
        ckpt_right=Path("dummy_right.pt"),
        out_path=out_path,
        max_steps=5,
        device="cpu",
    )

    # Verify
    assert mock_mimsave.called
    args, _ = mock_mimsave.call_args
    path_arg, frames_arg = args
    assert str(path_arg) == str(out_path)
    assert len(frames_arg) > 0
    # Check dimensions: 2 images of 10 width side-by-side -> width 20 approx
    # (PIL text overlay might resize, but basic stitching structure should hold)
    frame = frames_arg[0]
    assert frame.shape[2] == 3  # RGB
    assert frame.shape[1] >= 20  # Width at least 10+10


def test_video_suite_renders_every_checkpoint(tmp_path, monkeypatch):
    """run_video_suite should render one video per ckpt_step_*.pt."""
    runs_root = tmp_path / "runs"
    results_dir = tmp_path / "results"

    # Setup directories
    runs_root.mkdir()
    results_dir.mkdir()

    # Create fake run dirs
    run_l = runs_root / "vanilla__Env-v0__seed1__A"
    run_r = runs_root / "proposed__Env-v0__seed2__B"
    run_l.mkdir(parents=True)
    run_r.mkdir(parents=True)

    # Create checkpoints for both sides (two steps)
    ckpt_dir_l = run_l / "checkpoints"
    ckpt_dir_r = run_r / "checkpoints"
    ckpt_dir_l.mkdir()
    ckpt_dir_r.mkdir()

    # Step checkpoints (the suite should iterate these)
    (ckpt_dir_l / "ckpt_step_10.pt").touch()
    (ckpt_dir_l / "ckpt_step_20.pt").touch()
    (ckpt_dir_r / "ckpt_step_10.pt").touch()
    (ckpt_dir_r / "ckpt_step_20.pt").touch()

    # Latest alias (not required by the new logic, but common in real runs)
    (ckpt_dir_l / "ckpt_latest.pt").touch()
    (ckpt_dir_r / "ckpt_latest.pt").touch()

    # Create summary_raw.csv so selection prefers these runs.
    df = pd.DataFrame(
        [
            {
                "env_id": "Env-v0",
                "method": "vanilla",
                "seed": 1,
                "mean_return": 10.0,
                "ckpt_path": str(ckpt_dir_l / "ckpt_step_20.pt"),
            },
            {
                "env_id": "Env-v0",
                "method": "proposed",
                "seed": 2,
                "mean_return": 100.0,
                "ckpt_path": str(ckpt_dir_r / "ckpt_step_20.pt"),
            },
        ]
    )
    df.to_csv(results_dir / "summary_raw.csv", index=False)

    # Mock render_side_by_side to verify it gets called once per checkpoint step
    mock_render = MagicMock()
    monkeypatch.setattr("irl.experiments.videos.render_side_by_side", mock_render)

    run_video_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        device="cpu",
        baseline="vanilla",
        method="proposed",
    )

    # Two checkpoints -> two rendered videos
    assert mock_render.call_count == 2

    # Verify each call uses the correct checkpoint pairing and step-tagged filename
    seen_steps = set()
    for _, kwargs in mock_render.call_args_list:
        assert kwargs["env_id"] == "Env-v0"
        assert "vanilla__Env-v0__seed1__A" in str(kwargs["ckpt_left"])
        assert "proposed__Env-v0__seed2__B" in str(kwargs["ckpt_right"])

        out_path = str(kwargs["out_path"])
        assert "/videos/" in out_path.replace("\\", "/")
        assert "Env-v0" in out_path
        # Capture step tag from filename
        if "step000000010" in out_path:
            seen_steps.add(10)
        if "step000000020" in out_path:
            seen_steps.add(20)

    assert seen_steps == {10, 20}
