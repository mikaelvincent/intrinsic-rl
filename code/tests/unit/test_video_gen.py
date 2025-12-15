from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import torch

from irl.experiments import run_video_suite
import irl.video


def test_render_video_frame_composition(tmp_path, monkeypatch):
    mock_pol = MagicMock()
    mock_pol.act.return_value = (torch.tensor([0]), torch.tensor([0.0]))

    def mock_load(path, device):
        return mock_pol, {"env": {"frame_skip": 1, "discrete_actions": True}}

    monkeypatch.setattr(irl.video, "_load_policy_for_eval", mock_load)

    class MockEnv:
        def __init__(self, *args, **kwargs):
            self.action_space = MagicMock()
            self.action_space.shape = ()

        def reset(self, **kwargs):
            return np.zeros(2), {}

        def step(self, action):
            return np.zeros(2), 1.0, False, False, {}

        def render(self):
            return np.full((10, 10, 3), 255, dtype=np.uint8)

        def close(self):
            pass

    mock_manager = MagicMock()
    mock_manager.make.return_value = MockEnv()
    monkeypatch.setattr(irl.video, "EnvManager", MagicMock(return_value=mock_manager))

    mock_mimsave = MagicMock()
    monkeypatch.setattr(irl.video.imageio, "mimsave", mock_mimsave)

    out_path = tmp_path / "test_video.mp4"
    irl.video.render_side_by_side(
        env_id="Dummy-v0",
        ckpt_left=Path("dummy_left.pt"),
        ckpt_right=Path("dummy_right.pt"),
        out_path=out_path,
        max_steps=5,
        device="cpu",
    )

    assert mock_mimsave.called
    args, _ = mock_mimsave.call_args
    path_arg, frames_arg = args
    assert str(path_arg) == str(out_path)
    assert len(frames_arg) > 0
    frame = frames_arg[0]
    assert frame.shape[2] == 3
    assert frame.shape[1] >= 20


def test_video_suite_renders_every_checkpoint(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    results_dir = tmp_path / "results"

    runs_root.mkdir()
    results_dir.mkdir()

    run_l = runs_root / "vanilla__Env-v0__seed1__A"
    run_r = runs_root / "proposed__Env-v0__seed2__B"
    run_l.mkdir(parents=True)
    run_r.mkdir(parents=True)

    ckpt_dir_l = run_l / "checkpoints"
    ckpt_dir_r = run_r / "checkpoints"
    ckpt_dir_l.mkdir()
    ckpt_dir_r.mkdir()

    (ckpt_dir_l / "ckpt_step_10.pt").touch()
    (ckpt_dir_l / "ckpt_step_20.pt").touch()
    (ckpt_dir_r / "ckpt_step_10.pt").touch()
    (ckpt_dir_r / "ckpt_step_20.pt").touch()
    (ckpt_dir_l / "ckpt_latest.pt").touch()
    (ckpt_dir_r / "ckpt_latest.pt").touch()

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

    seen_steps = set()
    for _, kwargs in mock_render.call_args_list:
        assert kwargs["env_id"] == "Env-v0"
        assert "vanilla__Env-v0__seed1__A" in str(kwargs["ckpt_left"])
        assert "proposed__Env-v0__seed2__B" in str(kwargs["ckpt_right"])

        out_path = str(kwargs["out_path"])
        assert "/videos/" in out_path.replace("\\", "/")
        assert "Env-v0" in out_path
        if "step000000010" in out_path:
            seen_steps.add(10)
        if "step000000020" in out_path:
            seen_steps.add(20)

    assert seen_steps == {10, 20}
