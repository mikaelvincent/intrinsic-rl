from __future__ import annotations

from pathlib import Path

import torch

from irl.experiments.evaluation import _discover_run_dirs_with_ckpt


def _write_ckpt(path: Path, *, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "cfg": {"env": {"id": "DummyEval-v0"}, "method": "vanilla", "seed": 1},
        },
        path,
    )


def test_discover_run_dirs_with_ckpt_handles_nested(tmp_path: Path) -> None:
    root = tmp_path / "runs"

    run_latest = root / "groupA" / "vanilla__DummyEval-v0__seed1__cfgA"
    _write_ckpt(run_latest / "checkpoints" / "ckpt_latest.pt", step=10)

    run_steps = root / "glpe__DummyEval-v0__seed2__cfgB"
    _write_ckpt(run_steps / "checkpoints" / "ckpt_step_10.pt", step=10)
    _write_ckpt(run_steps / "checkpoints" / "ckpt_step_20.pt", step=20)

    found = _discover_run_dirs_with_ckpt(root)
    by_name = {rd.name: ckpt.name for rd, ckpt in found}

    assert by_name["vanilla__DummyEval-v0__seed1__cfgA"] == "ckpt_latest.pt"
    assert by_name["glpe__DummyEval-v0__seed2__cfgB"] == "ckpt_step_20.pt"
