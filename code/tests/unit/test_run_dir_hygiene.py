from __future__ import annotations

from pathlib import Path

import pytest

from irl.cfg import Config
from irl.trainer import train as run_train


def test_nonempty_run_dir_no_resume_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dirty"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "ckpt_latest.pt").write_bytes(b"not a real checkpoint")

    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / "scalars.csv").write_text("step,reward_total_mean\n0,0.0\n", encoding="utf-8")

    cfg = Config()
    with pytest.raises(RuntimeError, match="resume"):
        run_train(cfg, total_steps=1, run_dir=run_dir, resume=False)
