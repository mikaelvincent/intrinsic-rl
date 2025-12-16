from pathlib import Path
from tempfile import TemporaryDirectory

from irl.utils.checkpoint import CheckpointManager


def test_checkpoint_unlimited_retention():
    with TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        cm = CheckpointManager(run_dir, interval_steps=1, max_to_keep=None)

        for step in (1, 2, 3, 4):
            cm.save(step=step, payload={"step": step})

        all_ckpts = sorted(p.name for p in (run_dir / "checkpoints").glob("ckpt_step_*.pt"))
        assert all_ckpts == [
            "ckpt_step_1.pt",
            "ckpt_step_2.pt",
            "ckpt_step_3.pt",
            "ckpt_step_4.pt",
        ]
