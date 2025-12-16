from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from irl.utils.checkpoint import CheckpointManager


def _payload(step: int) -> dict:
    return {"step": int(step), "meta": {"note": "test"}}


def test_checkpoint_warmup_schedule_before_interval() -> None:
    with TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        cm = CheckpointManager(run_dir, interval_steps=50, max_to_keep=None)

        cm.save(step=0, payload=_payload(0))

        for target in (5, 10, 15, 20, 25, 30, 35, 40, 45):
            assert not cm.should_save(target - 1)
            assert cm.should_save(target)
            cm.save(step=target, payload=_payload(target))

        assert not cm.should_save(49)
        assert cm.should_save(50)
        cm.save(step=50, payload=_payload(50))

        assert not cm.should_save(99)
        assert cm.should_save(100)


def test_checkpoint_prune_preserves_step0() -> None:
    with TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        cm = CheckpointManager(run_dir, interval_steps=10, max_to_keep=2)

        for step in (0, 10, 20, 30):
            cm.save(step=step, payload=_payload(step))

        ckpt_dir = run_dir / "checkpoints"
        assert (ckpt_dir / "ckpt_step_0.pt").exists()
        assert (ckpt_dir / "ckpt_step_20.pt").exists()
        assert (ckpt_dir / "ckpt_step_30.pt").exists()
        assert not (ckpt_dir / "ckpt_step_10.pt").exists()


def test_checkpoint_no_prune_when_unlimited() -> None:
    with TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        cm = CheckpointManager(run_dir, interval_steps=1, max_to_keep=None)

        for step in (0, 1, 2, 3):
            cm.save(step=step, payload=_payload(step))

        ckpt_dir = run_dir / "checkpoints"
        kept = sorted(p.name for p in ckpt_dir.glob("ckpt_step_*.pt"))
        assert kept == [
            "ckpt_step_0.pt",
            "ckpt_step_1.pt",
            "ckpt_step_2.pt",
            "ckpt_step_3.pt",
        ]
