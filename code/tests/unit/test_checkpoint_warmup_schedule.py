from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from irl.utils.checkpoint import CheckpointManager


def _payload(step: int) -> dict:
    return {"step": int(step), "meta": {"note": "test"}}


def test_checkpoint_should_save_warmup_then_interval() -> None:
    with TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        cm = CheckpointManager(run_dir, interval_steps=50, max_to_keep=None)

        cm.save(step=0, payload=_payload(0))

        assert not cm.should_save(4)
        assert cm.should_save(5)
        cm.save(step=5, payload=_payload(5))

        for step in range(6, 50):
            if cm.should_save(step):
                cm.save(step=step, payload=_payload(step))

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
        kept = sorted(p.name for p in ckpt_dir.glob("ckpt_step_*.pt"))

        assert "ckpt_step_0.pt" in kept
        assert "ckpt_step_30.pt" in kept
        assert "ckpt_step_20.pt" in kept
        assert "ckpt_step_10.pt" not in kept
