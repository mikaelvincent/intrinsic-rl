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

        expected = {5, 10, 15, 20, 25, 30, 35, 40, 45}
        for s in range(1, 50):
            if s in expected:
                assert cm.should_save(s), f"expected should_save at step={s}"
                cm.save(step=s, payload=_payload(s))
            else:
                assert not cm.should_save(s), f"did not expect should_save at step={s}"

        assert cm.should_save(50)
        cm.save(step=50, payload=_payload(50))

        assert not cm.should_save(99)
        assert cm.should_save(100)


def test_checkpoint_prune_preserves_step0() -> None:
    with TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        cm = CheckpointManager(run_dir, interval_steps=10, max_to_keep=2)

        cm.save(step=0, payload=_payload(0))
        cm.save(step=10, payload=_payload(10))
        cm.save(step=20, payload=_payload(20))
        cm.save(step=30, payload=_payload(30))

        ckpt_dir = run_dir / "checkpoints"
        assert (ckpt_dir / "ckpt_step_0.pt").exists(), "baseline checkpoint should be preserved"
        assert (ckpt_dir / "ckpt_step_20.pt").exists()
        assert (ckpt_dir / "ckpt_step_30.pt").exists()
        assert not (ckpt_dir / "ckpt_step_10.pt").exists(), (
            "older non-baseline checkpoints should be pruned"
        )
