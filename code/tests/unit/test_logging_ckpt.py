import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def test_csv_and_checkpoint_roundtrip():
    # Lazy import to keep dependencies light in test collection
    from irl.cfg import LoggingConfig
    from irl.utils.loggers import MetricLogger
    from irl.utils.checkpoint import CheckpointManager

    with TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        # Logging: CSV only (TensorBoard disabled for the test)
        log_cfg = LoggingConfig(tb=False, csv_interval=1, checkpoint_interval=1000)
        ml = MetricLogger(run_dir, log_cfg)

        # Log a couple of steps
        ml.log(step=0, loss=1.0, ep_return=0.0)
        ml.log(step=1, loss=0.9, ep_return=1.0)
        ml.close()

        csv_path = run_dir / "logs" / "scalars.csv"
        assert csv_path.exists()
        content = csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert content[0].startswith("step,")
        assert any(line.startswith("1,") for line in content[1:])

        # Checkpointing
        cm = CheckpointManager(run_dir, interval_steps=2, max_to_keep=2)
        state1 = {"step": 2, "model": {"w": torch.tensor([1.0, 2.0])}, "meta": {"note": "first"}}
        cm.save(step=2, payload=state1)

        state2 = {"step": 4, "model": {"w": torch.tensor([3.0])}, "meta": {"note": "second"}}
        cm.save(step=4, payload=state2)

        # Load latest
        payload, step = cm.load_latest()
        assert step == 4
        assert payload["meta"]["note"] == "second"

        # Oldest should be pruned after saving second (max_to_keep=2 allows both step2 and step4)
        all_ckpts = sorted(p.name for p in (run_dir / "checkpoints").glob("ckpt_step_*.pt"))
        assert "ckpt_step_2.pt" in all_ckpts
        assert "ckpt_step_4.pt" in all_ckpts


def test_checkpoint_unlimited_retention():
    from irl.utils.checkpoint import CheckpointManager

    with TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        cm = CheckpointManager(run_dir, interval_steps=1, max_to_keep=None)

        for step in (1, 2, 3, 4):
            cm.save(step=step, payload={"step": step, "meta": step})

        # All numbered checkpoints should remain when max_to_keep is None.
        all_ckpts = sorted(p.name for p in (run_dir / "checkpoints").glob("ckpt_step_*.pt"))
        assert all_ckpts == ["ckpt_step_1.pt", "ckpt_step_2.pt", "ckpt_step_3.pt", "ckpt_step_4.pt"]
