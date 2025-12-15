from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def test_csv_and_checkpoint_roundtrip():
    from irl.cfg import LoggingConfig
    from irl.utils.checkpoint import CheckpointManager
    from irl.utils.loggers import MetricLogger

    with TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        log_cfg = LoggingConfig(tb=False, csv_interval=1, checkpoint_interval=1000)
        ml = MetricLogger(run_dir, log_cfg)

        ml.log(step=0, loss=1.0, ep_return=0.0)
        ml.log(step=1, loss=0.9, ep_return=1.0)
        ml.close()

        csv_path = run_dir / "logs" / "scalars.csv"
        assert csv_path.exists()
        content = csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert content[0].startswith("step,")
        assert any(line.startswith("1,") for line in content[1:])

        cm = CheckpointManager(run_dir, interval_steps=2, max_to_keep=2)
        state1 = {"step": 2, "model": {"w": torch.tensor([1.0, 2.0])}, "meta": {"note": "first"}}
        cm.save(step=2, payload=state1)

        state2 = {"step": 4, "model": {"w": torch.tensor([3.0])}, "meta": {"note": "second"}}
        cm.save(step=4, payload=state2)

        payload, step = cm.load_latest()
        assert step == 4
        assert payload["meta"]["note"] == "second"

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

        all_ckpts = sorted(p.name for p in (run_dir / "checkpoints").glob("ckpt_step_*.pt"))
        assert all_ckpts == ["ckpt_step_1.pt", "ckpt_step_2.pt", "ckpt_step_3.pt", "ckpt_step_4.pt"]
