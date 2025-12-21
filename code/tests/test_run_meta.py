from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from irl.cfg import Config, validate_config
from irl.trainer import train as run_train
from irl.utils.checkpoint import load_checkpoint


def test_run_meta_written_and_embedded(tmp_path: Path) -> None:
    base = Config()
    cfg = replace(
        base,
        seed=123,
        device="cpu",
        method="vanilla",
        env=replace(base.env, id="MountainCar-v0", vec_envs=1, frame_skip=1, async_vector=False),
        ppo=replace(base.ppo, steps_per_update=1, minibatches=1, epochs=1),
        logging=replace(base.logging, csv_interval=1, checkpoint_interval=100_000),
        evaluation=replace(base.evaluation, interval_steps=0, episodes=1),
        adaptation=replace(base.adaptation, enabled=False),
        exp=replace(base.exp, deterministic=True),
    )
    validate_config(cfg)

    out_dir = run_train(cfg, total_steps=1, run_dir=tmp_path / "run_meta", resume=False)

    meta_path = out_dir / "run_meta.json"
    assert meta_path.exists() and meta_path.is_file()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert isinstance(meta, dict)
    assert str(meta.get("python", "")).strip() != ""
    assert str(meta.get("torch", "")).strip() != ""

    payload = load_checkpoint(out_dir / "checkpoints" / "ckpt_latest.pt", map_location="cpu")
    assert isinstance(payload, dict)
    assert payload.get("run_meta") == meta
