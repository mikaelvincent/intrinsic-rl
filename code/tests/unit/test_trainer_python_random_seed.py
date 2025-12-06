from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import pytest

from irl.cfg import Config
from irl.trainer import train as run_train


def test_trainer_seeds_python_random(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that the trainer seeds Python's RNG at startup.

    The test monkeypatches the trainer's ``seed_everything`` helper to
    call the real implementation and then record a short
    :func:`random.random` sequence immediately after seeding. After a
    tiny training run, the same sequence is reproduced independently to
    confirm that the trainer reinitialises the Python RNG at the start
    of training.
    """
    samples: dict[str, object] = {}

    def fake_seed_everything(seed: int, deterministic: bool = False) -> None:
        # Call the real seeding helper so NumPy / torch are configured as usual.
        from irl.utils.determinism import seed_everything as real_seed

        real_seed(seed, deterministic=deterministic)

        # Immediately draw a small sequence from Python's RNG after seeding.
        samples["seed"] = seed
        samples["draws"] = [random.random() for _ in range(3)]

    # Patch the alias used inside the trainer loop (irl.trainer.loop.train).
    monkeypatch.setattr("irl.trainer.loop.seed_everything", fake_seed_everything, raising=True)

    # Build a tiny config for a short MountainCar run.
    cfg = Config()
    cfg = replace(cfg, method="vanilla")
    cfg = replace(
        cfg,
        env=replace(
            cfg.env,
            id="MountainCar-v0",
            vec_envs=1,
            frame_skip=1,
            domain_randomization=False,
        ),
    )
    cfg = replace(
        cfg,
        ppo=replace(
            cfg.ppo,
            steps_per_update=4,
            minibatches=1,
            epochs=1,
            entropy_coef=0.0,
        ),
    )
    # Disable intrinsic rewards to keep the run minimal.
    cfg = replace(cfg, intrinsic=replace(cfg.intrinsic, eta=0.0))
    # Keep logging light for the test.
    cfg = replace(
        cfg,
        logging=replace(
            cfg.logging,
            tb=False,
            csv_interval=1000,
            checkpoint_interval=1000,
        ),
    )
    # Turn off adaptation and effectively disable eval.
    cfg = replace(cfg, adaptation=replace(cfg.adaptation, enabled=False))
    cfg = replace(
        cfg,
        evaluation=replace(cfg.evaluation, interval_steps=10_000, episodes=1),
    )

    seed_value = 123
    cfg = replace(cfg, seed=seed_value, device="cpu")

    run_dir = tmp_path / "run_seed_test"
    run_train(cfg, total_steps=4, run_dir=run_dir, resume=False)

    # fake_seed_everything must have been called once at the start of training.
    assert samples.get("seed") == seed_value
    draws = samples.get("draws")
    assert isinstance(draws, list) and len(draws) == 3

    # Independently reproduce the expected sequence from Python's RNG for the same seed.
    prev_state = random.getstate()
    random.seed(seed_value)
    expected = [random.random() for _ in range(3)]
    random.setstate(prev_state)

    assert draws == expected
