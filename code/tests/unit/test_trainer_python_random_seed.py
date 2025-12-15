import random
from dataclasses import replace

from irl.cfg import Config
from irl.trainer import train as run_train


def test_trainer_seeds_python_random(tmp_path, monkeypatch):
    samples = {}

    def fake_seed_everything(seed: int, deterministic: bool = False) -> None:
        from irl.utils.determinism import seed_everything as real_seed

        real_seed(seed, deterministic=deterministic)
        samples["seed"] = seed
        samples["draws"] = [random.random() for _ in range(3)]

    monkeypatch.setattr(
        "irl.trainer.training_setup.seed_everything",
        fake_seed_everything,
        raising=True,
    )

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
    cfg = replace(cfg, intrinsic=replace(cfg.intrinsic, eta=0.0))
    cfg = replace(
        cfg,
        logging=replace(
            cfg.logging,
            tb=False,
            csv_interval=1000,
            checkpoint_interval=1000,
        ),
    )
    cfg = replace(cfg, adaptation=replace(cfg.adaptation, enabled=False))
    cfg = replace(
        cfg,
        evaluation=replace(cfg.evaluation, interval_steps=10_000, episodes=1),
    )

    seed_value = 123
    cfg = replace(cfg, seed=seed_value, device="cpu")

    run_dir = tmp_path / "run_seed_test"
    run_train(cfg, total_steps=4, run_dir=run_dir, resume=False)

    assert samples.get("seed") == seed_value
    draws = samples.get("draws")
    assert isinstance(draws, list) and len(draws) == 3

    prev_state = random.getstate()
    random.seed(seed_value)
    expected = [random.random() for _ in range(3)]
    random.setstate(prev_state)

    assert draws == expected
