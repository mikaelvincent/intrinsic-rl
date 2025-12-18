from __future__ import annotations

import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from irl.cfg import ConfigError, loads_config
from irl.cfg.schema import LoggingConfig
from irl.cli.validators import normalize_policy_mode
from irl.intrinsic.factory import create_intrinsic_module
from irl.pipelines.discovery import discover_run_dirs_with_latest_ckpt
from irl.results.summary import RunResult, aggregate_results, write_raw_csv, write_summary_csv
from irl.stats.mannwhitney import mannwhitney_u
from irl.utils.checkpoint import compute_cfg_hash
from irl.utils.image import ImagePreprocessConfig, preprocess_image
from irl.utils.images import infer_channels_hw
from irl.utils.io import atomic_write_csv
from irl.utils.loggers import MetricLogger
from irl.utils.runs import parse_run_name
from irl.visualization.data import aggregate_runs


def _read_header(path: Path) -> list[str]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        row = next(csv.reader(f), None)
    assert row is not None
    return [str(x) for x in row]


def test_atomic_write_csv_overwrites_and_rejects_extra(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    tmp = path.with_suffix(path.suffix + ".tmp")

    atomic_write_csv(path, ["a", "b"], [{"a": 1, "b": "x"}])
    assert path.exists()
    assert not tmp.exists()

    atomic_write_csv(path, ["a", "b"], [{"a": 2, "b": "y"}])
    assert path.read_text(encoding="utf-8").splitlines() == ["a,b", "2,y"]
    assert not tmp.exists()

    with pytest.raises(ValueError):
        atomic_write_csv(tmp_path / "bad.csv", ["a"], [{"a": 1, "b": 2}])


def test_metric_logger_persists_cfg_hash(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ml = MetricLogger(run_dir, LoggingConfig(csv_interval=1, checkpoint_interval=10_000))
    try:
        params = {
            "seed": 7,
            "device": "cpu",
            "env": {"id": "MountainCar-v0", "vec_envs": 2},
            "exp": {"deterministic": True},
        }

        ml.log_hparams(params)

        cfg_path = run_dir / "config.json"
        hash_path = run_dir / "config_hash.txt"

        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        expected = compute_cfg_hash(params)

        assert payload.get("cfg_hash") == expected
        assert hash_path.read_text(encoding="utf-8").strip() == expected

        before = cfg_path.read_text(encoding="utf-8")
        ml.log_hparams({"seed": 999})
        assert cfg_path.read_text(encoding="utf-8") == before
    finally:
        ml.close()


def test_compute_cfg_hash_is_stable_and_selective() -> None:
    base = {
        "method": "vanilla",
        "seed": 7,
        "env": {"id": "MountainCar-v0", "vec_envs": 8},
        "ppo": {"minibatches": 32, "steps_per_update": 128},
        "exp": {"deterministic": True},
    }
    reordered = {
        "ppo": {"steps_per_update": 128, "minibatches": 32},
        "env": {"vec_envs": 8, "id": "MountainCar-v0"},
        "seed": 7,
        "method": "vanilla",
        "exp": {"deterministic": True},
    }
    assert compute_cfg_hash(base) == compute_cfg_hash(reordered)

    changed = {
        "method": "vanilla",
        "seed": 7,
        "env": {"id": "MountainCar-v0", "vec_envs": 8},
        "ppo": {"minibatches": 64, "steps_per_update": 128},
        "exp": {"deterministic": True},
    }
    assert compute_cfg_hash(changed) != compute_cfg_hash(base)

    with_profile_flag = {
        "method": "vanilla",
        "seed": 7,
        "env": {"id": "MountainCar-v0", "vec_envs": 8},
        "ppo": {"minibatches": 32, "steps_per_update": 128},
        "exp": {"deterministic": True, "profile_cuda_sync": True},
    }
    assert compute_cfg_hash(with_profile_flag) == compute_cfg_hash(base)


def test_cli_and_run_name_parsers() -> None:
    assert normalize_policy_mode(" Mode ", allowed=("mode", "sample"), name="policy") == "mode"
    with pytest.raises(ValueError):
        normalize_policy_mode("bad", allowed=("mode", "sample"), name="policy")

    assert parse_run_name("glpe__MountainCar-v0__seed42__cfgA") == {
        "method": "glpe",
        "env": "MountainCar-v0",
        "seed": "42",
    }
    assert parse_run_name("vanilla__DummyEval-v0__notseed5__cfg") == {
        "method": "vanilla",
        "env": "DummyEval-v0",
    }


def test_loads_config_parses_numbers_and_validates() -> None:
    cfg = loads_config(
        """
method: vanilla
env:
  vec_envs: "8"
ppo:
  steps_per_update: "2_048"
  minibatches: 3_2
  learning_rate: "3e-4"
""".lstrip()
    )
    assert int(cfg.env.vec_envs) == 8
    assert int(cfg.ppo.steps_per_update) == 2048
    assert int(cfg.ppo.minibatches) == 32
    assert abs(float(cfg.ppo.learning_rate) - 3e-4) < 1e-12

    with pytest.raises(ConfigError):
        loads_config("seed: true\n")

    with pytest.raises(ConfigError):
        loads_config(
            """
seed: 1
unknown_top_level: 123
""".lstrip()
        )

    with pytest.raises(ConfigError):
        loads_config(
            """
method: vanilla
env:
  vec_envs: 8
ppo:
  steps_per_update: 130
  minibatches: 64
""".lstrip()
        )


def test_glpe_variants_enforce_constraints() -> None:
    cfg_lp = loads_config(
        """
method: glpe_lp_only
intrinsic:
  eta: 0.1
  alpha_impact: 0.0
""".lstrip()
    )
    assert str(cfg_lp.method).lower() == "glpe_lp_only"

    cfg_ng = loads_config(
        """
method: glpe_nogate
intrinsic:
  eta: 0.1
  gate:
    enabled: false
""".lstrip()
    )
    assert str(cfg_ng.method).lower() == "glpe_nogate"

    with pytest.raises(ConfigError):
        loads_config(
            """
method: glpe_lp_only
intrinsic:
  eta: 0.1
  alpha_impact: 0.25
""".lstrip()
        )

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    mod_lp = create_intrinsic_module(
        "glpe_lp_only",
        obs_space,
        act_space,
        device="cpu",
        alpha_impact=1.0,
        alpha_lp=0.5,
        gating_enabled=True,
    )
    assert float(mod_lp.alpha_impact) == 0.0

    mod_ng = create_intrinsic_module(
        "glpe_nogate",
        obs_space,
        act_space,
        device="cpu",
        alpha_impact=1.0,
        alpha_lp=0.5,
        gating_enabled=True,
    )
    assert bool(getattr(mod_ng, "gating_enabled", True)) is False


def test_results_csv_headers_are_stable(tmp_path: Path) -> None:
    rows = [
        RunResult(
            method="vanilla",
            env_id="DummyEval-v0",
            seed=1,
            ckpt_path=tmp_path / "ckpt_seed1.pt",
            ckpt_step=10,
            episodes=10,
            mean_return=1.0,
            std_return=0.0,
            min_return=1.0,
            max_return=1.0,
            mean_length=5.0,
            std_length=0.0,
        ),
        RunResult(
            method="vanilla",
            env_id="DummyEval-v0",
            seed=2,
            ckpt_path=tmp_path / "ckpt_seed2.pt",
            ckpt_step=20,
            episodes=10,
            mean_return=2.0,
            std_return=0.0,
            min_return=2.0,
            max_return=2.0,
            mean_length=5.0,
            std_length=0.0,
        ),
    ]

    raw_path = tmp_path / "summary_raw.csv"
    write_raw_csv(rows, raw_path)
    assert _read_header(raw_path) == [
        "method",
        "env_id",
        "seed",
        "policy_mode",
        "seed_offset",
        "episode_seeds_hash",
        "ckpt_step",
        "episodes",
        "mean_return",
        "std_return",
        "min_return",
        "max_return",
        "mean_length",
        "std_length",
        "ckpt_path",
    ]

    summary_path = tmp_path / "summary.csv"
    agg = aggregate_results(rows, n_boot=20)
    write_summary_csv(agg, summary_path)
    assert _read_header(summary_path) == [
        "method",
        "env_id",
        "per_seed_ckpt_policy",
        "episodes_per_seed",
        "n_runs",
        "n_seeds",
        "seeds",
        "mean_return_mean",
        "mean_return_std",
        "mean_return_se",
        "mean_return_ci95_lo",
        "mean_return_ci95_hi",
        "mean_length_mean",
        "mean_length_std",
        "mean_length_se",
        "mean_length_ci95_lo",
        "mean_length_ci95_hi",
        "step_min",
        "step_max",
        "step_mean",
    ]


def _rr(*, seed: int, step: int, mean_return: float, tmp_path: Path) -> RunResult:
    return RunResult(
        method="vanilla",
        env_id="DummyEval-v0",
        seed=int(seed),
        ckpt_path=tmp_path / f"ckpt_seed{seed}_step{step}.pt",
        ckpt_step=int(step),
        episodes=10,
        mean_return=float(mean_return),
        std_return=0.0,
        min_return=float(mean_return),
        max_return=float(mean_return),
        mean_length=0.0,
        std_length=0.0,
    )


def test_aggregate_results_dedups_latest_ckpt_per_seed(tmp_path: Path) -> None:
    rows = [
        _rr(seed=1, step=10, mean_return=1.0, tmp_path=tmp_path),
        _rr(seed=1, step=20, mean_return=3.0, tmp_path=tmp_path),
        _rr(seed=2, step=20, mean_return=5.0, tmp_path=tmp_path),
    ]

    agg = aggregate_results(rows, n_boot=50)
    assert len(agg) == 1
    r = agg[0]

    assert int(r["n_runs"]) == 3
    assert int(r["n_seeds"]) == 2
    assert str(r["seeds"]) == "1,2"
    assert float(r["mean_return_mean"]) == 4.0
    assert int(r["step_min"]) == 20
    assert int(r["step_max"]) == 20
    assert int(r["step_mean"]) == 20


def test_mannwhitney_u_respects_direction() -> None:
    x = np.arange(50, dtype=np.float64) + 100.0
    y = np.arange(50, dtype=np.float64)

    res = mannwhitney_u(x, y, alternative="greater")
    assert res.p_value < 1e-12
    assert res.cliffs_delta > 0.9

    res_swap = mannwhitney_u(y, x, alternative="greater")
    assert res_swap.cliffs_delta < -0.9

    res_less = mannwhitney_u(x, y, alternative="less")
    assert res_less.p_value > 1.0 - 1e-12


def _write_ckpt(path: Path, *, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": int(step), "cfg": {"env": {"id": "DummyEval-v0"}}}, path)


def test_discovery_picks_latest_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "runs"

    run_latest = root / "groupA" / "vanilla__DummyEval-v0__seed1__cfgA"
    _write_ckpt(run_latest / "checkpoints" / "ckpt_latest.pt", step=10)

    run_steps = root / "glpe__DummyEval-v0__seed2__cfgB"
    _write_ckpt(run_steps / "checkpoints" / "ckpt_step_10.pt", step=10)
    _write_ckpt(run_steps / "checkpoints" / "ckpt_step_20.pt", step=20)

    found = discover_run_dirs_with_latest_ckpt(root)
    by_name = {rd.name: ckpt.name for rd, ckpt in found}

    assert len(found) == 2
    assert by_name["vanilla__DummyEval-v0__seed1__cfgA"] == "ckpt_latest.pt"
    assert by_name["glpe__DummyEval-v0__seed2__cfgB"] == "ckpt_step_20.pt"


def test_aggregate_runs_dedups_steps_and_falls_back(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "vanilla__MountainCar-v0__seed1__20250101-000000"
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    (logs / "scalars.csv").write_text(
        "step,reward_total_mean\n0,0.0\n0,0.5\n1000,1.5\n",
        encoding="utf-8",
    )

    agg = aggregate_runs([run_dir], metric="reward_total_mean", smooth=1)
    assert agg.steps.tolist() == [0, 1000]
    assert agg.mean.tolist() == [0.5, 1.5]

    with pytest.warns(UserWarning, match="reward_mean"):
        agg_fb = aggregate_runs([run_dir], metric="reward_mean", smooth=1)
    assert agg_fb.steps.tolist() == [0, 1000]
    assert agg_fb.mean.tolist() == [0.5, 1.5]


def test_preprocess_image_scales_and_infers_channels() -> None:
    assert infer_channels_hw((32, 32, 3)) == (3, (32, 32))
    assert infer_channels_hw((3, 32, 32)) == (3, (32, 32))

    rng = np.random.default_rng(0)
    arr_255 = rng.integers(0, 256, size=(2, 8, 8, 3), dtype=np.uint8).astype(np.float32)

    cfg = ImagePreprocessConfig(grayscale=False, scale_uint8=True, channels_first=True)
    t = preprocess_image(arr_255, cfg=cfg, device="cpu")
    assert t.shape == (2, 3, 8, 8)
    assert float(t.min()) >= -1e-6
    assert float(t.max()) <= 1.0 + 1e-6
