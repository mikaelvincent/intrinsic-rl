from __future__ import annotations

import csv
from pathlib import Path

import pytest

from irl.cfg import ConfigError, loads_config
from irl.cli.validators import normalize_policy_mode
from irl.results.summary import RunResult, aggregate_results, write_raw_csv, write_summary_csv
from irl.utils.runs import parse_run_name


def _read_header(path: Path) -> list[str]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        row = next(csv.reader(f), None)
    assert row is not None
    return [str(x) for x in row]


def test_normalize_policy_mode_normalizes_and_rejects() -> None:
    assert normalize_policy_mode(" Mode ", allowed=("mode", "sample"), name="policy") == "mode"
    with pytest.raises(ValueError):
        normalize_policy_mode("bad", allowed=("mode", "sample"), name="policy")


def test_loads_config_parses_numbers_and_rejects_bool_seed() -> None:
    cfg = loads_config(
        """
method: vanilla
env:
  vec_envs: "8"
ppo:
  steps_per_update: "2_048"
  minibatches: 3_2
  learning_rate: "3e-4"
"""
    )
    assert int(cfg.env.vec_envs) == 8
    assert int(cfg.ppo.steps_per_update) == 2048
    assert int(cfg.ppo.minibatches) == 32
    assert abs(float(cfg.ppo.learning_rate) - 3e-4) < 1e-12

    with pytest.raises(ConfigError) as ei:
        loads_config(
            """
seed: true
"""
        )
    assert "config.seed" in str(ei.value)


def test_parse_run_name_extracts_method_env_seed() -> None:
    assert parse_run_name("glpe__MountainCar-v0__seed42__cfgA") == {
        "method": "glpe",
        "env": "MountainCar-v0",
        "seed": "42",
    }
    assert parse_run_name("vanilla__CartPole-v1__seed1__cfgA__extra") == {
        "method": "vanilla",
        "env": "CartPole-v1",
        "seed": "1",
    }
    assert parse_run_name("vanilla__DummyEval-v0__notseed5__cfg") == {
        "method": "vanilla",
        "env": "DummyEval-v0",
    }


def test_results_csv_headers_stable(tmp_path: Path) -> None:
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
    agg = aggregate_results(rows, n_boot=10)
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
