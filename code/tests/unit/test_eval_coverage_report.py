from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from irl.experiments.evaluation import run_eval_suite


def _write_latest_ckpt(run_dir: Path, *, env_id: str, method: str, seed: int, step: int) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "cfg": {"env": {"id": str(env_id)}, "method": str(method), "seed": int(seed)},
    }
    torch.save(payload, ckpt_dir / "ckpt_latest.pt")


def test_eval_suite_writes_coverage_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs_suite"
    results_dir = tmp_path / "results_suite"
    runs_root.mkdir(parents=True, exist_ok=True)

    r_v1 = runs_root / "vanilla__DummyEval-v0__seed1__cfgA"
    r_v2 = runs_root / "vanilla__DummyEval-v0__seed2__cfgA"
    r_g1 = runs_root / "glpe__DummyEval-v0__seed1__cfgA"

    for rd in (r_v1, r_v2, r_g1):
        rd.mkdir(parents=True, exist_ok=True)

    _write_latest_ckpt(r_v1, env_id="DummyEval-v0", method="vanilla", seed=1, step=100)
    _write_latest_ckpt(r_v2, env_id="DummyEval-v0", method="vanilla", seed=2, step=100)
    _write_latest_ckpt(r_g1, env_id="DummyEval-v0", method="glpe", seed=1, step=50)

    import irl.experiments.evaluation as eval_module

    def fake_evaluate(*, env: str, ckpt: Path, episodes: int, device: str, **kwargs) -> dict:
        return {
            "env_id": str(env),
            "episodes": int(episodes),
            "seed": 0,
            "mean_return": 1.0,
            "std_return": 0.0,
            "min_return": 1.0,
            "max_return": 1.0,
            "mean_length": 5.0,
            "std_length": 0.0,
            "returns": [1.0],
            "lengths": [5],
        }

    monkeypatch.setattr(eval_module, "evaluate", fake_evaluate)

    run_eval_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        episodes=1,
        device="cpu",
        policy_mode="mode",
    )

    cov_path = results_dir / "coverage.csv"
    assert cov_path.exists()

    with cov_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows
    glpe_rows = [r for r in rows if r["env_id"] == "DummyEval-v0" and r["method"] == "glpe"]
    assert len(glpe_rows) == 1
    assert glpe_rows[0]["missing_seeds"] == "2"


def test_eval_suite_strict_coverage_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs_suite"
    results_dir = tmp_path / "results_suite"
    runs_root.mkdir(parents=True, exist_ok=True)

    r_v1 = runs_root / "vanilla__DummyEval-v0__seed1__cfgA"
    r_g1 = runs_root / "glpe__DummyEval-v0__seed2__cfgA"
    for rd in (r_v1, r_g1):
        rd.mkdir(parents=True, exist_ok=True)

    _write_latest_ckpt(r_v1, env_id="DummyEval-v0", method="vanilla", seed=1, step=100)
    _write_latest_ckpt(r_g1, env_id="DummyEval-v0", method="glpe", seed=2, step=100)

    import irl.experiments.evaluation as eval_module

    def fake_evaluate(*, env: str, ckpt: Path, episodes: int, device: str, **kwargs) -> dict:
        return {
            "env_id": str(env),
            "episodes": int(episodes),
            "seed": 0,
            "mean_return": 1.0,
            "std_return": 0.0,
            "min_return": 1.0,
            "max_return": 1.0,
            "mean_length": 5.0,
            "std_length": 0.0,
            "returns": [1.0],
            "lengths": [5],
        }

    monkeypatch.setattr(eval_module, "evaluate", fake_evaluate)

    with pytest.raises(RuntimeError, match="Seed coverage mismatch"):
        run_eval_suite(
            runs_root=runs_root,
            results_dir=results_dir,
            episodes=1,
            device="cpu",
            policy_mode="mode",
            strict_coverage=True,
        )
