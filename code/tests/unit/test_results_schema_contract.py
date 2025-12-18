from __future__ import annotations

import csv
from pathlib import Path

from irl.multiseed.results import read_summary_raw
from irl.results.summary import RunResult, aggregate_results, write_raw_csv, write_summary_csv


def _read_header(path: Path) -> list[str]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        row = next(r, None)
    assert row is not None
    return [str(x) for x in row]


def _rr(tmp_path: Path, *, seed: int, step: int, mean_return: float) -> RunResult:
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
        mean_length=5.0,
        std_length=0.0,
        policy_mode="mode",
        seed_offset=0,
        episode_seeds_hash="deadbeef",
    )


def test_summary_raw_csv_schema_is_stable(tmp_path: Path) -> None:
    rows = [
        _rr(tmp_path, seed=1, step=10, mean_return=1.0),
        _rr(tmp_path, seed=2, step=20, mean_return=2.0),
    ]
    out = tmp_path / "summary_raw.csv"
    write_raw_csv(rows, out)

    expected = [
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
    assert _read_header(out) == expected

    parsed = read_summary_raw(out)
    assert len(parsed) == 2
    assert set(parsed[0].keys()) == {
        "method",
        "env_id",
        "seed",
        "ckpt_step",
        "episodes",
        "mean_return",
        "std_return",
        "min_return",
        "max_return",
        "mean_length",
        "std_length",
        "ckpt_path",
    }


def test_summary_csv_schema_is_stable(tmp_path: Path) -> None:
    rows = [
        _rr(tmp_path, seed=1, step=10, mean_return=1.0),
        _rr(tmp_path, seed=1, step=20, mean_return=3.0),
        _rr(tmp_path, seed=2, step=20, mean_return=5.0),
    ]
    agg = aggregate_results(rows, n_boot=10)
    out = tmp_path / "summary.csv"
    write_summary_csv(agg, out)

    expected = [
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
    assert _read_header(out) == expected
