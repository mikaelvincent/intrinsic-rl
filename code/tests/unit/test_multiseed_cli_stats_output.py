from __future__ import annotations

import csv
from pathlib import Path

from irl.multiseed.cli import cli_stats


def _write_summary_raw(path: Path, rows: list[dict[str, object]]) -> None:
    cols = [
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
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def test_cli_stats_prints_correct_median_delta(tmp_path: Path, capsys) -> None:
    p = tmp_path / "summary_raw.csv"

    rows: list[dict[str, object]] = []
    for seed in (1, 2, 3):
        rows.append(
            {
                "method": "A",
                "env_id": "DummyEval-v0",
                "seed": seed,
                "ckpt_step": 100,
                "episodes": 1,
                "mean_return": 1.0,
                "std_return": 0.0,
                "min_return": 1.0,
                "max_return": 1.0,
                "mean_length": 1.0,
                "std_length": 0.0,
                "ckpt_path": "",
            }
        )

    for seed, val in zip((1, 2, 3), (0.0, 0.0, 10.0)):
        rows.append(
            {
                "method": "B",
                "env_id": "DummyEval-v0",
                "seed": seed,
                "ckpt_step": 100,
                "episodes": 1,
                "mean_return": val,
                "std_return": 0.0,
                "min_return": val,
                "max_return": val,
                "mean_length": 1.0,
                "std_length": 0.0,
                "ckpt_path": "",
            }
        )

    _write_summary_raw(p, rows)

    cli_stats(
        summary_raw=p,
        env="DummyEval-v0",
        method_a="A",
        method_b="B",
        metric="mean_return",
        boot=0,
        alternative="two-sided",
        latest_per_seed=True,
    )

    out = capsys.readouterr().out
    assert "Medians :" in out
    assert "(Δ=+1.000)" in out
