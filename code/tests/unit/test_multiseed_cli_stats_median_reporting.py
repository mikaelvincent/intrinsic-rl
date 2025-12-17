from __future__ import annotations

import csv
import re
from pathlib import Path

from typer.testing import CliRunner

from irl.multiseed.cli import app


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def test_multiseed_stats_prints_median_y(tmp_path: Path) -> None:
    summary_raw = tmp_path / "summary_raw.csv"
    env = "DummyEval-v0"

    # method B: mean=3.333..., median=0.0 (intentionally different)
    rows: list[dict[str, object]] = []
    for seed, v in [(1, 0.0), (2, 10.0), (3, 10.0)]:
        rows.append(
            {
                "method": "A",
                "env_id": env,
                "seed": seed,
                "ckpt_step": 100,
                "episodes": 10,
                "mean_return": v,
                "std_return": 0.0,
                "min_return": v,
                "max_return": v,
                "mean_length": 100.0,
                "std_length": 0.0,
                "ckpt_path": f"/tmp/ckpt_A_seed{seed}.pt",
            }
        )
    for seed, v in [(1, 0.0), (2, 0.0), (3, 10.0)]:
        rows.append(
            {
                "method": "B",
                "env_id": env,
                "seed": seed,
                "ckpt_step": 100,
                "episodes": 10,
                "mean_return": v,
                "std_return": 0.0,
                "min_return": v,
                "max_return": v,
                "mean_length": 100.0,
                "std_length": 0.0,
                "ckpt_path": f"/tmp/ckpt_B_seed{seed}.pt",
            }
        )

    _write_summary_raw(summary_raw, rows)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "stats",
            "--summary-raw",
            str(summary_raw),
            "--env",
            env,
            "--method-a",
            "A",
            "--method-b",
            "B",
            "--metric",
            "mean_return",
            "--boot",
            "0",
        ],
        color=False,
    )
    assert res.exit_code == 0, res.stdout

    m = re.search(r"Medians\s*:\s*A=([-0-9.]+),\s*B=([-0-9.]+)", res.stdout)
    assert m is not None, res.stdout

    median_b = float(m.group(2))
    assert abs(median_b - 0.0) < 1e-9
