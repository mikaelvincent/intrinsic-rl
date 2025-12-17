from __future__ import annotations

from pathlib import Path

from irl.multiseed.results import RunResult, _aggregate


def _rr(*, seed: int, step: int, mean_return: float) -> RunResult:
    return RunResult(
        method="vanilla",
        env_id="DummyEval-v0",
        seed=int(seed),
        ckpt_path=Path(f"/tmp/ckpt_seed{seed}_step{step}.pt"),
        ckpt_step=int(step),
        episodes=10,
        mean_return=float(mean_return),
        std_return=0.0,
        min_return=float(mean_return),
        max_return=float(mean_return),
        mean_length=0.0,
        std_length=0.0,
    )


def test_aggregate_dedups_to_latest_ckpt_per_seed():
    rows = [
        _rr(seed=1, step=10, mean_return=1.0),
        _rr(seed=1, step=20, mean_return=3.0),
        _rr(seed=2, step=20, mean_return=5.0),
    ]

    agg = _aggregate(rows, n_boot=200)
    assert len(agg) == 1
    r = agg[0]

    assert int(r["n_runs"]) == 3
    assert int(r["n_seeds"]) == 2
    assert str(r["seeds"]) == "1,2"
    assert float(r["mean_return_mean"]) == 4.0
    assert int(r["step_min"]) == 20
    assert int(r["step_max"]) == 20
    assert int(r["step_mean"]) == 20
