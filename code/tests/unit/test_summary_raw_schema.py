from __future__ import annotations

from pathlib import Path

from irl.results.summary import RunResult, _write_raw_csv


def test_summary_raw_includes_eval_metadata_columns(tmp_path: Path) -> None:
    rr = RunResult(
        method="vanilla",
        env_id="DummyEval-v0",
        seed=1,
        ckpt_path=tmp_path / "ckpt.pt",
        ckpt_step=0,
        episodes=1,
        mean_return=1.0,
        std_return=0.0,
        min_return=1.0,
        max_return=1.0,
        mean_length=1.0,
        std_length=0.0,
        policy_mode="sample",
        seed_offset=7,
        episode_seeds_hash="deadbeef",
    )

    out = tmp_path / "summary_raw.csv"
    _write_raw_csv([rr], out)

    header = out.read_text(encoding="utf-8").splitlines()[0]
    cols = header.split(",")

    assert "policy_mode" in cols
    assert "seed_offset" in cols
    assert "episode_seeds_hash" in cols
