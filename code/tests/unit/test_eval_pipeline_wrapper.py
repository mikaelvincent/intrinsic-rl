from __future__ import annotations

from pathlib import Path

import pytest
import torch

import irl.multiseed.run_discovery as run_discovery


def test_evaluate_ckpt_wrapper_forwards_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt = tmp_path / "ckpt.pt"
    torch.save(
        {"step": 123, "cfg": {"env": {"id": "DummyEval-v0"}, "method": "glpe", "seed": 7}},
        ckpt,
    )

    calls: dict[str, object] = {}

    def fake_evaluate(*, env: str, ckpt: Path, episodes: int, device: str, policy_mode: str, **_):
        calls.update(env=env, episodes=int(episodes), device=str(device), policy_mode=str(policy_mode))
        return {
            "env_id": str(env),
            "episodes": int(episodes),
            "seed": 7,
            "mean_return": 1.0,
            "std_return": 0.0,
            "min_return": 1.0,
            "max_return": 1.0,
            "mean_length": 5.0,
            "std_length": 0.0,
            "returns": [1.0],
            "lengths": [5],
        }

    monkeypatch.setattr(run_discovery, "evaluate", fake_evaluate)

    rr = run_discovery._evaluate_ckpt(ckpt, episodes=3, device="cpu", policy_mode="mode")

    assert calls == {"env": "DummyEval-v0", "episodes": 3, "device": "cpu", "policy_mode": "mode"}
    assert (rr.method, rr.env_id, rr.seed, rr.ckpt_step, rr.episodes) == (
        "glpe",
        "DummyEval-v0",
        7,
        123,
        3,
    )
    assert rr.ckpt_path == ckpt
