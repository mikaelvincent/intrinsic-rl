from __future__ import annotations

from irl.utils.determinism import seed_everything


def test_seed_everything_deterministic_cpu_no_error() -> None:
    seed_everything(0, deterministic=True, device="cpu")


def test_seed_everything_warn_only_toggle(monkeypatch) -> None:
    monkeypatch.setenv("IRL_DETERMINISTIC_WARN_ONLY", "1")
    seed_everything(0, deterministic=True, device="cuda")
