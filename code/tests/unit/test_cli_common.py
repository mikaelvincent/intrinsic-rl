from __future__ import annotations

from dataclasses import replace

import pytest
import typer

from irl.cfg.schema import Config
from irl.cli.common import resolve_total_steps, validate_policy_mode
from irl.cli.validators import normalize_policy_mode


def test_normalize_policy_mode_normalizes() -> None:
    assert (
        normalize_policy_mode(" Mode ", allowed=("mode", "sample"), name="policy_mode") == "mode"
    )


def test_normalize_policy_mode_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        normalize_policy_mode("both", allowed=("mode", "sample"), name="policy_mode")


def test_validate_policy_mode_raises_badparameter() -> None:
    with pytest.raises(typer.BadParameter):
        validate_policy_mode("bad", allowed=("mode", "sample"), option="--policy")


def test_resolve_total_steps_precedence() -> None:
    base = Config()
    cfg = replace(base, exp=replace(base.exp, total_steps=42))

    assert resolve_total_steps(cfg, cli_total_steps=None, default_total_steps=10_000, prefer_cfg=True) == 42
    assert resolve_total_steps(cfg, cli_total_steps=100, default_total_steps=10_000, prefer_cfg=False) == 100
    assert resolve_total_steps(cfg, cli_total_steps=None, default_total_steps=10_000, prefer_cfg=False) == 42
