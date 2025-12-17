from __future__ import annotations

from dataclasses import replace

import pytest
import typer

from irl.cfg.schema import Config
from irl.cli.common import resolve_total_steps, validate_policy_mode


def test_validate_policy_mode_normalizes_and_rejects() -> None:
    assert validate_policy_mode(" Mode ", allowed=("mode", "sample"), option="--policy") == "mode"
    with pytest.raises(typer.BadParameter):
        validate_policy_mode("bad", allowed=("mode", "sample"), option="--policy")


def test_resolve_total_steps_precedence() -> None:
    base = Config()
    cfg = replace(base, exp=replace(base.exp, total_steps=42))

    assert (
        resolve_total_steps(cfg, cli_total_steps=None, default_total_steps=10_000, prefer_cfg=True)
        == 42
    )
    assert (
        resolve_total_steps(cfg, cli_total_steps=100, default_total_steps=10_000, prefer_cfg=False)
        == 100
    )
    assert (
        resolve_total_steps(cfg, cli_total_steps=None, default_total_steps=10_000, prefer_cfg=False)
        == 42
    )
