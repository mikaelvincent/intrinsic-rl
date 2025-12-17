from __future__ import annotations

from dataclasses import replace

import pytest
import typer

from irl.cfg import Config
from irl.cli.common import (
    resolve_default_method_for_entrypoint,
    resolve_total_steps,
    validate_policy_mode,
)


def test_cli_helpers():
    assert (
        resolve_default_method_for_entrypoint(
            config_provided=False,
            method=None,
            default_no_config="vanilla",
        )
        == "vanilla"
    )
    assert (
        resolve_default_method_for_entrypoint(
            config_provided=True,
            method=None,
            default_no_config="vanilla",
        )
        is None
    )

    assert validate_policy_mode(" Mode ", allowed=("mode", "sample")) == "mode"
    with pytest.raises(typer.BadParameter, match=r"--policy must be one of: mode, sample"):
        validate_policy_mode("nope", allowed=("mode", "sample"))

    base = Config()
    cfg = replace(base, exp=replace(base.exp, total_steps=123))

    assert resolve_total_steps(cfg, 999, prefer_cfg=True) == 123
    assert resolve_total_steps(cfg, 999, prefer_cfg=False) == 999
    assert resolve_total_steps(cfg, None, prefer_cfg=False) == 123
