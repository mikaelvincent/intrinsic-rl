from dataclasses import replace

import pytest
import typer

from irl.cli.common import (
    resolve_default_method_for_entrypoint,
    resolve_total_steps,
    validate_policy_mode,
)
from irl.cfg import Config


def test_resolve_default_method_for_entrypoint_matches_train_cli_default():
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
    assert (
        resolve_default_method_for_entrypoint(
            config_provided=False,
            method="glpe",
            default_no_config="vanilla",
        )
        == "glpe"
    )


def test_validate_policy_mode_normalizes_and_errors():
    assert validate_policy_mode(" Mode ", allowed=("mode", "sample")) == "mode"

    with pytest.raises(typer.BadParameter, match=r"--policy must be one of: mode, sample"):
        validate_policy_mode("nope", allowed=("mode", "sample"))

    with pytest.raises(typer.BadParameter, match=r"--policy must be one of: mode, sample, both"):
        validate_policy_mode("nope", allowed=("mode", "sample", "both"))


def test_resolve_total_steps_precedence():
    base = Config()
    cfg = replace(base, exp=replace(base.exp, total_steps=123))

    assert resolve_total_steps(cfg, 999, prefer_cfg=True) == 123
    assert resolve_total_steps(cfg, 999, prefer_cfg=False) == 999
    assert resolve_total_steps(cfg, None, prefer_cfg=False) == 123

    cfg2 = replace(base, exp=replace(base.exp, total_steps=None))
    assert resolve_total_steps(cfg2, None, default_total_steps=10_000, prefer_cfg=False) == 10_000
