# code/tests/unit/test_config_parsing_numbers.py
from __future__ import annotations

import pytest

from irl.cfg import ConfigError, loads_config


def test_loads_config_parses_numeric_strings() -> None:
    cfg = loads_config(
        """
method: vanilla
env:
  vec_envs: "8"
ppo:
  steps_per_update: "2_048"
  minibatches: "32"
  learning_rate: "3e-4"
"""
    )
    assert int(cfg.env.vec_envs) == 8
    assert int(cfg.ppo.steps_per_update) == 2048
    assert int(cfg.ppo.minibatches) == 32
    assert abs(float(cfg.ppo.learning_rate) - 3e-4) < 1e-12


def test_loads_config_parses_underscored_int_literals() -> None:
    cfg = loads_config(
        """
method: vanilla
ppo:
  steps_per_update: 2_048
  minibatches: 3_2
"""
    )
    assert int(cfg.ppo.steps_per_update) == 2048
    assert int(cfg.ppo.minibatches) == 32


def test_loads_config_rejects_bool_for_int_with_path() -> None:
    with pytest.raises(ConfigError) as ei:
        loads_config(
            """
seed: true
"""
        )
    assert "config.seed" in str(ei.value)
