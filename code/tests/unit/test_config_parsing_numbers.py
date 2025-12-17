from __future__ import annotations

import pytest

from irl.cfg import ConfigError, loads_config


def test_loads_config_parses_underscore_ints() -> None:
    cfg = loads_config(
        """
logging:
  csv_interval: 10_000
  checkpoint_interval: 100_000
"""
    )
    assert int(cfg.logging.csv_interval) == 10_000
    assert int(cfg.logging.checkpoint_interval) == 100_000


def test_loads_config_parses_scientific_notation_strings() -> None:
    cfg = loads_config(
        """
logging:
  checkpoint_interval: "1e5"
adaptation:
  interval_steps: "5e4"
ppo:
  learning_rate: "3e-4"
"""
    )
    assert int(cfg.logging.checkpoint_interval) == 100_000
    assert int(cfg.adaptation.interval_steps) == 50_000
    assert abs(float(cfg.ppo.learning_rate) - 3e-4) < 1e-12


def test_int_type_error_includes_field_path() -> None:
    with pytest.raises(ConfigError) as excinfo:
        _ = loads_config(
            """
seed: true
"""
        )
    msg = str(excinfo.value)
    assert "config.seed" in msg
    assert "{path}" not in msg
