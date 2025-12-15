import pytest

from irl.cfg import ConfigError, loads_config


def test_proposed_allows_alpha_impact_zero():
    yaml_text = """
method: proposed
intrinsic:
  eta: 0.1
  alpha_impact: 0.0
"""
    cfg = loads_config(yaml_text)
    assert cfg.method == "proposed"
    assert cfg.intrinsic.alpha_impact == 0.0


def test_ride_rejects_alpha_impact_nonpositive():
    yaml_bad = """
method: ride
intrinsic:
  eta: 0.1
  alpha_impact: 0.0
  bin_size: 0.25
"""
    with pytest.raises(ConfigError):
        loads_config(yaml_bad)
