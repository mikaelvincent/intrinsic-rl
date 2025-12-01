import pytest

from irl.cfg import loads_config, ConfigError


def test_proposed_allows_alpha_impact_zero():
    # Minimal YAML relying on defaults for divisibility and other fields
    yaml_text = """
    method: "proposed"
    intrinsic:
      eta: 0.1
      alpha_impact: 0.0
    """
    cfg = loads_config(yaml_text)
    # If we got here without exception, validation accepted alpha_impact == 0 for proposed
    assert cfg.method == "proposed"
    assert abs(cfg.intrinsic.alpha_impact - 0.0) < 1e-12


def test_ride_rejects_alpha_impact_nonpositive():
    yaml_bad = """
    method: "ride"
    intrinsic:
      eta: 0.1
      alpha_impact: 0.0
      bin_size: 0.25
    """
    with pytest.raises(ConfigError):
        loads_config(yaml_bad)
