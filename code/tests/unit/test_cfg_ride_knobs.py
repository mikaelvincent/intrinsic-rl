import pytest

from irl.cfg import ConfigError, loads_config, to_dict


def test_ride_knobs_roundtrip():
    yaml_text = """
seed: 3
device: cpu
method: ride
env:
  id: MountainCar-v0
  vec_envs: 4
ppo:
  steps_per_update: 128
  minibatches: 32
intrinsic:
  eta: 0.2
  alpha_impact: 1.7
  bin_size: 0.5
logging:
  tb: false
"""
    cfg = loads_config(yaml_text)
    d = to_dict(cfg)
    assert d["method"] == "ride"
    assert abs(d["intrinsic"]["alpha_impact"] - 1.7) < 1e-8
    assert abs(d["intrinsic"]["bin_size"] - 0.5) < 1e-8


def test_ride_knobs_validation_fails_on_nonpositive_bin_size():
    yaml_bad = """
method: ride
intrinsic:
  eta: 0.2
  alpha_impact: 1.0
  bin_size: 0.0
"""
    with pytest.raises(ConfigError):
        loads_config(yaml_bad)
