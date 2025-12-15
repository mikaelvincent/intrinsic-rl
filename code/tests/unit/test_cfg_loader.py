import pytest

from irl.cfg import ConfigError, loads_config, to_dict


def test_loads_config_roundtrip():
    yaml_text = """
seed: 7
device: cpu
method: vanilla
env:
  id: MountainCar-v0
  vec_envs: 8
ppo:
  steps_per_update: 128
  minibatches: 32
logging:
  tb: false
"""
    cfg = loads_config(yaml_text)
    d = to_dict(cfg)
    assert d["seed"] == 7
    assert d["method"] == "vanilla"
    assert d["env"]["id"] == "MountainCar-v0"
    assert d["ppo"]["minibatches"] == 32


def test_validation_divisibility_error():
    yaml_bad = """
method: vanilla
env:
  vec_envs: 8
ppo:
  steps_per_update: 130
  minibatches: 64
"""
    with pytest.raises(ConfigError):
        loads_config(yaml_bad)


def test_unknown_field_detection():
    yaml_unknown = """
seed: 1
device: cpu
unknown_top_level: 123
"""
    with pytest.raises(ConfigError):
        loads_config(yaml_unknown)
