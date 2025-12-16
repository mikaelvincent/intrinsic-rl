import pytest

from irl.cfg import ConfigError, loads_config


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
