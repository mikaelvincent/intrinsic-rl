import pytest

from irl.cfg import loads_config, to_dict, ConfigError


def test_loads_config_roundtrip():
    yaml_text = """
    seed: 7
    device: "cpu"
    method: "vanilla"
    env:
      id: "MountainCar-v0"
      vec_envs: 8
      frame_skip: 1
      domain_randomization: false
      discrete_actions: true
    ppo:
      steps_per_update: 128
      minibatches: 32
      epochs: 2
      learning_rate: 3.0e-4
      gamma: 0.99
      gae_lambda: 0.95
      clip_range: 0.2
      entropy_coef: 0.01
    intrinsic:
      eta: 0.0
    adaptation:
      enabled: false
    evaluation:
      interval_steps: 50000
      episodes: 5
    logging:
      tb: false
      csv_interval: 1000
      checkpoint_interval: 10000
    """
    cfg = loads_config(yaml_text)
    d = to_dict(cfg)
    assert d["seed"] == 7
    assert d["method"] == "vanilla"
    assert d["env"]["id"] == "MountainCar-v0"
    assert d["ppo"]["minibatches"] == 32


def test_validation_divisibility_error():
    # Violates minibatch divisibility in both interpretations
    yaml_bad = """
    seed: 1
    device: "cpu"
    method: "vanilla"
    env:
      id: "MountainCar-v0"
      vec_envs: 8
    ppo:
      steps_per_update: 130
      minibatches: 64
      epochs: 1
      learning_rate: 3.0e-4
      gamma: 0.99
      gae_lambda: 0.95
      clip_range: 0.2
    intrinsic:
      eta: 0.0
    adaptation:
      enabled: false
    evaluation:
      interval_steps: 1000
      episodes: 1
    logging:
      tb: false
      csv_interval: 100
      checkpoint_interval: 1000
    """
    with pytest.raises(ConfigError):
        loads_config(yaml_bad)


def test_unknown_field_detection():
    yaml_unknown = """
    seed: 1
    device: "cpu"
    unknown_top_level: 123
    """
    with pytest.raises(ConfigError):
        loads_config(yaml_unknown)
