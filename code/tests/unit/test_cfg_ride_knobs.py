import pytest

from irl.cfg import loads_config, ConfigError, to_dict


def test_ride_knobs_roundtrip():
    yaml_text = """
    seed: 3
    device: "cpu"
    method: "ride"
    env:
      id: "MountainCar-v0"
      vec_envs: 4
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
      eta: 0.2
      alpha_impact: 1.7
      bin_size: 0.5
      r_clip: 5.0
    adaptation:
      enabled: false
    evaluation:
      interval_steps: 10000
      episodes: 3
    logging:
      tb: false
      csv_interval: 100
      checkpoint_interval: 1000
    """
    cfg = loads_config(yaml_text)
    d = to_dict(cfg)
    assert d["method"] == "ride"
    assert abs(d["intrinsic"]["alpha_impact"] - 1.7) < 1e-8
    assert abs(d["intrinsic"]["bin_size"] - 0.5) < 1e-8


def test_ride_knobs_validation_fails_on_nonpositive_bin_size():
    yaml_bad = """
    method: "ride"
    intrinsic:
      eta: 0.2
      alpha_impact: 1.0
      bin_size: 0.0
    """
    with pytest.raises(ConfigError):
        loads_config(yaml_bad)
