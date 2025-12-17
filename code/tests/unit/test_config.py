from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from irl.cfg import ConfigError, loads_config
from irl.intrinsic.factory import create_intrinsic_module
from irl.intrinsic.glpe import GLPE
from irl.utils.checkpoint import compute_cfg_hash


def test_compute_cfg_hash_is_stable_and_selective():
    base = {
        "method": "vanilla",
        "seed": 7,
        "env": {"id": "MountainCar-v0", "vec_envs": 8},
        "ppo": {"minibatches": 32, "steps_per_update": 128},
        "exp": {"deterministic": True},
    }
    reordered = {
        "ppo": {"steps_per_update": 128, "minibatches": 32},
        "env": {"vec_envs": 8, "id": "MountainCar-v0"},
        "seed": 7,
        "method": "vanilla",
        "exp": {"deterministic": True},
    }
    assert compute_cfg_hash(base) == compute_cfg_hash(reordered)

    changed = {
        "method": "vanilla",
        "seed": 7,
        "env": {"id": "MountainCar-v0", "vec_envs": 8},
        "ppo": {"minibatches": 64, "steps_per_update": 128},
        "exp": {"deterministic": True},
    }
    assert compute_cfg_hash(changed) != compute_cfg_hash(base)

    with_profile_flag = {
        "method": "vanilla",
        "seed": 7,
        "env": {"id": "MountainCar-v0", "vec_envs": 8},
        "ppo": {"minibatches": 32, "steps_per_update": 128},
        "exp": {"deterministic": True, "profile_cuda_sync": True},
    }
    assert compute_cfg_hash(with_profile_flag) == compute_cfg_hash(base)


def test_loads_config_rejects_unknown_fields():
    with pytest.raises(ConfigError):
        loads_config(
            """
seed: 1
unknown_top_level: 123
"""
        )


def test_loads_config_rejects_minibatch_divisibility():
    with pytest.raises(ConfigError):
        loads_config(
            """
method: vanilla
env:
  vec_envs: 8
ppo:
  steps_per_update: 130
  minibatches: 64
"""
        )


def test_loads_config_accepts_glpe_variants():
    cases = [
        (
            """
method: glpe_lp_only
intrinsic:
  eta: 0.1
  alpha_impact: 0.0
""",
            "glpe_lp_only",
        ),
        (
            """
method: glpe_nogate
intrinsic:
  eta: 0.1
  gate:
    enabled: false
""",
            "glpe_nogate",
        ),
    ]

    for yaml_text, expected_method in cases:
        cfg = loads_config(yaml_text)
        assert str(cfg.method).lower() == expected_method


def test_loads_config_rejects_invalid_glpe_variant():
    with pytest.raises(ConfigError):
        loads_config(
            """
method: glpe_lp_only
intrinsic:
  eta: 0.1
  alpha_impact: 0.25
"""
        )


def test_loads_config_rejects_invalid_ride_knob():
    with pytest.raises(ConfigError):
        loads_config(
            """
method: ride
intrinsic:
  eta: 0.1
  alpha_impact: 0.0
  bin_size: 0.25
"""
        )


def test_factory_applies_glpe_variant_overrides():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    cases = [
        ("glpe_lp_only", 0.0, 0.5, True),
        ("glpe_nogate", 1.0, 0.5, False),
    ]

    for method, expected_impact, expected_lp, expected_gate in cases:
        mod = create_intrinsic_module(
            method,
            obs_space,
            act_space,
            device="cpu",
            alpha_impact=1.0,
            alpha_lp=0.5,
            gating_enabled=True,
        )
        assert isinstance(mod, GLPE)
        assert float(mod.alpha_impact) == expected_impact
        assert float(mod.alpha_lp) == expected_lp
        assert bool(getattr(mod, "gating_enabled", True)) is expected_gate
