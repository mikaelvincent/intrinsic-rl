import gymnasium as gym
import numpy as np
import pytest

from irl.cfg import ConfigError, loads_config
from irl.intrinsic.factory import create_intrinsic_module
from irl.intrinsic.proposed import Proposed


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


@pytest.mark.parametrize(
    "yaml_text, expected_method",
    [
        (
            """
method: proposed
intrinsic:
  eta: 0.1
  alpha_impact: 0.0
""",
            "proposed",
        ),
        (
            """
method: proposed_lp_only
intrinsic:
  eta: 0.1
  alpha_impact: 0.0
""",
            "proposed_lp_only",
        ),
        (
            """
method: proposed_impact_only
intrinsic:
  eta: 0.1
  alpha_lp: 0.0
""",
            "proposed_impact_only",
        ),
        (
            """
method: proposed_nogate
intrinsic:
  eta: 0.1
  gate:
    enabled: false
""",
            "proposed_nogate",
        ),
    ],
)
def test_loads_config_accepts_proposed_variants(yaml_text: str, expected_method: str):
    cfg = loads_config(yaml_text)
    assert str(cfg.method).lower() == expected_method


@pytest.mark.parametrize(
    "yaml_text",
    [
        """
method: proposed_lp_only
intrinsic:
  eta: 0.1
  alpha_impact: 0.25
""",
        """
method: proposed_impact_only
intrinsic:
  eta: 0.1
  alpha_lp: 0.5
""",
        """
method: proposed_nogate
intrinsic:
  eta: 0.1
  gate:
    enabled: true
""",
    ],
)
def test_loads_config_rejects_invalid_proposed_variants(yaml_text: str):
    with pytest.raises(ConfigError):
        loads_config(yaml_text)


@pytest.mark.parametrize(
    "yaml_text",
    [
        """
method: ride
intrinsic:
  eta: 0.1
  alpha_impact: 0.0
  bin_size: 0.25
""",
        """
method: ride
intrinsic:
  eta: 0.1
  alpha_impact: 1.0
  bin_size: 0.0
""",
    ],
)
def test_loads_config_rejects_invalid_ride_knobs(yaml_text: str):
    with pytest.raises(ConfigError):
        loads_config(yaml_text)


@pytest.mark.parametrize(
    "method, expected_impact, expected_lp, expected_gate",
    [
        ("proposed_lp_only", 0.0, 0.5, True),
        ("proposed_impact_only", 1.0, 0.0, True),
        ("proposed_nogate", 1.0, 0.5, False),
    ],
)
def test_factory_applies_proposed_variant_overrides(
    method: str, expected_impact: float, expected_lp: float, expected_gate: bool
):
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    mod = create_intrinsic_module(
        method,
        obs_space,
        act_space,
        device="cpu",
        alpha_impact=1.0,
        alpha_lp=0.5,
        gating_enabled=True,
    )
    assert isinstance(mod, Proposed)
    assert float(mod.alpha_impact) == expected_impact
    assert float(mod.alpha_lp) == expected_lp
    assert bool(getattr(mod, "gating_enabled", True)) is expected_gate


def test_factory_passes_ride_knobs():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    act_space = gym.spaces.Discrete(5)

    ride = create_intrinsic_module(
        "ride",
        obs_space,
        act_space,
        device="cpu",
        bin_size=0.5,
        alpha_impact=1.7,
    )
    assert abs(float(ride.bin_size) - 0.5) < 1e-8
    assert abs(float(ride.alpha_impact) - 1.7) < 1e-8
