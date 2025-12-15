import gymnasium as gym
import numpy as np
import pytest

from irl.cfg import ConfigError, loads_config
from irl.intrinsic.factory import create_intrinsic_module
from irl.intrinsic.proposed import Proposed


def test_cfg_accepts_proposed_lp_only():
    yaml_text = """
method: proposed_lp_only
intrinsic:
  eta: 0.1
  alpha_impact: 0.0
"""
    cfg = loads_config(yaml_text)
    assert str(cfg.method).lower() == "proposed_lp_only"
    assert cfg.intrinsic.alpha_impact == 0.0
    assert cfg.intrinsic.alpha_lp > 0.0


def test_cfg_rejects_proposed_lp_only_nonzero_alpha_impact():
    yaml_bad = """
method: proposed_lp_only
intrinsic:
  eta: 0.1
  alpha_impact: 0.25
"""
    with pytest.raises(ConfigError):
        loads_config(yaml_bad)


def test_cfg_accepts_proposed_impact_only():
    yaml_text = """
method: proposed_impact_only
intrinsic:
  eta: 0.1
  alpha_lp: 0.0
"""
    cfg = loads_config(yaml_text)
    assert str(cfg.method).lower() == "proposed_impact_only"
    assert cfg.intrinsic.alpha_lp == 0.0
    assert cfg.intrinsic.alpha_impact > 0.0


def test_cfg_rejects_proposed_impact_only_nonzero_alpha_lp():
    yaml_bad = """
method: proposed_impact_only
intrinsic:
  eta: 0.1
  alpha_lp: 0.5
"""
    with pytest.raises(ConfigError):
        loads_config(yaml_bad)


def test_cfg_accepts_proposed_nogate():
    yaml_text = """
method: proposed_nogate
intrinsic:
  eta: 0.1
  gate:
    enabled: false
"""
    cfg = loads_config(yaml_text)
    assert str(cfg.method).lower() == "proposed_nogate"
    assert bool(cfg.intrinsic.gate.enabled) is False


def test_cfg_rejects_proposed_nogate_when_gate_enabled():
    yaml_bad = """
method: proposed_nogate
intrinsic:
  eta: 0.1
  gate:
    enabled: true
"""
    with pytest.raises(ConfigError):
        loads_config(yaml_bad)


def test_factory_builds_distinct_proposed_variants():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    lp_only = create_intrinsic_module(
        "proposed_lp_only",
        obs_space,
        act_space,
        device="cpu",
        alpha_impact=0.0,
        alpha_lp=0.5,
        gating_enabled=True,
    )
    assert isinstance(lp_only, Proposed)
    assert float(lp_only.alpha_impact) == 0.0

    impact_only = create_intrinsic_module(
        "proposed_impact_only",
        obs_space,
        act_space,
        device="cpu",
        alpha_impact=1.0,
        alpha_lp=0.0,
        gating_enabled=True,
    )
    assert isinstance(impact_only, Proposed)
    assert float(impact_only.alpha_lp) == 0.0

    nogate = create_intrinsic_module(
        "proposed_nogate",
        obs_space,
        act_space,
        device="cpu",
        alpha_impact=1.0,
        alpha_lp=0.5,
        gating_enabled=True,
    )
    assert isinstance(nogate, Proposed)
    assert bool(getattr(nogate, "gating_enabled", True)) is False
