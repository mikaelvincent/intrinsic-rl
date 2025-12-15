import gymnasium as gym
import numpy as np

from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.ride import RIDE


def test_ride_binning_repeated_state_reduces_reward():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)

    ride = RIDE(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=ICMConfig(phi_dim=16, hidden=(32, 32)),
        bin_size=0.5,
        alpha_impact=1.0,
    )

    o = np.zeros((1, 3), dtype=np.float32)
    op = np.ones((1, 3), dtype=np.float32)

    r1 = ride.compute_impact_binned(o, op, dones=np.array([False], dtype=bool)).item()
    assert np.isfinite(r1) and r1 > 0.0

    r2 = ride.compute_impact_binned(o, op, dones=np.array([False], dtype=bool)).item()
    assert np.isfinite(r2) and r2 < r1
    assert abs((r2 * 2.0) - r1) < 1e-4


def test_ride_binning_resets_on_done():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    ride = RIDE(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=ICMConfig(phi_dim=16, hidden=(32, 32)),
        bin_size=0.25,
        alpha_impact=1.0,
    )

    o = np.zeros((1, 4), dtype=np.float32)
    op = np.ones((1, 4), dtype=np.float32)

    r1 = ride.compute_impact_binned(o, op, dones=np.array([False], dtype=bool)).item()
    r2 = ride.compute_impact_binned(o, op, dones=np.array([False], dtype=bool)).item()
    assert r2 < r1

    r3 = ride.compute_impact_binned(o, op, dones=np.array([True], dtype=bool)).item()
    assert abs(r3 - r1) < 1e-4
