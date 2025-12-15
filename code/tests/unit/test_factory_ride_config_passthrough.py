import gymnasium as gym

from irl.intrinsic.factory import create_intrinsic_module


def test_ride_factory_wires_alpha_and_bin_size():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=float)
    act_space = gym.spaces.Discrete(5)
    ride = create_intrinsic_module(
        "ride",
        obs_space,
        act_space,
        device="cpu",
        bin_size=0.5,
        alpha_impact=1.7,
    )
    assert hasattr(ride, "bin_size")
    assert hasattr(ride, "alpha_impact")
    assert abs(float(ride.bin_size) - 0.5) < 1e-8
    assert abs(float(ride.alpha_impact) - 1.7) < 1e-8
