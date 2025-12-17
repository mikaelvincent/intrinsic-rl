from __future__ import annotations

import gymnasium as gym
import numpy as np

from irl.utils.images import infer_channels_hw
from irl.utils.spaces import is_image_space


def test_is_image_space_and_channel_inference():
    assert not is_image_space(gym.spaces.Discrete(3))
    assert not is_image_space(gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32))
    assert is_image_space(gym.spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8))
    assert is_image_space(gym.spaces.Box(low=0, high=255, shape=(3, 32, 32), dtype=np.uint8))

    assert infer_channels_hw((32, 32, 3)) == (3, (32, 32))
    assert infer_channels_hw((3, 32, 32)) == (3, (32, 32))
    assert infer_channels_hw((32, 32, 4)) == (4, (32, 32))
