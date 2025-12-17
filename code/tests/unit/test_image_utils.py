from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from irl.utils.image import ImagePreprocessConfig, preprocess_image
from irl.utils.images import infer_channels_hw
from irl.utils.spaces import is_image_space


def test_image_detection_and_preprocess_scaling():
    assert not is_image_space(gym.spaces.Discrete(3))
    assert not is_image_space(gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32))
    assert is_image_space(gym.spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8))

    assert infer_channels_hw((32, 32, 3)) == (3, (32, 32))
    assert infer_channels_hw((3, 32, 32)) == (3, (32, 32))

    rng = np.random.default_rng(0)
    arr_255 = rng.integers(0, 256, size=(2, 8, 8, 3), dtype=np.uint8).astype(np.float32)

    cfg_cf = ImagePreprocessConfig(grayscale=False, scale_uint8=True, channels_first=True)
    t = preprocess_image(arr_255, cfg=cfg_cf, device="cpu")
    assert isinstance(t, torch.Tensor)
    assert t.shape == (2, 3, 8, 8)
    assert float(t.min()) >= -1e-6
    assert float(t.max()) <= 1.0 + 1e-6

    unit = rng.random(size=(2, 8, 8, 3), dtype=np.float32)
    cfg_cl = ImagePreprocessConfig(grayscale=False, scale_uint8=True, channels_first=False)
    t2 = preprocess_image(unit, cfg=cfg_cl, device="cpu")
    out = t2.cpu().numpy()
    assert out.shape == unit.shape
    assert np.allclose(out, unit, atol=1e-6)
