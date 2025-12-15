import numpy as np
import torch

from irl.utils.image import ImagePreprocessConfig, preprocess_image


def test_preprocess_scales_float_0_255_to_unit_range():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(4, 16, 16, 3), dtype=np.uint8).astype(np.float32)

    cfg = ImagePreprocessConfig(grayscale=False, scale_uint8=True, channels_first=True)
    t = preprocess_image(arr, cfg=cfg, device="cpu")

    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float32
    assert t.dim() == 4
    assert float(t.min()) >= -1e-6
    assert float(t.max()) <= 1.0 + 1e-6


def test_preprocess_preserves_unit_float_images_when_already_scaled():
    rng = np.random.default_rng(1)
    arr = rng.random(size=(2, 8, 8, 3), dtype=np.float32)

    cfg = ImagePreprocessConfig(grayscale=False, scale_uint8=True, channels_first=False)
    t = preprocess_image(arr, cfg=cfg, device="cpu")

    out = t.cpu().numpy()
    assert out.shape == arr.shape
    assert np.allclose(out, arr, atol=1e-6)
