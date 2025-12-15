import random

import numpy as np
import torch

from irl.utils.determinism import seed_everything


def test_seed_everything_produces_repeatable_streams():
    seed_everything(123, deterministic=True)
    py_1 = [random.random() for _ in range(3)]
    np_1 = np.random.rand(3).tolist()
    t_1 = torch.rand(3).tolist()

    seed_everything(123, deterministic=True)
    py_2 = [random.random() for _ in range(3)]
    np_2 = np.random.rand(3).tolist()
    t_2 = torch.rand(3).tolist()

    assert py_1 == py_2
    assert np.allclose(np_1, np_2)
    assert np.allclose(t_1, t_2)
