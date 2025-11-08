import random

import numpy as np
import torch

from irl.utils.determinism import seed_everything


def test_seed_everything_produces_repeatable_streams():
    # First run
    seed_everything(123, deterministic=True)
    py_vals_1 = [random.random() for _ in range(3)]
    np_vals_1 = np.random.rand(3).tolist()
    torch_vals_1 = torch.rand(3).tolist()

    # Second run (re-seed to same)
    seed_everything(123, deterministic=True)
    py_vals_2 = [random.random() for _ in range(3)]
    np_vals_2 = np.random.rand(3).tolist()
    torch_vals_2 = torch.rand(3).tolist()

    assert py_vals_1 == py_vals_2
    assert np.allclose(np_vals_1, np_vals_2)
    assert np.allclose(torch_vals_1, torch_vals_2)
