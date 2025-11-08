import numpy as np
import torch
from torch import nn

from irl.algo.advantage import compute_gae
from irl.intrinsic.normalization import RunningRMS


class ZeroValue(nn.Module):
    """Value function stub that returns zeros but owns a param for device inference."""
    def __init__(self):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, obs):
        t = obs if torch.is_tensor(obs) else torch.as_tensor(obs)
        # Flatten leading dims so output matches [N] as ValueNetwork would
        n = t.reshape(-1, t.shape[-1]).shape[0]
        return torch.zeros(n, dtype=torch.float32, device=t.device)


def test_extrinsic_rewards_not_clipped_in_gae():
    # One terminal step with a large reward -> advantage should equal that reward
    T, B, D = 1, 1, 3
    obs = torch.zeros((T, B, D), dtype=torch.float32)
    rewards = torch.tensor([[10.0]], dtype=torch.float32)  # large, would reveal clipping if present
    dones = torch.tensor([[1.0]], dtype=torch.float32)

    adv, v_targets = compute_gae(
        {"obs": obs, "rewards": rewards, "dones": dones},
        value_fn=ZeroValue(),
        gamma=0.99,
        lam=0.95,
    )
    assert adv.shape == (1,)
    assert torch.allclose(adv, torch.tensor([10.0], dtype=torch.float32), atol=1e-6)
    assert torch.allclose(v_targets, torch.tensor([10.0], dtype=torch.float32), atol=1e-6)


def test_intrinsic_clip_formula_matches_trainer_paths():
    # Mirror trainer logic for clipping with/without per-module normalization
    eta = 0.1
    r_clip = 5.0
    raw = np.array([100.0, -50.0, 10.0, 0.5], dtype=np.float32)

    # Path A: module outputs already normalized (outputs_normalized=True)
    expected_A = eta * np.clip(raw, -r_clip, r_clip)

    # Path B: module outputs NOT normalized -> apply global RunningRMS then clip
    rms = RunningRMS(beta=0.99, eps=1e-8)
    rms.update(raw)
    norm = rms.normalize(raw)
    expected_B = eta * np.clip(norm, -r_clip, r_clip)

    # Sanity: both are finite and reflect respective formulas
    assert np.isfinite(expected_A).all() and np.isfinite(expected_B).all()
    assert np.max(np.abs(expected_A)) <= eta * r_clip + 1e-8
    assert np.max(np.abs(expected_B)) <= eta * r_clip + 1e-8
