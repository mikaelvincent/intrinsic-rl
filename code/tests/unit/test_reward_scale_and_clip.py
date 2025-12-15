import numpy as np
import torch
from torch import nn

from irl.algo.advantage import compute_gae
from irl.intrinsic.normalization import RunningRMS


class ZeroValue(nn.Module):
    def __init__(self):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))

    def forward(self, obs):
        t = obs if torch.is_tensor(obs) else torch.as_tensor(obs)
        n = t.reshape(-1, t.shape[-1]).shape[0]
        return torch.zeros(n, dtype=torch.float32, device=t.device)


def test_extrinsic_rewards_not_clipped_in_gae():
    obs = torch.zeros((1, 1, 3), dtype=torch.float32)
    rewards = torch.tensor([[10.0]], dtype=torch.float32)
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
    eta = 0.1
    r_clip = 5.0
    raw = np.array([100.0, -50.0, 10.0, 0.5], dtype=np.float32)

    expected_A = eta * np.clip(raw, -r_clip, r_clip)

    rms = RunningRMS(beta=0.99, eps=1e-8)
    rms.update(raw)
    norm = rms.normalize(raw)
    expected_B = eta * np.clip(norm, -r_clip, r_clip)

    assert np.isfinite(expected_A).all() and np.isfinite(expected_B).all()
    assert np.max(np.abs(expected_A)) <= eta * r_clip + 1e-8
    assert np.max(np.abs(expected_B)) <= eta * r_clip + 1e-8


class _DummyValueForTimeouts(nn.Module):
    def __init__(self, v_t, v_tp1):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))
        self.v_t = torch.as_tensor(v_t, dtype=torch.float32).view(-1)
        self.v_tp1 = torch.as_tensor(v_tp1, dtype=torch.float32).view(-1)
        self._calls = 0

    def forward(self, obs):
        self._calls += 1
        return self.v_t if self._calls == 1 else self.v_tp1


def test_gae_bootstraps_on_timeouts_when_masks_provided():
    obs = torch.zeros((3, 1, 1), dtype=torch.float32)
    next_obs = torch.zeros((3, 1, 1), dtype=torch.float32)

    rewards = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32)
    v_t = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    v_tp1 = torch.tensor([20.0, 30.0, 40.0], dtype=torch.float32)

    dones = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float32)
    vf_term = _DummyValueForTimeouts(v_t, v_tp1)
    adv_term, vt_term = compute_gae(
        {"obs": obs, "next_observations": next_obs, "rewards": rewards, "dones": dones},
        value_fn=vf_term,
        gamma=1.0,
        lam=1.0,
        bootstrap_on_timeouts=False,
    )

    expected_adv_term = torch.tensor([-7.0, -18.0, -29.0], dtype=torch.float32)
    expected_vt_term = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
    assert adv_term.shape == (3,)
    assert vt_term.shape == (3,)
    assert torch.allclose(adv_term, expected_adv_term, atol=1e-6)
    assert torch.allclose(vt_term, expected_vt_term, atol=1e-6)

    terminals = torch.tensor([[0.0], [0.0], [0.0]], dtype=torch.float32)
    truncs = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float32)
    vf_boot = _DummyValueForTimeouts(v_t, v_tp1)
    adv_boot, vt_boot = compute_gae(
        {
            "obs": obs,
            "next_observations": next_obs,
            "rewards": rewards,
            "terminals": terminals,
            "truncations": truncs,
        },
        value_fn=vf_boot,
        gamma=1.0,
        lam=1.0,
        bootstrap_on_timeouts=True,
    )

    expected_adv_boot = torch.tensor([33.0, 22.0, 11.0], dtype=torch.float32)
    expected_vt_boot = torch.tensor([43.0, 42.0, 41.0], dtype=torch.float32)
    assert adv_boot.shape == (3,)
    assert vt_boot.shape == (3,)
    assert torch.allclose(adv_boot, expected_adv_boot, atol=1e-6)
    assert torch.allclose(vt_boot, expected_vt_boot, atol=1e-6)

    assert not torch.allclose(adv_boot, adv_term)
    assert not torch.allclose(vt_boot, vt_term)
