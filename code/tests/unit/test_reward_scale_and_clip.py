import torch
from torch import nn

from irl.algo.advantage import compute_gae


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

    assert torch.allclose(adv, torch.tensor([10.0], dtype=torch.float32), atol=1e-6)
    assert torch.allclose(v_targets, torch.tensor([10.0], dtype=torch.float32), atol=1e-6)


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

    assert torch.allclose(
        adv_term, torch.tensor([-7.0, -18.0, -29.0], dtype=torch.float32), atol=1e-6
    )
    assert torch.allclose(vt_term, torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32), atol=1e-6)

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

    assert torch.allclose(
        adv_boot, torch.tensor([33.0, 22.0, 11.0], dtype=torch.float32), atol=1e-6
    )
    assert torch.allclose(
        vt_boot, torch.tensor([43.0, 42.0, 41.0], dtype=torch.float32), atol=1e-6
    )
