from __future__ import annotations

import torch
from torch import nn

from irl.algo.advantage import compute_gae


class _DummyValueMidTrunc(nn.Module):
    def __init__(self, v_t, v_tp1):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))
        self.v_t = torch.as_tensor(v_t, dtype=torch.float32).view(-1)
        self.v_tp1 = torch.as_tensor(v_tp1, dtype=torch.float32).view(-1)
        self._calls = 0

    def forward(self, _obs):
        self._calls += 1
        return self.v_t if self._calls == 1 else self.v_tp1


def test_gae_no_leak_across_truncation_boundary_mid_rollout():
    obs = torch.zeros((4, 1, 1), dtype=torch.float32)
    next_obs = torch.zeros((4, 1, 1), dtype=torch.float32)

    rewards = torch.tensor([[0.0], [0.0], [10.0], [0.0]], dtype=torch.float32)
    terminals = torch.tensor([[0.0], [0.0], [0.0], [1.0]], dtype=torch.float32)
    truncations = torch.tensor([[0.0], [1.0], [0.0], [0.0]], dtype=torch.float32)

    vf = _DummyValueMidTrunc(
        v_t=[0.0, 0.0, 0.0, 0.0],
        v_tp1=[0.0, 5.0, 0.0, 0.0],
    )

    adv, v_targets = compute_gae(
        {
            "obs": obs,
            "next_observations": next_obs,
            "rewards": rewards,
            "terminals": terminals,
            "truncations": truncations,
        },
        value_fn=vf,
        gamma=1.0,
        lam=1.0,
        bootstrap_on_timeouts=True,
    )

    expected = torch.tensor([5.0, 5.0, 10.0, 0.0], dtype=torch.float32)
    assert torch.allclose(adv, expected, atol=1e-6)
    assert torch.allclose(v_targets, expected, atol=1e-6)
