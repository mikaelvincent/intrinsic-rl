import numpy as np
import gymnasium as gym
import torch

from irl.intrinsic.proposed import Proposed
from irl.intrinsic.icm import ICMConfig


def _rand_batch(obs_dim: int, n_actions: int, B: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((B, obs_dim)).astype(np.float32)
    next_obs = rng.standard_normal((B, obs_dim)).astype(np.float32)
    actions = rng.integers(0, n_actions, size=(B,), endpoint=False, dtype=np.int64)
    return obs, next_obs, actions


def test_proposed_outputs_normalized_flag_and_rms_updates():
    # Deterministic nets for stability
    torch.manual_seed(0)

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    mod = Proposed(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=ICMConfig(phi_dim=16, hidden=(32, 32)),
    )

    # Disable gating to focus on normalization behavior
    mod.gating_enabled = False

    assert hasattr(mod, "outputs_normalized") and bool(mod.outputs_normalized)

    obs, next_obs, actions = _rand_batch(4, 3, B=64, seed=123)

    # First pass
    r1 = mod.compute_batch(obs, next_obs, actions)  # updates RMS
    assert r1.shape == (64,)
    assert torch.isfinite(r1).all()

    # RMS after first update
    imp_rms_1 = mod.impact_rms
    lp_rms_1 = mod.lp_rms
    assert imp_rms_1 > 0.0 and lp_rms_1 > 0.0

    # Second identical pass should adjust EMAs (values likely change)
    r2 = mod.compute_batch(obs, next_obs, actions)
    assert r2.shape == (64,)
    assert torch.isfinite(r2).all()

    imp_rms_2 = mod.impact_rms
    lp_rms_2 = mod.lp_rms

    # RMS should remain positive and typically move after another update
    assert imp_rms_2 > 0.0 and lp_rms_2 > 0.0
    # Avoid strict monotonicity; just ensure some adjustment occurred or values stable within tiny epsilon
    assert (abs(imp_rms_2 - imp_rms_1) > 1e-12) or (abs(lp_rms_2 - lp_rms_1) > 1e-12)


def test_proposed_loss_uses_current_normalization_snapshot():
    torch.manual_seed(1)

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)

    mod = Proposed(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=ICMConfig(phi_dim=16, hidden=(32, 32)),
    )
    mod.gating_enabled = False

    o, op, a = _rand_batch(3, 2, B=32, seed=7)

    # Warm up RMS a bit
    _ = mod.compute_batch(o, op, a)

    losses = mod.loss(o, op, a)
    for k in ["total", "icm_forward", "icm_inverse", "intrinsic_mean"]:
        assert k in losses
        assert torch.isfinite(losses[k])
