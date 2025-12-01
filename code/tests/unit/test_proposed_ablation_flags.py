import numpy as np
import torch
import gymnasium as gym

from irl.intrinsic.proposed import Proposed
from irl.intrinsic.icm import ICMConfig


def _rand_batch(obs_dim: int, n_actions: int, B: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((B, obs_dim)).astype(np.float32)
    next_obs = rng.standard_normal((B, obs_dim)).astype(np.float32)
    actions = rng.integers(0, n_actions, size=(B,), endpoint=False, dtype=np.int64)
    return obs, next_obs, actions


def test_proposed_normalize_inside_false_emits_raw_and_differs():
    torch.manual_seed(0)

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    # Build a "raw" instance and a "normalized" instance with identical weights
    mod_raw = Proposed(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=ICMConfig(phi_dim=16, hidden=(32, 32)),
        normalize_inside=False,
        gating_enabled=False,
    )
    mod_norm = Proposed(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=ICMConfig(phi_dim=16, hidden=(32, 32)),
        normalize_inside=True,
        gating_enabled=False,
    )
    # Share identical ICM/backbone weights for a fair comparison
    mod_norm.load_state_dict(mod_raw.state_dict())

    assert hasattr(mod_raw, "outputs_normalized") and not bool(mod_raw.outputs_normalized)
    assert hasattr(mod_norm, "outputs_normalized") and bool(mod_norm.outputs_normalized)

    o, op, a = _rand_batch(4, 3, B=64, seed=123)

    with torch.no_grad():
        r_raw = mod_raw.compute_batch(o, op, a)  # raw components (no internal RMS)
        r_norm = mod_norm.compute_batch(o, op, a)  # internally normalized

    assert r_raw.shape == r_norm.shape == (64,)
    # Expect a measurable difference between raw and normalized paths
    assert not torch.allclose(r_raw, r_norm, atol=1e-6)


def test_proposed_gating_disabled_never_gates():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    mod = Proposed(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=ICMConfig(phi_dim=16, hidden=(32, 32)),
        gating_enabled=False,  # explicitly disable gating
    )

    # Drive a few updates to populate regions/stats without enabling gating
    rng = np.random.default_rng(7)
    for _ in range(5):
        o = rng.standard_normal((32, 5)).astype(np.float32)
        op = rng.standard_normal((32, 5)).astype(np.float32)
        a = rng.integers(0, 3, size=(32,), endpoint=False, dtype=np.int64)
        _ = mod.compute_batch(o, op, a)

    # With gating disabled, no region should be marked off
    assert mod.gate_rate == 0.0
