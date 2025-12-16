import gymnasium as gym
import numpy as np
import torch

from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.proposed import Proposed
from irl.intrinsic.riac import RIAC


class _Transition:
    def __init__(self, s, a, r_ext, s_next):
        self.s = s
        self.a = a
        self.r_ext = r_ext
        self.s_next = s_next


def _make_vector_spaces(obs_dim: int = 4, n_actions: int = 3):
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Discrete(n_actions)
    return obs_space, act_space


def test_proposed_compute_batch_matches_sequential_compute():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    obs_space, act_space = _make_vector_spaces(obs_dim=4, n_actions=3)

    icm_cfg = ICMConfig(phi_dim=16, hidden=(32, 32))
    kwargs = dict(
        device="cpu",
        icm_cfg=icm_cfg,
        region_capacity=1000,
        depth_max=0,
        gate_tau_lp_mult=0.5,
        gate_tau_s=0.5,
        gate_hysteresis_up_mult=1.1,
        gate_min_consec_to_gate=1,
        gate_min_regions_for_gating=1,
        normalize_inside=True,
        gating_enabled=True,
    )

    mod_step = Proposed(obs_space, act_space, **kwargs)
    mod_batch = Proposed(obs_space, act_space, **kwargs)
    mod_batch.load_state_dict(mod_step.state_dict())

    B = 64
    obs = rng.standard_normal((B, 4)).astype(np.float32)
    next_obs = rng.standard_normal((B, 4)).astype(np.float32)
    actions = rng.integers(0, 3, size=(B,), endpoint=False, dtype=np.int64)

    step_vals = np.array(
        [
            float(mod_step.compute(_Transition(obs[i], int(actions[i]), 0.0, next_obs[i])).r_int)
            for i in range(B)
        ],
        dtype=np.float32,
    )

    with torch.no_grad():
        batch_vals = (
            mod_batch.compute_batch(obs, next_obs, actions, reduction="none")
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    assert np.allclose(step_vals, batch_vals, atol=1e-5)
    assert abs(mod_step.gate_rate - mod_batch.gate_rate) < 1e-6
    assert abs(mod_step.impact_rms - mod_batch.impact_rms) < 1e-6
    assert abs(mod_step.lp_rms - mod_batch.lp_rms) < 1e-6


def test_riac_compute_batch_matches_sequential_compute():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    obs_space, act_space = _make_vector_spaces(obs_dim=4, n_actions=3)

    icm_cfg = ICMConfig(phi_dim=16, hidden=(32, 32))
    kwargs = dict(
        device="cpu",
        icm_cfg=icm_cfg,
        region_capacity=1000,
        depth_max=0,
        ema_beta_long=0.995,
        ema_beta_short=0.90,
        alpha_lp=0.5,
    )

    mod_step = RIAC(obs_space, act_space, **kwargs)
    mod_batch = RIAC(obs_space, act_space, **kwargs)
    mod_batch.load_state_dict(mod_step.state_dict())

    B = 64
    obs = rng.standard_normal((B, 4)).astype(np.float32)
    next_obs = rng.standard_normal((B, 4)).astype(np.float32)
    actions = rng.integers(0, 3, size=(B,), endpoint=False, dtype=np.int64)

    step_vals = np.array(
        [
            float(mod_step.compute(_Transition(obs[i], int(actions[i]), 0.0, next_obs[i])).r_int)
            for i in range(B)
        ],
        dtype=np.float32,
    )

    with torch.no_grad():
        batch_vals = (
            mod_batch.compute_batch(obs, next_obs, actions, reduction="none")
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    assert np.allclose(step_vals, batch_vals, atol=1e-5)
    assert abs(mod_step.lp_rms - mod_batch.lp_rms) < 1e-6
