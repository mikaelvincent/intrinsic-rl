import gymnasium as gym
import numpy as np
import torch

from irl.intrinsic import IntrinsicOutput
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

    cfg_icm = ICMConfig(phi_dim=16, hidden=(32, 32))
    mod_step = Proposed(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=cfg_icm,
        region_capacity=4,
        depth_max=4,
        gate_tau_lp_mult=0.5,
        gate_tau_s=0.5,
        gate_hysteresis_up_mult=1.1,
        gate_min_consec_to_gate=1,
        gate_min_regions_for_gating=1,
        normalize_inside=True,
        gating_enabled=True,
    )
    mod_batch = Proposed(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=cfg_icm,
        region_capacity=4,
        depth_max=4,
        gate_tau_lp_mult=0.5,
        gate_tau_s=0.5,
        gate_hysteresis_up_mult=1.1,
        gate_min_consec_to_gate=1,
        gate_min_regions_for_gating=1,
        normalize_inside=True,
        gating_enabled=True,
    )
    mod_batch.load_state_dict(mod_step.state_dict())

    B = 64
    obs = rng.standard_normal((B, 4)).astype(np.float32)
    next_obs = rng.standard_normal((B, 4)).astype(np.float32)
    actions = rng.integers(0, 3, size=(B,), endpoint=False, dtype=np.int64)

    step_vals = []
    for i in range(B):
        tr = _Transition(obs[i], int(actions[i]), 0.0, next_obs[i])
        out = mod_step.compute(tr)
        assert isinstance(out, IntrinsicOutput)
        step_vals.append(float(out.r_int))
    step_vals = np.asarray(step_vals, dtype=np.float32)

    with torch.no_grad():
        batch_vals = mod_batch.compute_batch(obs, next_obs, actions, reduction="none")
    assert batch_vals.shape == (B,)
    batch_vals_np = batch_vals.cpu().numpy().astype(np.float32)

    assert np.allclose(step_vals, batch_vals_np, atol=1e-5)

    stats_step = mod_step._stats
    stats_batch = mod_batch._stats
    assert set(stats_step.keys()) == set(stats_batch.keys())
    for rid in stats_step.keys():
        a = stats_step[rid]
        b = stats_batch[rid]
        assert a.count == b.count
        assert a.gate == b.gate
        assert abs(a.ema_long - b.ema_long) < 1e-6
        assert abs(a.ema_short - b.ema_short) < 1e-6
        assert a.bad_consec == b.bad_consec
        assert a.good_consec == b.good_consec

    assert abs(mod_step.gate_rate - mod_batch.gate_rate) < 1e-6
    assert abs(mod_step.impact_rms - mod_batch.impact_rms) < 1e-6
    assert abs(mod_step.lp_rms - mod_batch.lp_rms) < 1e-6


def test_riac_compute_batch_matches_sequential_compute():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    obs_space, act_space = _make_vector_spaces(obs_dim=4, n_actions=3)

    cfg_icm = ICMConfig(phi_dim=16, hidden=(32, 32))
    mod_step = RIAC(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=cfg_icm,
        region_capacity=4,
        depth_max=4,
        ema_beta_long=0.995,
        ema_beta_short=0.90,
        alpha_lp=0.5,
    )
    mod_batch = RIAC(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=cfg_icm,
        region_capacity=4,
        depth_max=4,
        ema_beta_long=0.995,
        ema_beta_short=0.90,
        alpha_lp=0.5,
    )
    mod_batch.load_state_dict(mod_step.state_dict())

    B = 64
    obs = rng.standard_normal((B, 4)).astype(np.float32)
    next_obs = rng.standard_normal((B, 4)).astype(np.float32)
    actions = rng.integers(0, 3, size=(B,), endpoint=False, dtype=np.int64)

    step_vals = []
    for i in range(B):
        tr = _Transition(obs[i], int(actions[i]), 0.0, next_obs[i])
        out = mod_step.compute(tr)
        assert isinstance(out, IntrinsicOutput)
        step_vals.append(float(out.r_int))
    step_vals = np.asarray(step_vals, dtype=np.float32)

    with torch.no_grad():
        batch_vals = mod_batch.compute_batch(obs, next_obs, actions, reduction="none")
    assert batch_vals.shape == (B,)
    batch_vals_np = batch_vals.cpu().numpy().astype(np.float32)

    assert np.allclose(step_vals, batch_vals_np, atol=1e-5)

    stats_step = mod_step._stats
    stats_batch = mod_batch._stats
    assert set(stats_step.keys()) == set(stats_batch.keys())
    for rid in stats_step.keys():
        a = stats_step[rid]
        b = stats_batch[rid]
        assert a.count == b.count
        assert abs(a.ema_long - b.ema_long) < 1e-6
        assert abs(a.ema_short - b.ema_short) < 1e-6

    assert abs(mod_step.lp_rms - mod_batch.lp_rms) < 1e-6
