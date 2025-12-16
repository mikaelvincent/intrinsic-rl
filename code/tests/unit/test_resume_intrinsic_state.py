from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from torch.optim import Adam

from irl.intrinsic import RunningRMS
from irl.intrinsic.glpe import GLPE
from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.riac import RIAC
from irl.models.networks import PolicyNetwork, ValueNetwork
from irl.trainer.training_setup import _restore_from_checkpoint
from irl.utils.loggers import get_logger


def _rand_batch(rng: np.random.Generator, obs_dim: int, n_actions: int, B: int):
    obs = rng.standard_normal((B, obs_dim)).astype(np.float32)
    next_obs = rng.standard_normal((B, obs_dim)).astype(np.float32)
    actions = rng.integers(0, n_actions, size=(B,), endpoint=False, dtype=np.int64)
    return obs, next_obs, actions


def _make_glpe(obs_space, act_space, icm_cfg: ICMConfig) -> GLPE:
    return GLPE(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=icm_cfg,
        region_capacity=128,
        depth_max=0,
        gate_tau_lp_mult=0.5,
        gate_tau_s=0.5,
        gate_hysteresis_up_mult=1.1,
        gate_min_consec_to_gate=1,
        gate_min_regions_for_gating=1,
        normalize_inside=True,
        gating_enabled=True,
    )


def _make_riac(obs_space, act_space, icm_cfg: ICMConfig) -> RIAC:
    return RIAC(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=icm_cfg,
        region_capacity=128,
        depth_max=0,
        ema_beta_long=0.995,
        ema_beta_short=0.90,
        alpha_lp=0.5,
    )


def _assert_resume_restores_extra_state(make_module, method_l: str):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    obs_dim = 4
    n_actions = 3
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Discrete(n_actions)

    icm_cfg = ICMConfig(phi_dim=16, hidden=(32, 32))
    mod1 = make_module(obs_space, act_space, icm_cfg)

    warmup = _rand_batch(rng, obs_dim, n_actions, B=64)
    _ = mod1.compute_batch(*warmup, reduction="none")

    sd_full = mod1.state_dict()
    extra_state = sd_full.get("_extra_state")
    assert extra_state is not None

    sd_no_extra = dict(sd_full)
    sd_no_extra.pop("_extra_state", None)

    eval_batch = _rand_batch(rng, obs_dim, n_actions, B=32)
    with torch.no_grad():
        ref = mod1.compute_batch(*eval_batch, reduction="none").detach().cpu()

    policy = PolicyNetwork(obs_space, act_space)
    value = ValueNetwork(obs_space)
    pol_opt = Adam(policy.parameters(), lr=1e-3)
    val_opt = Adam(value.parameters(), lr=1e-3)

    mod2 = make_module(obs_space, act_space, icm_cfg)

    payload = {
        "step": 100,
        "policy": policy.state_dict(),
        "value": value.state_dict(),
        "obs_norm": None,
        "intrinsic_norm": {},
        "meta": {"updates": 0},
        "optimizers": {"policy": pol_opt.state_dict(), "value": val_opt.state_dict()},
        "intrinsic": {
            "method": method_l,
            "state_dict": sd_no_extra,
            "extra_state": extra_state,
        },
    }

    int_rms = RunningRMS(beta=0.99, eps=1e-8)
    _restore_from_checkpoint(
        resume_payload=payload,
        resume_step=int(payload["step"]),
        policy=policy,
        value=value,
        pol_opt=pol_opt,
        val_opt=val_opt,
        intrinsic_module=mod2,
        method_l=method_l,
        int_rms=int_rms,
        obs_norm=None,
        is_image=False,
        device=torch.device("cpu"),
        logger=get_logger("test_resume_intrinsic"),
    )

    with torch.no_grad():
        out = mod2.compute_batch(*eval_batch, reduction="none").detach().cpu()

    assert torch.allclose(ref, out, atol=1e-6)
    return mod1, mod2


def test_resume_restores_glpe_extra_state():
    m1, m2 = _assert_resume_restores_extra_state(_make_glpe, method_l="glpe")
    assert abs(float(m1.gate_rate) - float(m2.gate_rate)) < 1e-6
    assert abs(float(m1.impact_rms) - float(m2.impact_rms)) < 1e-6
    assert abs(float(m1.lp_rms) - float(m2.lp_rms)) < 1e-6


def test_resume_restores_riac_extra_state():
    m1, m2 = _assert_resume_restores_extra_state(_make_riac, method_l="riac")
    assert abs(float(m1.lp_rms) - float(m2.lp_rms)) < 1e-6
