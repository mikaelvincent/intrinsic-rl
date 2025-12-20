from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from irl.cfg.schema import Config
from irl.intrinsic import RunningRMS
from irl.intrinsic.icm import ICM, ICMConfig
from irl.models.networks import PolicyNetwork, ValueNetwork
from irl.trainer.training_setup import _restore_from_checkpoint
from irl.utils.checkpoint_schema import build_checkpoint_payload
from irl.utils.loggers import get_logger


def _flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().cpu().view(-1) for p in model.parameters()])


def test_resume_restores_intrinsic_optimizer_state_equivalence() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)
    icm_cfg = ICMConfig(phi_dim=8, hidden=(16, 16), lr=1e-3)

    base = ICM(obs_space, act_space, device="cpu", cfg=icm_cfg)
    init_sd = {k: v.clone() for k, v in base.state_dict().items()}

    B = 64
    obs = rng.standard_normal((B, 4)).astype(np.float32)
    next_obs = rng.standard_normal((B, 4)).astype(np.float32)
    actions = rng.integers(0, int(act_space.n), size=(B,), endpoint=False, dtype=np.int64)

    K = 3

    expected = ICM(obs_space, act_space, device="cpu", cfg=icm_cfg)
    expected.load_state_dict(init_sd, strict=True)
    for _ in range(K + 1):
        _ = expected.update(obs, next_obs, actions, steps=1)

    trained = ICM(obs_space, act_space, device="cpu", cfg=icm_cfg)
    trained.load_state_dict(init_sd, strict=True)
    for _ in range(K):
        _ = trained.update(obs, next_obs, actions, steps=1)

    cfg = replace(Config(), method="icm")
    policy = PolicyNetwork(obs_space, act_space)
    value = ValueNetwork(obs_space)
    pol_opt = Adam(policy.parameters(), lr=3e-4)
    val_opt = Adam(value.parameters(), lr=3e-4)

    payload = build_checkpoint_payload(
        cfg,
        global_step=10,
        update_idx=0,
        policy=policy,
        value=value,
        is_image=False,
        obs_norm=None,
        int_rms=RunningRMS(beta=0.99, eps=1e-8),
        pol_opt=pol_opt,
        val_opt=val_opt,
        intrinsic_module=trained,
        method_l="icm",
    )

    intr = payload.get("intrinsic")
    assert isinstance(intr, dict)
    assert isinstance(intr.get("optimizers"), dict)
    assert "main" in set(intr["optimizers"].keys())

    resumed = ICM(obs_space, act_space, device="cpu", cfg=icm_cfg)

    policy2 = PolicyNetwork(obs_space, act_space)
    value2 = ValueNetwork(obs_space)
    pol_opt2 = Adam(policy2.parameters(), lr=3e-4)
    val_opt2 = Adam(value2.parameters(), lr=3e-4)

    _restore_from_checkpoint(
        resume_payload=payload,
        resume_step=int(payload["step"]),
        policy=policy2,
        value=value2,
        pol_opt=pol_opt2,
        val_opt=val_opt2,
        intrinsic_module=resumed,
        method_l="icm",
        int_rms=RunningRMS(beta=0.99, eps=1e-8),
        obs_norm=None,
        is_image=False,
        device=torch.device("cpu"),
        logger=get_logger("test_resume_intrinsic_optimizer_state"),
    )

    _ = resumed.update(obs, next_obs, actions, steps=1)

    diff = float((_flat_params(expected) - _flat_params(resumed)).abs().max().item())
    assert diff < 1e-7
