from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.optim import Adam

from irl.intrinsic import RunningRMS
from irl.intrinsic.glpe import GLPE
from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.riac import RIAC
from irl.models.networks import PolicyNetwork, ValueNetwork
from irl.trainer.resume import _restore_from_checkpoint
from irl.utils.loggers import get_logger


def _spaces():
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)
    return obs_space, act_space


def _state_dict_without_points(method_l: str, obs_space, act_space) -> dict:
    icm_cfg = ICMConfig(phi_dim=16, hidden=(32, 32))
    if str(method_l).startswith("glpe"):
        mod = GLPE(
            obs_space,
            act_space,
            device="cpu",
            icm_cfg=icm_cfg,
            region_capacity=8,
            depth_max=2,
            normalize_inside=False,
            gating_enabled=False,
            checkpoint_include_points=False,
        )
    else:
        mod = RIAC(
            obs_space,
            act_space,
            device="cpu",
            icm_cfg=icm_cfg,
            region_capacity=8,
            depth_max=2,
            checkpoint_include_points=False,
        )
    return mod.state_dict()


def _fresh_intrinsic(method_l: str, obs_space, act_space):
    icm_cfg = ICMConfig(phi_dim=16, hidden=(32, 32))
    if str(method_l).startswith("glpe"):
        return GLPE(
            obs_space,
            act_space,
            device="cpu",
            icm_cfg=icm_cfg,
            region_capacity=8,
            depth_max=2,
            normalize_inside=False,
            gating_enabled=False,
            checkpoint_include_points=True,
        )
    return RIAC(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=icm_cfg,
        region_capacity=8,
        depth_max=2,
        checkpoint_include_points=True,
    )


def _payload(method_l: str, intrinsic_sd: dict, obs_space, act_space) -> tuple[dict, object, object, object, object]:
    torch.manual_seed(0)
    policy = PolicyNetwork(obs_space, act_space)
    value = ValueNetwork(obs_space)
    pol_opt = Adam(policy.parameters(), lr=1e-3)
    val_opt = Adam(value.parameters(), lr=1e-3)

    payload = {
        "step": 10,
        "policy": policy.state_dict(),
        "value": value.state_dict(),
        "obs_norm": None,
        "intrinsic_norm": {},
        "meta": {"updates": 0},
        "optimizers": {"policy": pol_opt.state_dict(), "value": val_opt.state_dict()},
        "intrinsic": {"method": str(method_l), "state_dict": intrinsic_sd},
    }
    return payload, policy, value, pol_opt, val_opt


@pytest.mark.parametrize("method_l", ["glpe", "riac"])
def test_resume_refuses_when_kdtree_points_omitted(method_l: str) -> None:
    obs_space, act_space = _spaces()
    sd = _state_dict_without_points(method_l, obs_space, act_space)
    payload, policy, value, pol_opt, val_opt = _payload(method_l, sd, obs_space, act_space)

    intrinsic_module = _fresh_intrinsic(method_l, obs_space, act_space)
    int_rms = RunningRMS(beta=0.99, eps=1e-8)

    with pytest.raises(RuntimeError, match="KDTree points"):
        _restore_from_checkpoint(
            resume_payload=payload,
            resume_step=int(payload["step"]),
            policy=policy,
            value=value,
            pol_opt=pol_opt,
            val_opt=val_opt,
            intrinsic_module=intrinsic_module,
            method_l=str(method_l),
            int_rms=int_rms,
            obs_norm=None,
            is_image=False,
            device=torch.device("cpu"),
            logger=get_logger("test_resume_points"),
        )


@pytest.mark.parametrize("method_l", ["glpe", "riac"])
def test_resume_allows_with_env_override(monkeypatch: pytest.MonkeyPatch, method_l: str) -> None:
    monkeypatch.setenv("IRL_ALLOW_RESUME_WITHOUT_KDTREE_POINTS", "1")

    obs_space, act_space = _spaces()
    sd = _state_dict_without_points(method_l, obs_space, act_space)
    payload, policy, value, pol_opt, val_opt = _payload(method_l, sd, obs_space, act_space)

    intrinsic_module = _fresh_intrinsic(method_l, obs_space, act_space)
    int_rms = RunningRMS(beta=0.99, eps=1e-8)

    global_step, _updates = _restore_from_checkpoint(
        resume_payload=payload,
        resume_step=int(payload["step"]),
        policy=policy,
        value=value,
        pol_opt=pol_opt,
        val_opt=val_opt,
        intrinsic_module=intrinsic_module,
        method_l=str(method_l),
        int_rms=int_rms,
        obs_norm=None,
        is_image=False,
        device=torch.device("cpu"),
        logger=get_logger("test_resume_points_override"),
    )

    assert int(global_step) == int(payload["step"])
    assert int(getattr(intrinsic_module.store, "depth_max", -1)) == 0
