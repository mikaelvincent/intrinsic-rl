from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import Adam

from irl.cfg import Config
from irl.intrinsic import RunningRMS
from irl.intrinsic.glpe import GLPE
from irl.intrinsic.glpe.gating import _RegionStats, update_region_gate
from irl.intrinsic.icm import ICM, ICMConfig
from irl.intrinsic.regions.kdtree import KDTreeRegionStore
from irl.models.networks import PolicyNetwork, ValueNetwork
from irl.trainer.training_setup import _restore_from_checkpoint
from irl.utils.checkpoint import CheckpointManager
from irl.utils.checkpoint_schema import build_checkpoint_payload
from irl.utils.loggers import get_logger


def test_checkpoint_manager_prune_keeps_step0(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cm = CheckpointManager(run_dir, interval_steps=10, max_to_keep=2)

    for step in (0, 10, 20, 30):
        cm.save(step=step, payload={"step": int(step), "meta": {"note": "test"}})

    kept = sorted(p.name for p in (run_dir / "checkpoints").glob("ckpt_step_*.pt"))
    assert "ckpt_step_0.pt" in kept
    assert "ckpt_step_20.pt" in kept
    assert "ckpt_step_30.pt" in kept
    assert "ckpt_step_10.pt" not in kept


def test_kdtree_bulk_insert_matches_sequential() -> None:
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((100, 3)).astype(np.float32)

    store_seq = KDTreeRegionStore(dim=3, capacity=4, depth_max=6)
    rids_seq = np.array([store_seq.insert(p) for p in pts], dtype=np.int64)

    store_bulk = KDTreeRegionStore(dim=3, capacity=4, depth_max=6)
    rids_bulk = store_bulk.bulk_insert(pts)

    assert np.array_equal(rids_seq, rids_bulk)
    assert store_seq.num_regions() == store_bulk.num_regions()

    store = KDTreeRegionStore(dim=3, capacity=2, depth_max=4)
    rids = store.bulk_insert(np.zeros((5, 3), dtype=np.float32))
    assert np.all(rids == 0)
    assert int(store.num_regions()) == 1


def test_update_region_gate_transitions_and_resets() -> None:
    st = _RegionStats(ema_long=10.0, ema_short=10.0, count=10, gate=1)
    for _ in range(3):
        update_region_gate(
            st,
            lp_i=0.0,
            tau_lp=1.0,
            tau_s=0.5,
            median_error_global=1.0,
            hysteresis_up_mult=1.1,
            min_consec_to_gate=3,
            sufficient_regions=True,
        )
    assert st.gate == 0

    for _ in range(2):
        update_region_gate(
            st,
            lp_i=1.2,
            tau_lp=1.0,
            tau_s=0.5,
            median_error_global=1.0,
            hysteresis_up_mult=1.1,
            min_consec_to_gate=3,
            sufficient_regions=True,
        )
    assert st.gate == 1

    st2 = _RegionStats(ema_long=1.0, ema_short=10.0, count=10, gate=0, bad_consec=5, good_consec=1)
    assert (
        update_region_gate(
            st2,
            lp_i=0.0,
            tau_lp=1.0,
            tau_s=2.0,
            median_error_global=1.0,
            hysteresis_up_mult=2.0,
            min_consec_to_gate=3,
            sufficient_regions=False,
        )
        == 1
    )
    assert st2.bad_consec == 0
    assert st2.good_consec == 0


def test_resume_refuses_kdtree_without_points(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRL_ALLOW_RESUME_WITHOUT_KDTREE_POINTS", raising=False)

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)
    icm_cfg = ICMConfig(phi_dim=16, hidden=(32, 32))

    mod = GLPE(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=icm_cfg,
        region_capacity=4,
        depth_max=4,
        normalize_inside=True,
        gating_enabled=False,
        checkpoint_include_points=False,
    )

    rng = np.random.default_rng(0)
    pts = rng.standard_normal((200, int(mod.phi_dim))).astype(np.float32)
    _ = mod.store.bulk_insert(pts)
    assert int(mod.store.num_regions()) > 1

    cfg = replace(Config(), method="glpe")
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
        intrinsic_module=mod,
        method_l="glpe",
    )

    resumed_mod = GLPE(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=icm_cfg,
        region_capacity=4,
        depth_max=4,
        normalize_inside=True,
        gating_enabled=False,
        checkpoint_include_points=True,
    )

    policy2 = PolicyNetwork(obs_space, act_space)
    value2 = ValueNetwork(obs_space)
    pol_opt2 = Adam(policy2.parameters(), lr=3e-4)
    val_opt2 = Adam(value2.parameters(), lr=3e-4)

    with pytest.raises(RuntimeError, match=r"KDTree points"):
        _restore_from_checkpoint(
            resume_payload=payload,
            resume_step=int(payload["step"]),
            policy=policy2,
            value=value2,
            pol_opt=pol_opt2,
            val_opt=val_opt2,
            intrinsic_module=resumed_mod,
            method_l="glpe",
            int_rms=RunningRMS(beta=0.99, eps=1e-8),
            obs_norm=None,
            is_image=False,
            device=torch.device("cpu"),
            logger=get_logger("test_resume_points_guard"),
        )

    monkeypatch.setenv("IRL_ALLOW_RESUME_WITHOUT_KDTREE_POINTS", "1")
    global_step, _updates = _restore_from_checkpoint(
        resume_payload=payload,
        resume_step=int(payload["step"]),
        policy=policy2,
        value=value2,
        pol_opt=pol_opt2,
        val_opt=val_opt2,
        intrinsic_module=resumed_mod,
        method_l="glpe",
        int_rms=RunningRMS(beta=0.99, eps=1e-8),
        obs_norm=None,
        is_image=False,
        device=torch.device("cpu"),
        logger=get_logger("test_resume_points_override"),
    )
    assert int(global_step) == 10
    assert int(resumed_mod.store.num_regions()) == int(mod.store.num_regions())
    assert int(resumed_mod.store.depth_max) == 0


def test_glpe_restore_with_points_keeps_store_depth_max() -> None:
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    depth_max = 4
    phi_dim = 8
    mod = GLPE(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=ICMConfig(phi_dim=phi_dim, hidden=(16, 16)),
        region_capacity=4,
        depth_max=depth_max,
        normalize_inside=True,
        gating_enabled=False,
        checkpoint_include_points=True,
    )

    rng = np.random.default_rng(0)
    pts = rng.standard_normal((200, int(mod.phi_dim))).astype(np.float32)
    _ = mod.store.bulk_insert(pts)

    assert int(mod.store.num_regions()) > 1
    assert int(mod.store.depth_max) == int(depth_max)

    sd = mod.state_dict()

    restored = GLPE(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=ICMConfig(phi_dim=phi_dim, hidden=(16, 16)),
        region_capacity=4,
        depth_max=depth_max,
        normalize_inside=True,
        gating_enabled=False,
        checkpoint_include_points=True,
    )
    restored.load_state_dict(sd, strict=True)

    assert int(restored.store.depth_max) == int(depth_max)
    assert int(restored.store.num_regions()) == int(mod.store.num_regions())


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
