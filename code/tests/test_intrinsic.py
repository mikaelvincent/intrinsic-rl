from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.optim import Adam

from irl.cfg import Config
from irl.intrinsic import RunningRMS
from irl.intrinsic.glpe import GLPE
from irl.intrinsic.glpe.gating import _RegionStats, update_region_gate
from irl.intrinsic.icm import ICM, ICMConfig
from irl.intrinsic.riac import RIAC
from irl.intrinsic.rnd import RND, RNDConfig
from irl.models.networks import PolicyNetwork, ValueNetwork
from irl.trainer.training_setup import _restore_from_checkpoint
from irl.utils.checkpoint_schema import build_checkpoint_payload
from irl.utils.images import infer_channels_hw
from irl.utils.loggers import get_logger


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


class _Transition:
    def __init__(self, s, a, s_next):
        self.s = s
        self.a = a
        self.r_ext = 0.0
        self.s_next = s_next


def _make_vector_spaces(obs_dim: int = 4, n_actions: int = 3):
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Discrete(n_actions)
    return obs_space, act_space


def _make_glpe(obs_space, act_space, icm_cfg: ICMConfig) -> GLPE:
    return GLPE(
        obs_space,
        act_space,
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


def _make_riac(obs_space, act_space, icm_cfg: ICMConfig) -> RIAC:
    return RIAC(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=icm_cfg,
        region_capacity=1000,
        depth_max=0,
        ema_beta_long=0.995,
        ema_beta_short=0.90,
        alpha_lp=0.5,
    )


def test_intrinsic_compute_batch_matches_step() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    obs_space, act_space = _make_vector_spaces()
    icm_cfg = ICMConfig(phi_dim=16, hidden=(32, 32))

    for make_mod, attrs in (
        (_make_glpe, ("gate_rate", "impact_rms", "lp_rms")),
        (_make_riac, ("lp_rms",)),
    ):
        mod_step = make_mod(obs_space, act_space, icm_cfg)
        mod_batch = make_mod(obs_space, act_space, icm_cfg)
        mod_batch.load_state_dict(mod_step.state_dict())

        B = 32
        obs = rng.standard_normal((B, 4)).astype(np.float32)
        next_obs = rng.standard_normal((B, 4)).astype(np.float32)
        actions = rng.integers(0, 3, size=(B,), endpoint=False, dtype=np.int64)

        step_vals = np.array(
            [
                float(mod_step.compute(_Transition(obs[i], int(actions[i]), next_obs[i])).r_int)
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
        for name in attrs:
            assert abs(float(getattr(mod_step, name)) - float(getattr(mod_batch, name))) < 1e-6


def _rand_batch(rng: np.random.Generator, obs_dim: int, n_actions: int, B: int):
    obs = rng.standard_normal((B, obs_dim)).astype(np.float32)
    next_obs = rng.standard_normal((B, obs_dim)).astype(np.float32)
    actions = rng.integers(0, n_actions, size=(B,), endpoint=False, dtype=np.int64)
    return obs, next_obs, actions


def test_resume_restores_intrinsic_extra_state() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    obs_dim = 4
    n_actions = 3
    obs_space, act_space = _make_vector_spaces(obs_dim=obs_dim, n_actions=n_actions)
    icm_cfg = ICMConfig(phi_dim=16, hidden=(32, 32))

    for make_module, method_l, attrs in (
        (_make_glpe, "glpe", ("gate_rate", "impact_rms", "lp_rms")),
        (_make_riac, "riac", ("lp_rms",)),
    ):
        mod1 = make_module(obs_space, act_space, icm_cfg)
        warmup = _rand_batch(rng, obs_dim, n_actions, B=32)
        _ = mod1.compute_batch(*warmup, reduction="none")

        sd_full = mod1.state_dict()
        extra_state = sd_full.get("_extra_state")
        assert extra_state is not None

        sd_no_extra = dict(sd_full)
        sd_no_extra.pop("_extra_state", None)

        eval_batch = _rand_batch(rng, obs_dim, n_actions, B=16)
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
            "intrinsic": {"method": method_l, "state_dict": sd_no_extra, "extra_state": extra_state},
        }

        _restore_from_checkpoint(
            resume_payload=payload,
            resume_step=int(payload["step"]),
            policy=policy,
            value=value,
            pol_opt=pol_opt,
            val_opt=val_opt,
            intrinsic_module=mod2,
            method_l=method_l,
            int_rms=RunningRMS(beta=0.99, eps=1e-8),
            obs_norm=None,
            is_image=False,
            device=torch.device("cpu"),
            logger=get_logger("test_resume_intrinsic"),
        )

        with torch.no_grad():
            out = mod2.compute_batch(*eval_batch, reduction="none").detach().cpu()

        assert torch.allclose(ref, out, atol=1e-6)
        for name in attrs:
            assert abs(float(getattr(mod1, name)) - float(getattr(mod2, name))) < 1e-6


def test_state_dict_can_omit_kdtree_points(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRL_ALLOW_RESUME_WITHOUT_KDTREE_POINTS", raising=False)

    def _save_state_dict(sd: dict, path: Path) -> int:
        torch.save(sd, path)
        return int(path.stat().st_size)

    def _torch_load_any(path: Path) -> dict:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    def _fill_store(mod: object, *, n_points: int, seed: int) -> None:
        rng = np.random.default_rng(int(seed))
        dim = int(getattr(mod, "phi_dim"))
        pts = rng.standard_normal((int(n_points), dim)).astype(np.float32)
        getattr(mod, "store").bulk_insert(pts)

    def _make_glpe_points(*, include_points: bool) -> GLPE:
        icm_cfg = ICMConfig(phi_dim=32, hidden=(32, 32))
        return GLPE(
            obs_space,
            act_space,
            device="cpu",
            icm_cfg=icm_cfg,
            region_capacity=100_000,
            depth_max=12,
            normalize_inside=False,
            gating_enabled=False,
            checkpoint_include_points=bool(include_points),
        )

    def _make_riac_points(*, include_points: bool) -> RIAC:
        icm_cfg = ICMConfig(phi_dim=32, hidden=(32, 32))
        return RIAC(
            obs_space,
            act_space,
            device="cpu",
            icm_cfg=icm_cfg,
            region_capacity=100_000,
            depth_max=12,
            checkpoint_include_points=bool(include_points),
        )

    def _payload(method_l: str, intrinsic_sd: dict):
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

    for make_module, tag, method_l in (
        (_make_glpe_points, "glpe", "glpe"),
        (_make_riac_points, "riac", "riac"),
    ):
        monkeypatch.delenv("IRL_ALLOW_RESUME_WITHOUT_KDTREE_POINTS", raising=False)

        mod = make_module(include_points=True)
        _fill_store(mod, n_points=5000, seed=0)

        p_with = tmp_path / f"{tag}_with_points.pt"
        size_with = _save_state_dict(mod.state_dict(), p_with)

        mod.checkpoint_include_points = False
        sd_omitted = mod.state_dict()
        p_without = tmp_path / f"{tag}_without_points.pt"
        size_without = _save_state_dict(sd_omitted, p_without)

        assert size_without < size_with
        assert size_without <= int(size_with * 0.6)

        mod2 = make_module(include_points=False)
        mod2.load_state_dict(_torch_load_any(p_without), strict=True)
        assert int(mod2.store.num_regions()) == int(mod.store.num_regions())
        assert int(mod2.store.depth_max) == 0

        payload, policy, value, pol_opt, val_opt = _payload(method_l, _torch_load_any(p_without))

        with pytest.raises(RuntimeError, match="KDTree points"):
            _restore_from_checkpoint(
                resume_payload=payload,
                resume_step=int(payload["step"]),
                policy=policy,
                value=value,
                pol_opt=pol_opt,
                val_opt=val_opt,
                intrinsic_module=make_module(include_points=True),
                method_l=str(method_l),
                int_rms=RunningRMS(beta=0.99, eps=1e-8),
                obs_norm=None,
                is_image=False,
                device=torch.device("cpu"),
                logger=get_logger("test_resume_points"),
            )

        monkeypatch.setenv("IRL_ALLOW_RESUME_WITHOUT_KDTREE_POINTS", "1")
        global_step, _updates = _restore_from_checkpoint(
            resume_payload=payload,
            resume_step=int(payload["step"]),
            policy=policy,
            value=value,
            pol_opt=pol_opt,
            val_opt=val_opt,
            intrinsic_module=make_module(include_points=True),
            method_l=str(method_l),
            int_rms=RunningRMS(beta=0.99, eps=1e-8),
            obs_norm=None,
            is_image=False,
            device=torch.device("cpu"),
            logger=get_logger("test_resume_points_override"),
        )
        assert int(global_step) == int(payload["step"])


def test_grayscale_hw_observations_work_end_to_end() -> None:
    c, hw = infer_channels_hw((32, 32))
    assert c == 1
    assert hw == (32, 32)

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    H, W = 32, 32
    obs_space = gym.spaces.Box(low=0, high=255, shape=(H, W), dtype=np.uint8)
    act_space = gym.spaces.Discrete(3)

    policy = PolicyNetwork(obs_space, act_space).to(torch.device("cpu"))
    value = ValueNetwork(obs_space).to(torch.device("cpu"))

    obs = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
    assert policy.distribution(obs).sample().shape == (1,)
    v = value(obs)
    assert v.shape == (1,)
    assert torch.isfinite(v).all()

    B = 4
    obs_b = rng.integers(0, 256, size=(B, H, W), dtype=np.uint8)
    dist_b = policy.distribution(obs_b)
    assert dist_b.logits.shape == (B, int(act_space.n))
    v_b = value(obs_b)
    assert v_b.shape == (B,)
    assert torch.isfinite(v_b).all()

    icm = ICM(obs_space, act_space, device="cpu", cfg=ICMConfig(phi_dim=32, hidden=(64, 64)))
    next_obs_b = rng.integers(0, 256, size=(B, H, W), dtype=np.uint8)
    actions = rng.integers(0, int(act_space.n), size=(B,), endpoint=False, dtype=np.int64)
    r_icm = icm.compute_batch(obs_b, next_obs_b, actions, reduction="none")
    assert r_icm.shape == (B,)
    assert torch.isfinite(r_icm).all()

    rnd = RND(obs_space, device="cpu", cfg=RNDConfig(feature_dim=32, hidden=(64, 64)))
    r_rnd = rnd.compute_batch(obs_b, reduction="none")
    assert r_rnd.shape == (B,)
    assert torch.isfinite(r_rnd).all()


def _flat_params(model: torch.nn.Module) -> torch.Tensor:
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
