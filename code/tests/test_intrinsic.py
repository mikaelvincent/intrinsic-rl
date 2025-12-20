from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import register
from torch.nn import functional as F
from torch.optim import Adam

from irl.cfg import Config, validate_config
from irl.intrinsic import RunningRMS
from irl.intrinsic.glpe import GLPE
from irl.intrinsic.glpe.gating import _RegionStats, update_region_gate
from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.riac import RIAC
from irl.intrinsic.rnd import RND, RNDConfig
from irl.models.networks import PolicyNetwork, ValueNetwork
from irl.trainer import train as run_train
from irl.trainer.training_setup import _restore_from_checkpoint
from irl.utils.loggers import get_logger


class _Transition:
    def __init__(self, s, a, s_next):
        self.s = s
        self.a = a
        self.r_ext = 0.0
        self.s_next = s_next


def test_update_region_gate_transitions_and_resets() -> None:
    st = _RegionStats(ema_long=10.0, ema_short=10.0, count=10, gate=1)

    for _ in range(2):
        assert (
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
            == 1
        )

    assert (
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
        == 0
    )

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


class _DummyGateEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(4,)).astype(np.float32)
        return obs, {}

    def step(self, action):
        self._t += 1
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(4,)).astype(np.float32)
        terminated = self._t >= 3
        return obs, 0.0, bool(terminated), False, {}


try:
    register(id="DummyGate-v0", entry_point=_DummyGateEnv)
except Exception:
    pass


def _make_cfg(*, method: str, eta: float = 0.1) -> Config:
    base = Config()
    env_cfg = replace(
        base.env,
        id="DummyGate-v0",
        vec_envs=1,
        frame_skip=1,
        domain_randomization=False,
        discrete_actions=True,
    )
    ppo_cfg = replace(
        base.ppo,
        steps_per_update=4,
        minibatches=1,
        epochs=1,
        entropy_coef=0.0,
    )
    intrinsic_cfg = replace(base.intrinsic, eta=float(eta), alpha_impact=0.0, alpha_lp=0.5)
    log_cfg = replace(base.logging, csv_interval=1, checkpoint_interval=100_000)
    eval_cfg = replace(base.evaluation, interval_steps=100_000, episodes=1)
    adapt_cfg = replace(base.adaptation, enabled=False)

    cfg = replace(
        base,
        device="cpu",
        method=str(method),
        env=env_cfg,
        ppo=ppo_cfg,
        intrinsic=intrinsic_cfg,
        logging=log_cfg,
        evaluation=eval_cfg,
        adaptation=adapt_cfg,
    )
    validate_config(cfg)
    return cfg


def test_glpe_lp_only_logs_gate_rate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_glpe_lp_only"
    cfg = _make_cfg(method="glpe_lp_only", eta=0.1)
    out = run_train(cfg, total_steps=8, run_dir=run_dir, resume=False)

    csv_path = out / "logs" / "scalars.csv"
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        cols = set(reader.fieldnames)

    assert {"gate_rate", "gate_rate_pct"} <= cols


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

        B = 64
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
            "intrinsic": {"method": method_l, "state_dict": sd_no_extra, "extra_state": extra_state},
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
        for name in attrs:
            assert abs(float(getattr(mod1, name)) - float(getattr(mod2, name))) < 1e-6


def test_state_dict_can_omit_kdtree_points(tmp_path: Path) -> None:
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

    for make_module, tag in ((_make_glpe_points, "glpe"), (_make_riac_points, "riac")):
        mod = make_module(include_points=True)
        _fill_store(mod, n_points=5000, seed=0)

        p_with = tmp_path / f"{tag}_with_points.pt"
        size_with = _save_state_dict(mod.state_dict(), p_with)

        mod.checkpoint_include_points = False
        p_without = tmp_path / f"{tag}_without_points.pt"
        size_without = _save_state_dict(mod.state_dict(), p_without)

        assert size_without < size_with
        assert size_without <= int(size_with * 0.6)

        mod2 = make_module(include_points=False)
        mod2.load_state_dict(_torch_load_any(p_without), strict=True)

        assert int(mod2.store.num_regions()) == int(mod.store.num_regions())
        assert int(mod2.store.depth_max) == 0


def test_rnd_next_obs_and_rms_update() -> None:
    obs_space_img = gym.spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8)
    rnd_img = RND(obs_space_img, device="cpu", cfg=RNDConfig(feature_dim=32, hidden=(64, 64)))

    rng = np.random.default_rng(1)
    B = 10
    H, W, C = (int(x) for x in obs_space_img.shape)
    obs = rng.integers(0, 256, size=(B, H, W, C), dtype=np.uint8)
    next_obs = rng.integers(0, 256, size=(B, H, W, C), dtype=np.uint8)

    r2 = rnd_img.compute_batch(obs, next_obs)
    r3 = rnd_img.compute_batch(next_obs)

    assert r2.shape == r3.shape == (B,)
    assert torch.isfinite(r2).all()
    assert torch.isfinite(r3).all()
    assert torch.allclose(r2, r3, atol=1e-6)

    obs_space_vec = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    cfg = RNDConfig(feature_dim=16, hidden=(32, 32), rms_beta=0.0, normalize_intrinsic=True)
    rnd = RND(obs_space_vec, device="cpu", cfg=cfg)

    obs2 = rng.standard_normal((64, 5)).astype(np.float32)

    with torch.no_grad():
        x = torch.as_tensor(obs2, dtype=torch.float32)
        p = rnd.predictor(x)
        tgt = rnd.target(x)
        per = F.mse_loss(p, tgt, reduction="none").mean(dim=-1)
        expected_rms = float(torch.sqrt((per**2).mean() + float(cfg.rms_eps)).item())

    _ = rnd.compute_batch(obs2)
    assert abs(float(rnd.rms) - expected_rms) < 1e-6
