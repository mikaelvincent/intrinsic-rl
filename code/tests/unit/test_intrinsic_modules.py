from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.envs.registration import register
from torch.optim import Adam

from irl.cfg import Config, validate_config
from irl.intrinsic import RunningRMS
from irl.intrinsic.glpe import GLPE
from irl.intrinsic.glpe.gating import _RegionStats, update_region_gate
from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.riac import RIAC
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


def test_update_region_gate_transitions_and_resets():
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


def test_glpe_normalize_inside_changes_output():
    torch.manual_seed(0)
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)
    icm_cfg = ICMConfig(phi_dim=16, hidden=(32, 32))

    mod_raw = GLPE(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=icm_cfg,
        normalize_inside=False,
        gating_enabled=False,
    )
    mod_norm = GLPE(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=icm_cfg,
        normalize_inside=True,
        gating_enabled=False,
    )
    mod_norm.load_state_dict(mod_raw.state_dict())

    rng = np.random.default_rng(123)
    obs = rng.standard_normal((128, 4)).astype(np.float32)
    next_obs = rng.standard_normal((128, 4)).astype(np.float32)
    actions = rng.integers(0, 3, size=(128,), endpoint=False, dtype=np.int64)

    with torch.no_grad():
        r_raw = mod_raw.compute_batch(obs, next_obs, actions)
        r_norm = mod_norm.compute_batch(obs, next_obs, actions)

    assert not torch.allclose(r_raw, r_norm, atol=1e-6)


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


def test_glpe_compute_batch_matches_sequential_compute():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    obs_space, act_space = _make_vector_spaces()
    icm_cfg = ICMConfig(phi_dim=16, hidden=(32, 32))

    mod_step = _make_glpe(obs_space, act_space, icm_cfg)
    mod_batch = _make_glpe(obs_space, act_space, icm_cfg)
    mod_batch.load_state_dict(mod_step.state_dict())

    B = 64
    obs = rng.standard_normal((B, 4)).astype(np.float32)
    next_obs = rng.standard_normal((B, 4)).astype(np.float32)
    actions = rng.integers(0, 3, size=(B,), endpoint=False, dtype=np.int64)

    step_vals = np.array(
        [float(mod_step.compute(_Transition(obs[i], int(actions[i]), next_obs[i])).r_int) for i in range(B)],
        dtype=np.float32,
    )

    with torch.no_grad():
        batch_vals = (
            mod_batch.compute_batch(obs, next_obs, actions, reduction="none").detach().cpu().numpy().astype(np.float32)
        )

    assert np.allclose(step_vals, batch_vals, atol=1e-5)
    assert abs(mod_step.gate_rate - mod_batch.gate_rate) < 1e-6
    assert abs(mod_step.impact_rms - mod_batch.impact_rms) < 1e-6
    assert abs(mod_step.lp_rms - mod_batch.lp_rms) < 1e-6


def test_riac_compute_batch_matches_sequential_compute():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    obs_space, act_space = _make_vector_spaces()
    icm_cfg = ICMConfig(phi_dim=16, hidden=(32, 32))

    mod_step = _make_riac(obs_space, act_space, icm_cfg)
    mod_batch = _make_riac(obs_space, act_space, icm_cfg)
    mod_batch.load_state_dict(mod_step.state_dict())

    B = 64
    obs = rng.standard_normal((B, 4)).astype(np.float32)
    next_obs = rng.standard_normal((B, 4)).astype(np.float32)
    actions = rng.integers(0, 3, size=(B,), endpoint=False, dtype=np.int64)

    step_vals = np.array(
        [float(mod_step.compute(_Transition(obs[i], int(actions[i]), next_obs[i])).r_int) for i in range(B)],
        dtype=np.float32,
    )

    with torch.no_grad():
        batch_vals = (
            mod_batch.compute_batch(obs, next_obs, actions, reduction="none").detach().cpu().numpy().astype(np.float32)
        )

    assert np.allclose(step_vals, batch_vals, atol=1e-5)
    assert abs(mod_step.lp_rms - mod_batch.lp_rms) < 1e-6


def _rand_batch(rng: np.random.Generator, obs_dim: int, n_actions: int, B: int):
    obs = rng.standard_normal((B, obs_dim)).astype(np.float32)
    next_obs = rng.standard_normal((B, obs_dim)).astype(np.float32)
    actions = rng.integers(0, n_actions, size=(B,), endpoint=False, dtype=np.int64)
    return obs, next_obs, actions


@pytest.mark.parametrize(
    "make_module, method_l, attrs",
    [
        (_make_glpe, "glpe", ("gate_rate", "impact_rms", "lp_rms")),
        (_make_riac, "riac", ("lp_rms",)),
    ],
)
def test_resume_restores_intrinsic_extra_state(make_module, method_l: str, attrs: tuple[str, ...]):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    obs_dim = 4
    n_actions = 3
    obs_space, act_space = _make_vector_spaces(obs_dim=obs_dim, n_actions=n_actions)
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
