from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.envs.registration import register
from torch import nn

from irl.algo.advantage import compute_gae
from irl.algo.ppo import ppo_update
from irl.cfg import Config, validate_config
from irl.cfg.schema import PPOConfig
from irl.envs.wrappers import FrameSkip
from irl.evaluator import evaluate
from irl.intrinsic.config import build_intrinsic_kwargs
from irl.intrinsic.factory import create_intrinsic_module
from irl.intrinsic.glpe import GLPE
from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.regions.kdtree import KDTreeRegionStore
from irl.intrinsic.riac import RIAC
from irl.models.networks import PolicyNetwork, ValueNetwork
from irl.pipelines.policy_rollout import iter_policy_rollout
from irl.trainer import train as run_train
from irl.trainer.runtime_utils import _apply_final_observation
from irl.utils.checkpoint import CheckpointManager, load_checkpoint


class _ValueSequence(nn.Module):
    def __init__(self, v_t, v_tp1):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))
        self.v_t = torch.as_tensor(v_t, dtype=torch.float32).view(-1)
        self.v_tp1 = torch.as_tensor(v_tp1, dtype=torch.float32).view(-1)
        self._calls = 0

    def forward(self, _obs):
        self._calls += 1
        return self.v_t if self._calls == 1 else self.v_tp1


class _BoundedBoxEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array([-0.5, -1.0], dtype=np.float32),
            high=np.array([0.5, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._t = 0

    def reset(self, *, seed=None, options=None):
        _ = options
        self._t = 0
        return np.zeros((4,), dtype=np.float32), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        lo = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
        hi = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)
        assert a.shape == lo.shape == hi.shape
        assert np.all(a >= lo - 1e-6)
        assert np.all(a <= hi + 1e-6)

        self._t += 1
        terminated = self._t >= 3
        obs = np.zeros((4,), dtype=np.float32)
        return obs, 0.0, bool(terminated), False, {}

    def close(self) -> None:
        return


class _FrameSkipEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.t = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.t = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        _ = action
        self.t += 1
        obs = np.array([float(self.t)], dtype=np.float32)
        reward = 1.0
        terminated = self.t >= 3
        truncated = False
        info = {"t": self.t}
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        return


class _DummyEvalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        _ = options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(4,)).astype(np.float32)
        return obs, {}

    def step(self, action):
        self._t += 1
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(4,)).astype(np.float32)
        reward = float(action)
        terminated = self._t >= 5
        truncated = False
        return obs, reward, bool(terminated), bool(truncated), {}


try:
    register(id="DummyEval-v0", entry_point=_DummyEvalEnv)
except Exception:
    pass


class _DummyTrajEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        _ = options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(4,)).astype(np.float32)
        return obs, {}

    def step(self, action):
        self._t += 1
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(4,)).astype(np.float32)
        reward = float(action)
        terminated = self._t >= 3
        truncated = False
        return obs, reward, bool(terminated), bool(truncated), {}


try:
    register(id="DummyTraj-v0", entry_point=_DummyTrajEnv)
except Exception:
    pass


class _StepBudgetEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        _ = seed, options
        return np.zeros((2,), dtype=np.float32), {}

    def step(self, action):
        _ = action
        obs = np.zeros((2,), dtype=np.float32)
        return obs, 0.0, False, False, {}


try:
    register(id="StepBudget-v0", entry_point=_StepBudgetEnv)
except Exception:
    pass


class _TimeoutMaskEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._init_seed = 0 if seed is None else int(seed)
        self._episode = 0
        self._t = 0
        self._mode = "term"

    def reset(self, *, seed=None, options=None):
        _ = options
        if seed is not None:
            self._init_seed = int(seed)
        self._episode += 1
        self._t = 0

        if self._episode == 1:
            self._mode = "term" if (int(self._init_seed) % 2 == 1) else "trunc"
        else:
            self._mode = "term"

        obs = np.array([float(self._episode), 0.0, 0.0, 0.0], dtype=np.float32)
        return obs, {}

    def step(self, action):
        self._t += 1
        obs = np.array([float(self._episode), float(self._t), float(action), 1.0], dtype=np.float32)

        terminated = self._t >= 1 and self._mode == "term"
        truncated = self._t >= 1 and self._mode == "trunc"
        return obs, 0.0, bool(terminated), bool(truncated), {}


try:
    register(id="TimeoutMask-v0", entry_point=_TimeoutMaskEnv)
except Exception:
    pass


def _flat_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().cpu().view(-1) for p in model.parameters()])


def _make_step_budget_cfg(*, vec_envs: int) -> Config:
    base = Config()
    env_cfg = replace(
        base.env,
        id="StepBudget-v0",
        vec_envs=int(vec_envs),
        frame_skip=1,
        domain_randomization=False,
        discrete_actions=True,
        async_vector=False,
    )
    ppo_cfg = replace(
        base.ppo,
        steps_per_update=2,
        minibatches=1,
        epochs=1,
        entropy_coef=0.0,
    )
    intrinsic_cfg = replace(base.intrinsic, eta=0.0)
    log_cfg = replace(base.logging, csv_interval=1, checkpoint_interval=100_000)
    eval_cfg = replace(base.evaluation, interval_steps=100_000, episodes=1)
    adapt_cfg = replace(base.adaptation, enabled=False)

    cfg = replace(
        base,
        device="cpu",
        method="vanilla",
        env=env_cfg,
        ppo=ppo_cfg,
        intrinsic=intrinsic_cfg,
        logging=log_cfg,
        evaluation=eval_cfg,
        adaptation=adapt_cfg,
    )
    validate_config(cfg)
    return cfg


def _make_timeout_mask_cfg() -> Config:
    base = Config()
    env_cfg = replace(
        base.env,
        id="TimeoutMask-v0",
        vec_envs=2,
        frame_skip=1,
        domain_randomization=False,
        discrete_actions=True,
        async_vector=False,
    )
    ppo_cfg = replace(
        base.ppo,
        steps_per_update=1,
        minibatches=1,
        epochs=1,
        entropy_coef=0.0,
    )
    intrinsic_cfg = replace(base.intrinsic, eta=0.1)
    log_cfg = replace(base.logging, csv_interval=1, checkpoint_interval=100_000)
    eval_cfg = replace(base.evaluation, interval_steps=100_000, episodes=1)
    adapt_cfg = replace(base.adaptation, enabled=False)

    cfg = replace(
        base,
        device="cpu",
        method="ride",
        env=env_cfg,
        ppo=ppo_cfg,
        intrinsic=intrinsic_cfg,
        logging=log_cfg,
        evaluation=eval_cfg,
        adaptation=adapt_cfg,
    )
    validate_config(cfg)
    return cfg


def _make_spaces() -> tuple[gym.Space, gym.Space]:
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)
    return obs_space, act_space


def _fill_store(mod: object, *, n_points: int, seed: int) -> None:
    rng = np.random.default_rng(int(seed))
    dim = int(getattr(mod, "phi_dim"))
    pts = rng.standard_normal((int(n_points), dim)).astype(np.float32)
    store = getattr(mod, "store")
    for p in pts:
        store.insert(p)


def test_compute_gae_bootstraps_timeouts_and_dones():
    obs = torch.zeros((3, 1, 1), dtype=torch.float32)
    next_obs = torch.zeros((3, 1, 1), dtype=torch.float32)

    rewards = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32)
    v_t = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    v_tp1 = torch.tensor([20.0, 30.0, 40.0], dtype=torch.float32)

    dones = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float32)
    vf_term = _ValueSequence(v_t, v_tp1)
    adv_term, vt_term = compute_gae(
        {"obs": obs, "next_observations": next_obs, "rewards": rewards, "dones": dones},
        value_fn=vf_term,
        gamma=1.0,
        lam=1.0,
        bootstrap_on_timeouts=False,
    )
    assert torch.allclose(
        adv_term, torch.tensor([-7.0, -18.0, -29.0], dtype=torch.float32), atol=1e-6
    )
    assert torch.allclose(vt_term, torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32), atol=1e-6)

    terminals = torch.tensor([[0.0], [0.0], [0.0]], dtype=torch.float32)
    truncs = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float32)
    vf_boot = _ValueSequence(v_t, v_tp1)
    adv_boot, vt_boot = compute_gae(
        {
            "obs": obs,
            "next_observations": next_obs,
            "rewards": rewards,
            "terminals": terminals,
            "truncations": truncs,
        },
        value_fn=vf_boot,
        gamma=1.0,
        lam=1.0,
        bootstrap_on_timeouts=True,
    )
    assert torch.allclose(
        adv_boot, torch.tensor([33.0, 22.0, 11.0], dtype=torch.float32), atol=1e-6
    )
    assert torch.allclose(
        vt_boot, torch.tensor([43.0, 42.0, 41.0], dtype=torch.float32), atol=1e-6
    )


def test_compute_gae_does_not_leak_across_truncation():
    obs = torch.zeros((4, 1, 1), dtype=torch.float32)
    next_obs = torch.zeros((4, 1, 1), dtype=torch.float32)

    rewards = torch.tensor([[0.0], [0.0], [10.0], [0.0]], dtype=torch.float32)
    terminals = torch.tensor([[0.0], [0.0], [0.0], [1.0]], dtype=torch.float32)
    truncations = torch.tensor([[0.0], [1.0], [0.0], [0.0]], dtype=torch.float32)

    vf = _ValueSequence(
        v_t=[0.0, 0.0, 0.0, 0.0],
        v_tp1=[0.0, 5.0, 0.0, 0.0],
    )
    adv, v_targets = compute_gae(
        {
            "obs": obs,
            "next_observations": next_obs,
            "rewards": rewards,
            "terminals": terminals,
            "truncations": truncations,
        },
        value_fn=vf,
        gamma=1.0,
        lam=1.0,
        bootstrap_on_timeouts=True,
    )

    expected = torch.tensor([5.0, 5.0, 10.0, 0.0], dtype=torch.float32)
    assert torch.allclose(adv, expected, atol=1e-6)
    assert torch.allclose(v_targets, expected, atol=1e-6)


def test_ppo_update_stats_and_kl_penalty_affect_policy():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(3)

    base_policy = PolicyNetwork(obs_space, act_space)
    base_value = ValueNetwork(obs_space)

    init_pol = {k: v.clone() for k, v in base_policy.state_dict().items()}
    init_val = {k: v.clone() for k, v in base_value.state_dict().items()}

    N = 64
    obs = rng.standard_normal((N, 4)).astype(np.float32)

    with torch.no_grad():
        dist0 = base_policy.distribution(torch.as_tensor(obs, dtype=torch.float32))
        actions_t = dist0.sample()
        old_logp_t = dist0.log_prob(actions_t)

    batch = {
        "obs": obs,
        "actions": actions_t.detach().cpu().numpy(),
        "old_log_probs": old_logp_t.detach().cpu().numpy().astype(np.float32),
    }
    advantages = rng.standard_normal(N).astype(np.float32)
    value_targets = rng.standard_normal(N).astype(np.float32)

    cfg = PPOConfig(
        steps_per_update=N,
        minibatches=2,
        epochs=2,
        learning_rate=3.0e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.0,
        value_coef=0.5,
        value_clip_range=0.0,
        kl_penalty_coef=0.0,
        kl_stop=0.0,
    )

    policy_a = PolicyNetwork(obs_space, act_space)
    value_a = ValueNetwork(obs_space)
    policy_a.load_state_dict(init_pol)
    value_a.load_state_dict(init_val)

    torch.manual_seed(123)
    stats = ppo_update(
        policy_a,
        value_a,
        batch,
        advantages,
        value_targets,
        cfg,
        return_stats=True,
    )
    assert stats is not None
    assert float(stats["epochs_ran"]) == float(cfg.epochs)
    for k in ("approx_kl", "clip_frac", "entropy", "policy_loss", "value_loss"):
        assert np.isfinite(float(stats[k]))

    policy_b = PolicyNetwork(obs_space, act_space)
    value_b = ValueNetwork(obs_space)
    policy_b.load_state_dict(init_pol)
    value_b.load_state_dict(init_val)

    torch.manual_seed(123)
    _ = ppo_update(
        policy_b,
        value_b,
        batch,
        advantages,
        value_targets,
        PPOConfig(**{**cfg.__dict__, "kl_penalty_coef": 10.0}),
        return_stats=False,
    )

    pa = _flat_params(policy_a)
    pb = _flat_params(policy_b)
    assert float((pa - pb).abs().max().item()) > 1e-7


def test_policy_rollout_continuous_actions_within_bounds():
    env = _BoundedBoxEnv()
    try:
        policy = PolicyNetwork(env.observation_space, env.action_space).to(torch.device("cpu"))
        policy.eval()

        for mode in ("mode", "sample"):
            obs0, _ = env.reset(seed=0)
            for step in iter_policy_rollout(
                env=env,
                policy=policy,
                obs0=obs0,
                act_space=env.action_space,
                device=torch.device("cpu"),
                policy_mode=mode,
                normalize_obs=None,
                max_steps=5,
            ):
                lp = policy.distribution(step.obs_t).log_prob(step.act_t)
                assert torch.isfinite(lp).all()
    finally:
        env.close()


def test_checkpoint_manager_prune_keeps_step0() -> None:
    def _payload(step: int) -> dict:
        return {"step": int(step), "meta": {"note": "test"}}

    with TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        cm = CheckpointManager(run_dir, interval_steps=10, max_to_keep=2)

        for step in (0, 10, 20, 30):
            cm.save(step=step, payload=_payload(step))

        ckpt_dir = run_dir / "checkpoints"
        kept = sorted(p.name for p in ckpt_dir.glob("ckpt_step_*.pt"))

        assert "ckpt_step_0.pt" in kept
        assert "ckpt_step_30.pt" in kept
        assert "ckpt_step_20.pt" in kept
        assert "ckpt_step_10.pt" not in kept


def test_state_dict_can_omit_kdtree_points(tmp_path: Path) -> None:
    def _save_state_dict(sd: dict, path: Path) -> int:
        torch.save(sd, path)
        return int(path.stat().st_size)

    def _torch_load_any(path: Path) -> dict:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    obs_space, act_space = _make_spaces()

    def _make_glpe(*, include_points: bool) -> GLPE:
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

    def _make_riac(*, include_points: bool) -> RIAC:
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

    for make_module, tag in ((_make_glpe, "glpe"), (_make_riac, "riac")):
        mod = make_module(include_points=True)
        _fill_store(mod, n_points=5000, seed=0)

        p_with = tmp_path / f"{tag}_with_points.pt"
        size_with = _save_state_dict(mod.state_dict(), p_with)

        mod.checkpoint_include_points = False
        p_without = tmp_path / f"{tag}_without_points.pt"
        size_without = _save_state_dict(mod.state_dict(), p_without)

        assert size_without < size_with
        assert size_without <= int(size_with * 0.4)

        mod2 = make_module(include_points=False)
        mod2.load_state_dict(_torch_load_any(p_without), strict=True)

        assert int(mod2.store.num_regions()) == int(mod.store.num_regions())
        assert int(mod2.store.depth_max) == 0


def test_train_refuses_dirty_run_dir_without_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dirty"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "ckpt_latest.pt").write_bytes(b"not a real checkpoint")
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / "scalars.csv").write_text(
        "step,reward_total_mean\n0,0.0\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="resume"):
        run_train(Config(), total_steps=1, run_dir=run_dir, resume=False)


def test_total_steps_aligns_to_vec_envs_budget(tmp_path: Path) -> None:
    cfg = _make_step_budget_cfg(vec_envs=2)
    out_dir = run_train(cfg, total_steps=5, run_dir=tmp_path / "run_budget", resume=False)

    payload = load_checkpoint(out_dir / "checkpoints" / "ckpt_latest.pt", map_location="cpu")
    step = int(payload.get("step", -1))

    assert step == 4
    assert step <= 5
    assert step % int(cfg.env.vec_envs) == 0


def test_intrinsic_not_masked_on_truncations(tmp_path: Path) -> None:
    out = run_train(_make_timeout_mask_cfg(), total_steps=4, run_dir=tmp_path / "run_timeout", resume=False)

    csv_path = out / "logs" / "scalars.csv"
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_step: dict[int, dict[str, str]] = {}
    for r in rows:
        try:
            by_step[int(r["step"])] = r
        except Exception:
            continue

    r2 = float(by_step[2]["r_int_mean"])
    r4 = float(by_step[4]["r_int_mean"])

    assert np.isfinite(r2) and r2 > 0.0
    assert abs(r4) < 1e-12


def test_evaluator_writes_trajectory_npz_and_gate_source(tmp_path: Path) -> None:
    def _write_ckpt(*, method: str, seed: int, include_intrinsic: bool) -> Path:
        obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        act_space = gym.spaces.Discrete(2)

        torch.manual_seed(0)
        policy = PolicyNetwork(obs_space, act_space)

        cfg = {
            "seed": int(seed),
            "method": str(method),
            "env": {
                "id": "DummyTraj-v0",
                "frame_skip": 1,
                "discrete_actions": True,
                "car_discrete_action_set": None,
            },
        }

        payload: dict[str, object] = {"step": 0, "policy": policy.state_dict(), "cfg": cfg, "obs_norm": None}

        if include_intrinsic:
            mod = create_intrinsic_module(
                str(method),
                obs_space,
                act_space,
                device="cpu",
                **build_intrinsic_kwargs(cfg),
            )
            payload["intrinsic"] = {"method": str(method), "state_dict": mod.state_dict()}

        ckpt_path = tmp_path / f"ckpt_{method}.pt"
        torch.save(payload, ckpt_path)
        return ckpt_path

    out_v = tmp_path / "vanilla_out"
    ckpt_v = _write_ckpt(method="vanilla", seed=123, include_intrinsic=False)
    _ = evaluate(
        env="DummyTraj-v0",
        ckpt=ckpt_v,
        episodes=1,
        device="cpu",
        save_traj=True,
        traj_out_dir=out_v,
        policy_mode="mode",
    )
    traj_v = out_v / "DummyTraj-v0_trajectory.npz"
    assert traj_v.exists()

    d_v = np.load(traj_v, allow_pickle=False)
    assert set(d_v.files) == {"obs", "gates", "intrinsic", "env_id", "method", "gate_source"}
    assert str(d_v["method"].reshape(-1)[0]) == "vanilla"
    assert str(d_v["gate_source"].reshape(-1)[0]) == "n/a"

    out_g = tmp_path / "glpe_out"
    ckpt_g = _write_ckpt(method="glpe", seed=7, include_intrinsic=True)
    _ = evaluate(
        env="DummyTraj-v0",
        ckpt=ckpt_g,
        episodes=1,
        device="cpu",
        save_traj=True,
        traj_out_dir=out_g,
        policy_mode="mode",
    )
    traj_g = out_g / "DummyTraj-v0_trajectory.npz"
    assert traj_g.exists()

    d_g = np.load(traj_g, allow_pickle=False)
    assert str(d_g["method"].reshape(-1)[0]) == "glpe"
    assert str(d_g["gate_source"].reshape(-1)[0]) == "checkpoint"


def test_apply_final_observation_handles_vector_and_scalar():
    next_obs = np.array([[0.0], [1.0]], dtype=np.float32)
    done = np.array([True, False], dtype=bool)
    infos = {
        "final_observation": np.array([np.array([42.0], dtype=np.float32), None], dtype=object),
    }
    fixed = _apply_final_observation(next_obs, done, infos)
    assert np.allclose(fixed, np.array([[42.0], [1.0]], dtype=np.float32))

    next_obs_1 = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    done_1 = np.array([True], dtype=bool)
    infos_1 = {"final_observation": np.array([10.0, 11.0, 12.0], dtype=np.float32)}
    fixed_1 = _apply_final_observation(next_obs_1, done_1, infos_1)
    assert np.allclose(fixed_1, np.array([10.0, 11.0, 12.0], dtype=np.float32))


def test_frameskip_accumulates_reward_and_stops():
    env = _FrameSkipEnv()
    try:
        env = FrameSkip(env, skip=2)
        obs, _ = env.reset()
        assert obs.shape == (1,)

        obs1, r1, term1, trunc1, _ = env.step(0)
        assert np.isclose(r1, 2.0)
        assert not term1 and not trunc1
        assert np.isclose(obs1[0], 2.0)

        obs2, r2, term2, trunc2, _ = env.step(1)
        assert np.isclose(r2, 1.0)
        assert term2 and not trunc2
        assert np.isclose(obs2[0], 3.0)
    finally:
        env.close()


def test_evaluator_is_repeatable_with_fixed_seed(tmp_path: Path):
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)

    torch.manual_seed(0)
    policy = PolicyNetwork(obs_space, act_space)

    ckpt_path = tmp_path / "ckpt_eval_repeatable.pt"
    torch.save(
        {
            "step": 0,
            "policy": policy.state_dict(),
            "cfg": {"env": {"id": "DummyEval-v0"}, "seed": 321},
            "obs_norm": None,
        },
        ckpt_path,
    )

    s1 = evaluate(env="DummyEval-v0", ckpt=ckpt_path, episodes=2, device="cpu")
    s2 = evaluate(env="DummyEval-v0", ckpt=ckpt_path, episodes=2, device="cpu")
    assert s1["returns"] == s2["returns"]
    assert s1["lengths"] == s2["lengths"]


def test_kdtree_bulk_insert_matches_sequential_and_dedup():
    rng = np.random.default_rng(0)
    dim = 3
    pts = rng.standard_normal((100, dim)).astype(np.float32)

    store_seq = KDTreeRegionStore(dim=dim, capacity=4, depth_max=6)
    rids_seq = np.array([store_seq.insert(p) for p in pts], dtype=np.int64)

    store_bulk = KDTreeRegionStore(dim=dim, capacity=4, depth_max=6)
    rids_bulk = store_bulk.bulk_insert(pts)

    assert np.all(rids_seq == rids_bulk)
    assert store_seq.num_regions() == store_bulk.num_regions()

    store = KDTreeRegionStore(dim=3, capacity=2, depth_max=4)
    rids = store.bulk_insert(np.zeros((5, 3), dtype=np.float32))
    assert np.all(rids == 0)
    assert store.num_regions() == 1
