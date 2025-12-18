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
from torch.nn import functional as F
from torch.optim import Adam

from irl.algo.advantage import compute_gae
from irl.algo.ppo import ppo_update
from irl.cfg import Config, ConfigError, loads_config, validate_config
from irl.cfg.schema import PPOConfig
from irl.envs.manager import EnvManager
from irl.envs.wrappers import DomainRandomizationWrapper, FrameSkip
from irl.evaluator import evaluate
from irl.experiments.evaluation import run_eval_suite
from irl.intrinsic.config import build_intrinsic_kwargs
from irl.intrinsic.factory import create_intrinsic_module
from irl.intrinsic.glpe import GLPE
from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.regions.kdtree import KDTreeRegionStore
from irl.intrinsic.riac import RIAC
from irl.intrinsic.ride import RIDE
from irl.intrinsic.rnd import RND, RNDConfig
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
        _ = seed, options
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
        _ = seed, options
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
        return obs, reward, bool(terminated), bool(truncated), info

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

    def close(self) -> None:
        return


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

    def close(self) -> None:
        return


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

    def close(self) -> None:
        return


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

    def close(self) -> None:
        return


class _ObsNormCountEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        _ = options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(3,)).astype(np.float32)
        return obs, {}

    def step(self, action):
        _ = action
        self._t += 1
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(3,)).astype(np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def close(self) -> None:
        return


class _DummyImageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, h: int = 32, w: int = 32, seed: int | None = None) -> None:
        super().__init__()
        self.H, self.W = int(h), int(w)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(3)
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        _ = options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        obs = self._rng.integers(0, 256, size=(self.H, self.W, 3), dtype=np.uint8)
        return obs, {}

    def step(self, action):
        _ = action
        self._t += 1
        obs = self._rng.integers(0, 256, size=(self.H, self.W, 3), dtype=np.uint8)
        reward = 0.1
        terminated = self._t >= 5
        truncated = False
        return obs, reward, bool(terminated), bool(truncated), {}

    def close(self) -> None:
        return


class _CarRacingLikeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._t = 0

    def reset(self, *, seed=None, options=None):
        _ = seed, options
        self._t = 0
        return np.zeros((4,), dtype=np.float32), {}

    def step(self, action):
        _ = action
        self._t += 1
        obs = np.zeros((4,), dtype=np.float32)
        return obs, 0.0, True, False, {}

    def close(self) -> None:
        return


class _DummyMujocoLikeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

        class _Model:
            pass

        class _Opt:
            pass

        self.model = _Model()
        self.model.opt = _Opt()
        self.model.opt.gravity = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self.model.geom_friction = np.ones((4, 3), dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        _ = options
        if seed is not None:
            np.random.seed(seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        _ = action
        obs = self.observation_space.sample()
        return obs, 0.0, False, False, {}

    def close(self) -> None:
        return


for _id, _cls in (
    ("DummyEval-v0", _DummyEvalEnv),
    ("DummyTraj-v0", _DummyTrajEnv),
    ("StepBudget-v0", _StepBudgetEnv),
    ("TimeoutMask-v0", _TimeoutMaskEnv),
    ("ObsNormCount-v0", _ObsNormCountEnv),
    ("DummyImage-v0", _DummyImageEnv),
    ("CarRacingLikeStrict-v0", _CarRacingLikeEnv),
):
    try:
        register(id=_id, entry_point=_cls)
    except Exception:
        pass


def _read_csv_column(path: Path, col: str) -> list[float]:
    out: list[float] = []
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or col not in set(r.fieldnames):
            return []
        for row in r:
            try:
                out.append(float(row[col]))
            except Exception:
                continue
    return out


def _make_cfg(
    *,
    env_id: str,
    method: str,
    vec_envs: int,
    steps_per_update: int,
    minibatches: int,
    epochs: int,
    eta: float,
) -> Config:
    base = Config()
    env_cfg = replace(
        base.env,
        id=str(env_id),
        vec_envs=int(vec_envs),
        frame_skip=1,
        domain_randomization=False,
        discrete_actions=True,
        async_vector=False,
    )
    ppo_cfg = replace(
        base.ppo,
        steps_per_update=int(steps_per_update),
        minibatches=int(minibatches),
        epochs=int(epochs),
        entropy_coef=0.0,
    )
    intrinsic_cfg = replace(base.intrinsic, eta=float(eta))
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


def test_loads_config_parses_numbers_and_validates() -> None:
    cfg = loads_config(
        """
method: vanilla
env:
  vec_envs: "8"
ppo:
  steps_per_update: "2_048"
  minibatches: 3_2
  learning_rate: "3e-4"
""".lstrip()
    )
    assert int(cfg.env.vec_envs) == 8
    assert int(cfg.ppo.steps_per_update) == 2048
    assert int(cfg.ppo.minibatches) == 32
    assert abs(float(cfg.ppo.learning_rate) - 3e-4) < 1e-12

    with pytest.raises(ConfigError):
        loads_config("seed: true\n")

    with pytest.raises(ConfigError):
        loads_config("seed: 1\nunknown_top_level: 123\n")

    with pytest.raises(ConfigError):
        loads_config(
            """
method: vanilla
env:
  vec_envs: 8
ppo:
  steps_per_update: 130
  minibatches: 64
""".lstrip()
        )


def test_compute_gae_bootstraps_timeouts_and_dones() -> None:
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


def test_compute_gae_does_not_leak_across_truncation() -> None:
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


def _flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().cpu().view(-1) for p in model.parameters()])


def test_ppo_update_kl_penalty_changes_policy() -> None:
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
    assert np.isfinite(float(stats["approx_kl"]))

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

    diff = float((_flat_params(policy_a) - _flat_params(policy_b)).abs().max().item())
    assert diff > 1e-7


def test_policy_rollout_continuous_actions_within_bounds() -> None:
    env = _BoundedBoxEnv()
    try:
        policy = PolicyNetwork(env.observation_space, env.action_space).to(torch.device("cpu"))
        policy.eval()

        for mode in ("mode", "sample"):
            obs0, _ = env.reset(seed=0)
            for _step in iter_policy_rollout(
                env=env,
                policy=policy,
                obs0=obs0,
                act_space=env.action_space,
                device=torch.device("cpu"),
                policy_mode=mode,
                normalize_obs=None,
                max_steps=5,
            ):
                pass
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


def test_kdtree_bulk_insert_matches_sequential_and_dedup() -> None:
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


def test_apply_final_observation_handles_vector_and_scalar() -> None:
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


def test_frameskip_accumulates_reward_and_stops() -> None:
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


def test_evaluator_is_repeatable_with_fixed_seed(tmp_path: Path) -> None:
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
        store = getattr(mod, "store")
        for p in pts:
            store.insert(p)

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
        assert size_without <= int(size_with * 0.6)

        mod2 = make_module(include_points=False)
        mod2.load_state_dict(_torch_load_any(p_without), strict=True)

        assert int(mod2.store.num_regions()) == int(mod.store.num_regions())
        assert int(mod2.store.depth_max) == 0


def test_train_refuses_dirty_run_dir_without_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dirty"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "ckpt_latest.pt").write_bytes(b"not a real checkpoint")
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / "scalars.csv").write_text("step,reward_total_mean\n0,0.0\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="resume"):
        run_train(Config(), total_steps=1, run_dir=run_dir, resume=False)


def test_total_steps_aligns_to_vec_envs_budget(tmp_path: Path) -> None:
    cfg = _make_cfg(
        env_id="StepBudget-v0",
        method="vanilla",
        vec_envs=2,
        steps_per_update=2,
        minibatches=1,
        epochs=1,
        eta=0.0,
    )
    out_dir = run_train(cfg, total_steps=5, run_dir=tmp_path / "run_budget", resume=False)

    payload = load_checkpoint(out_dir / "checkpoints" / "ckpt_latest.pt", map_location="cpu")
    step = int(payload.get("step", -1))

    assert step == 4
    assert step <= 5
    assert step % int(cfg.env.vec_envs) == 0


def test_obs_norm_counts_once_per_transition(tmp_path: Path) -> None:
    cfg = _make_cfg(
        env_id="ObsNormCount-v0",
        method="vanilla",
        vec_envs=2,
        steps_per_update=2,
        minibatches=1,
        epochs=1,
        eta=0.0,
    )
    out_dir = run_train(cfg, total_steps=4, run_dir=tmp_path / "run_obs_norm_count", resume=False)

    payload = load_checkpoint(out_dir / "checkpoints" / "ckpt_latest.pt", map_location="cpu")
    step = int(payload.get("step", -1))
    assert step == 4

    obs_norm = payload.get("obs_norm")
    assert isinstance(obs_norm, dict)

    count = float(obs_norm.get("count", float("nan")))
    assert np.isfinite(count)
    assert abs(count - float(step)) < 1e-6


def test_intrinsic_not_masked_on_truncations(tmp_path: Path) -> None:
    cfg = _make_cfg(
        env_id="TimeoutMask-v0",
        method="ride",
        vec_envs=2,
        steps_per_update=1,
        minibatches=1,
        epochs=1,
        eta=0.1,
    )
    out_dir = run_train(cfg, total_steps=4, run_dir=tmp_path / "run_timeout", resume=False)

    csv_path = out_dir / "logs" / "scalars.csv"
    rows = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows[int(r["step"])] = r
            except Exception:
                continue

    r2 = float(rows[2]["r_int_mean"])
    r4 = float(rows[4]["r_int_mean"])

    assert np.isfinite(r2) and r2 > 0.0
    assert abs(r4) < 1e-12


def test_trainer_image_pipeline_riac_logs_intrinsic(tmp_path: Path) -> None:
    cfg = _make_cfg(
        env_id="DummyImage-v0",
        method="riac",
        vec_envs=1,
        steps_per_update=6,
        minibatches=1,
        epochs=1,
        eta=0.1,
    )
    out_dir = run_train(cfg, total_steps=6, run_dir=tmp_path / "run_riac_img", resume=False)

    csv_path = out_dir / "logs" / "scalars.csv"
    vals = _read_csv_column(csv_path, "r_int_mean")
    assert vals and any(np.isfinite(v) for v in vals)


def _write_latest_ckpt(run_dir: Path, *, env_id: str, method: str, seed: int, step: int) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {"step": int(step), "cfg": {"env": {"id": str(env_id)}, "method": str(method), "seed": int(seed)}}
    torch.save(payload, ckpt_dir / "ckpt_latest.pt")


def _fake_evaluate(*, env: str, ckpt: Path, episodes: int, device: str, **kwargs) -> dict:
    _ = ckpt, device, kwargs
    return {
        "env_id": str(env),
        "episodes": int(episodes),
        "seed": 0,
        "mean_return": 1.0,
        "std_return": 0.0,
        "min_return": 1.0,
        "max_return": 1.0,
        "mean_length": 5.0,
        "std_length": 0.0,
        "returns": [1.0],
        "lengths": [5],
    }


def test_eval_suite_seed_coverage_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs_suite"
    results_dir = tmp_path / "results_suite"
    group = runs_root / "groupA"
    group.mkdir(parents=True, exist_ok=True)

    r_v1 = group / "vanilla__DummyEval-v0__seed1__cfgA"
    r_v2 = group / "vanilla__DummyEval-v0__seed2__cfgA"
    r_g1 = group / "glpe__DummyEval-v0__seed1__cfgA"
    for rd in (r_v1, r_v2, r_g1):
        rd.mkdir(parents=True, exist_ok=True)

    _write_latest_ckpt(r_v1, env_id="DummyEval-v0", method="vanilla", seed=1, step=100)
    _write_latest_ckpt(r_v2, env_id="DummyEval-v0", method="vanilla", seed=2, step=100)
    _write_latest_ckpt(r_g1, env_id="DummyEval-v0", method="glpe", seed=1, step=50)

    import irl.experiments.evaluation as eval_module

    monkeypatch.setattr(eval_module, "evaluate", _fake_evaluate)

    with pytest.raises(RuntimeError, match="Seed coverage mismatch"):
        run_eval_suite(
            runs_root=runs_root,
            results_dir=results_dir,
            episodes=1,
            device="cpu",
            policy_mode="mode",
        )

    cov_path = results_dir / "coverage.csv"
    assert cov_path.exists()
    with cov_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    glpe_rows = [r for r in rows if r["env_id"] == "DummyEval-v0" and r["method"] == "glpe"]
    assert len(glpe_rows) == 1
    assert glpe_rows[0]["missing_seeds"] == "2"


def test_eval_suite_step_parity_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs_suite"
    results_dir = tmp_path / "results_suite"
    runs_root.mkdir(parents=True, exist_ok=True)

    r_v1 = runs_root / "vanilla__DummyEval-v0__seed1__cfgA"
    r_g1 = runs_root / "glpe__DummyEval-v0__seed1__cfgA"
    for rd in (r_v1, r_g1):
        rd.mkdir(parents=True, exist_ok=True)

    _write_latest_ckpt(r_v1, env_id="DummyEval-v0", method="vanilla", seed=1, step=100_000)
    _write_latest_ckpt(r_g1, env_id="DummyEval-v0", method="glpe", seed=1, step=1_000)

    import irl.experiments.evaluation as eval_module

    monkeypatch.setattr(eval_module, "evaluate", _fake_evaluate)

    with pytest.raises(RuntimeError, match="Step parity mismatch"):
        run_eval_suite(
            runs_root=runs_root,
            results_dir=results_dir,
            episodes=1,
            device="cpu",
            policy_mode="mode",
        )


def test_domain_randomization_mujoco_stays_near_baseline() -> None:
    env = _DummyMujocoLikeEnv()
    try:
        wrapped = DomainRandomizationWrapper(env, seed=123)
        baseline = wrapped.unwrapped.model.opt.gravity.copy()

        unique_scales: set[float] = set()
        for _ in range(12):
            _, info = wrapped.reset()
            g = wrapped.unwrapped.model.opt.gravity
            ratio = g / baseline

            assert np.all(np.isfinite(ratio))
            assert np.all(ratio >= 0.95 - 1e-6)
            assert np.all(ratio <= 1.05 + 1e-6)

            assert isinstance(info, dict)
            diag = info.get("dr_applied")
            assert isinstance(diag, dict)

            unique_scales.add(round(float(ratio.reshape(-1)[0]), 3))

        assert len(unique_scales) > 1
    finally:
        env.close()


def test_env_manager_carracing_wrapper_failure_raises() -> None:
    mgr = EnvManager(
        env_id="CarRacingLikeStrict-v0",
        num_envs=1,
        seed=0,
        discrete_actions=True,
        car_action_set=[[0.0, 0.0]],
    )
    with pytest.raises(ValueError, match="car_action_set must have shape"):
        _ = mgr.make()


def test_build_intrinsic_kwargs_glpe_nogate_forces_disabled() -> None:
    kw = build_intrinsic_kwargs({"method": "glpe_nogate", "intrinsic": {"gate": {"enabled": True}}})
    assert kw["gating_enabled"] is False


def test_ride_binning_repeats_and_resets() -> None:
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)

    ride = RIDE(
        obs_space,
        act_space,
        device="cpu",
        icm_cfg=ICMConfig(phi_dim=16, hidden=(32, 32)),
        bin_size=0.5,
        alpha_impact=1.0,
    )

    o = np.zeros((1, 3), dtype=np.float32)
    op = np.ones((1, 3), dtype=np.float32)

    r1 = ride.compute_impact_binned(o, op, dones=np.array([False], dtype=bool)).item()
    r2 = ride.compute_impact_binned(o, op, dones=np.array([False], dtype=bool)).item()

    assert np.isfinite(r1) and r1 > 0.0
    assert np.isfinite(r2) and r2 < r1
    assert abs((r2 * 2.0) - r1) < 1e-4

    r3 = ride.compute_impact_binned(o, op, dones=np.array([True], dtype=bool)).item()
    assert abs(r3 - r1) < 1e-4


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
