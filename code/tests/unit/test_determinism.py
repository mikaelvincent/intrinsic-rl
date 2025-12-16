from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import register

from irl.evaluator import evaluate
from irl.envs import EnvManager
from irl.models import PolicyNetwork
from irl.trainer.build import single_spaces


class _DummyEvalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        obs = self._rng.standard_normal(self.observation_space.shape).astype(np.float32)
        return obs, {}

    def step(self, action):
        self._t += 1
        obs = self._rng.standard_normal(self.observation_space.shape).astype(np.float32)
        reward = float(action)
        terminated = self._t >= 5
        truncated = False
        return obs, reward, terminated, truncated, {}


try:
    register(id="DummyEval-v0", entry_point=_DummyEvalEnv)
except Exception:
    pass


def _make_dummy_ckpt(tmp_path: Path, seed: int) -> Path:
    env = EnvManager(env_id="DummyEval-v0", num_envs=1, seed=seed).make()
    try:
        obs_space, act_space = single_spaces(env)
    finally:
        env.close()

    torch.manual_seed(0)
    policy = PolicyNetwork(obs_space, act_space)

    payload = {
        "step": 0,
        "policy": policy.state_dict(),
        "cfg": {"env": {"id": "DummyEval-v0"}, "seed": int(seed)},
        "obs_norm": None,
    }
    ckpt_path = tmp_path / "ckpt_eval_determinism.pt"
    torch.save(payload, ckpt_path)
    return ckpt_path


def test_evaluator_is_repeatable_with_same_seed(tmp_path: Path):
    ckpt = _make_dummy_ckpt(tmp_path, seed=321)
    s1 = evaluate(env="DummyEval-v0", ckpt=ckpt, episodes=2, device="cpu")
    s2 = evaluate(env="DummyEval-v0", ckpt=ckpt, episodes=2, device="cpu")
    assert s1["returns"] == s2["returns"]
    assert s1["lengths"] == s2["lengths"]
