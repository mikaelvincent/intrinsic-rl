from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import register

from irl.evaluator import evaluate
from irl.intrinsic.config import build_intrinsic_kwargs
from irl.intrinsic.factory import create_intrinsic_module
from irl.models import PolicyNetwork


class _DummyTrajEnv(gym.Env):
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


def _write_ckpt(tmp_path: Path, *, method: str, seed: int, include_intrinsic: bool) -> Path:
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

    payload: dict[str, object] = {
        "step": 0,
        "policy": policy.state_dict(),
        "cfg": cfg,
        "obs_norm": None,
    }

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


def test_evaluator_saves_trajectory_npz_vanilla(tmp_path: Path) -> None:
    ckpt = _write_ckpt(tmp_path, method="vanilla", seed=123, include_intrinsic=False)
    out_dir = tmp_path / "vanilla_out"
    _ = evaluate(
        env="DummyTraj-v0",
        ckpt=ckpt,
        episodes=1,
        device="cpu",
        save_traj=True,
        traj_out_dir=out_dir,
        policy_mode="mode",
    )

    traj_path = out_dir / "DummyTraj-v0_trajectory.npz"
    assert traj_path.exists()

    data = np.load(traj_path, allow_pickle=False)
    assert set(data.files) == {"obs", "gates", "intrinsic", "env_id", "method", "gate_source"}

    assert str(data["env_id"].reshape(-1)[0]) == "DummyTraj-v0"
    assert str(data["method"].reshape(-1)[0]) == "vanilla"
    assert str(data["gate_source"].reshape(-1)[0]) == "n/a"


def test_evaluator_gate_source_checkpoint_for_glpe(tmp_path: Path) -> None:
    ckpt = _write_ckpt(tmp_path, method="glpe", seed=7, include_intrinsic=True)
    out_dir = tmp_path / "glpe_out"
    _ = evaluate(
        env="DummyTraj-v0",
        ckpt=ckpt,
        episodes=1,
        device="cpu",
        save_traj=True,
        traj_out_dir=out_dir,
        policy_mode="mode",
    )

    traj_path = out_dir / "DummyTraj-v0_trajectory.npz"
    assert traj_path.exists()

    data = np.load(traj_path, allow_pickle=False)
    assert set(data.files) == {"obs", "gates", "intrinsic", "env_id", "method", "gate_source"}

    assert str(data["method"].reshape(-1)[0]) == "glpe"
    assert str(data["gate_source"].reshape(-1)[0]) == "checkpoint"
