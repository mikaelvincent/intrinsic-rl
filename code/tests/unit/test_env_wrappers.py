from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

import irl.evaluator as evaluator
import irl.video as video
from irl.envs.wrappers import CarRacingDiscreteActionWrapper, FrameSkip
from irl.models import PolicyNetwork
from irl.trainer.runtime_utils import _apply_final_observation


class _DummyCarEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        obs = self.observation_space.sample()
        return obs, 0.0, False, False, {}


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
        self.t += 1
        obs = np.array([float(self.t)], dtype=np.float32)
        reward = 1.0
        terminated = self.t >= 3
        truncated = False
        info = {"t": self.t}
        return obs, reward, terminated, truncated, info


class _RuntimeDummyEnv:
    metadata = {"render_modes": []}

    def __init__(self, obs_space: gym.Space, act_space: gym.Space) -> None:
        self.observation_space = obs_space
        self.action_space = act_space
        self._t = 0

    def reset(self, *, seed=None, options=None):
        self._t = 0
        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        return obs, {}

    def step(self, action):
        self._t += 1
        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        reward = 0.0
        terminated = self._t >= 1
        truncated = False
        return obs, reward, terminated, truncated, {}

    def render(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def close(self) -> None:
        return


def _make_fake_make_env(calls: list[dict[str, object]]):
    def fake_make_env(
        env_id: str,
        *,
        num_envs: int = 1,
        seed: int | None = 1,
        frame_skip: int = 1,
        domain_randomization: bool = False,
        discrete_actions: bool = True,
        car_action_set: object | None = None,
        render_mode: str | None = None,
        async_vector: bool = False,
        make_kwargs: dict | None = None,
    ):
        calls.append(
            {
                "env_id": str(env_id),
                "num_envs": int(num_envs),
                "seed": None if seed is None else int(seed),
                "frame_skip": int(frame_skip),
                "domain_randomization": bool(domain_randomization),
                "discrete_actions": bool(discrete_actions),
                "car_action_set": car_action_set,
                "render_mode": render_mode,
                "async_vector": bool(async_vector),
                "make_kwargs": make_kwargs,
            }
        )

        obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        act_space = (
            gym.spaces.Discrete(2)
            if bool(discrete_actions)
            else gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        )
        return _RuntimeDummyEnv(obs_space, act_space)

    return fake_make_env


def test_carracing_action_wrapper_maps_actions():
    env = _DummyCarEnv()
    try:
        wrapped = CarRacingDiscreteActionWrapper(env)
        assert wrapped.action_space.n == 5
    finally:
        wrapped.close()

    custom = [
        [0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
    ]
    env = _DummyCarEnv()
    try:
        wrapped = CarRacingDiscreteActionWrapper(env, action_set=custom)
        assert wrapped.action_space.n == len(custom)
        for i, row in enumerate(custom):
            assert np.allclose(wrapped.action(i), np.asarray(row, dtype=np.float32))
        with pytest.raises(ValueError):
            wrapped.action(len(custom))
    finally:
        wrapped.close()


def test_frameskip_accumulates_and_stops_early():
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


def test_apply_final_observation_variants():
    cases = [
        (
            np.array([[0.0], [1.0]], dtype=np.float32),
            np.array([True, False], dtype=bool),
            {"final_observation": np.array([np.array([42.0], dtype=np.float32), None], dtype=object)},
            np.array([[42.0], [1.0]], dtype=np.float32),
        ),
        (
            np.array([0.0], dtype=np.float32),
            np.array([True], dtype=bool),
            {"final_observation": np.array([123.0], dtype=np.float32)},
            np.array([123.0], dtype=np.float32),
        ),
        (
            np.array([0.0, 1.0, 2.0], dtype=np.float32),
            np.array([True], dtype=bool),
            {"final_observation": np.array([10.0, 11.0, 12.0], dtype=np.float32)},
            np.array([10.0, 11.0, 12.0], dtype=np.float32),
        ),
    ]

    for next_obs, done, infos, expected in cases:
        orig = np.array(next_obs, copy=True)
        fixed = _apply_final_observation(next_obs, done, infos)
        assert fixed.shape == orig.shape
        assert np.allclose(fixed, expected)
        assert np.allclose(next_obs, orig)


def test_evaluator_and_video_propagate_env_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video, "ensure_mujoco_gl", lambda *_a, **_k: "")
    monkeypatch.setattr(video.imageio, "mimsave", lambda *_a, **_k: None)
    monkeypatch.setattr(video, "_add_label", lambda frame, *_a, **_k: frame)

    env_id = "DummyRuntime-v0"
    cfg_seed = 123
    eval_seed = 999
    frame_skip = 4
    car_action_set = [[0.0, 0.0, 0.0]]

    for discrete_actions in (True, False):
        obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        act_space = (
            gym.spaces.Discrete(2)
            if discrete_actions
            else gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        )
        policy = PolicyNetwork(obs_space, act_space)

        ckpt_path = tmp_path / f"ckpt_{'disc' if discrete_actions else 'cont'}.pt"
        torch.save(
            {
                "step": 0,
                "policy": policy.state_dict(),
                "cfg": {
                    "seed": int(cfg_seed),
                    "method": "vanilla",
                    "env": {
                        "id": str(env_id),
                        "frame_skip": int(frame_skip),
                        "discrete_actions": bool(discrete_actions),
                        "car_discrete_action_set": car_action_set,
                    },
                },
                "obs_norm": None,
            },
            ckpt_path,
        )

        eval_calls: list[dict[str, object]] = []
        vid_calls: list[dict[str, object]] = []

        monkeypatch.setattr(evaluator, "make_env", _make_fake_make_env(eval_calls))
        monkeypatch.setattr(video, "make_env", _make_fake_make_env(vid_calls))

        _ = evaluator.evaluate(env=str(env_id), ckpt=ckpt_path, episodes=1, device="cpu", policy_mode="mode")

        video.render_rollout_video(
            ckpt_path=ckpt_path,
            out_path=tmp_path / "out.mp4",
            seed=int(eval_seed),
            max_steps=1,
            device="cpu",
            policy_mode="mode",
            fps=1,
        )

        assert len(eval_calls) == 1, discrete_actions
        assert len(vid_calls) == 1, discrete_actions

        e = eval_calls[0]
        v = vid_calls[0]

        assert e["env_id"] == str(env_id)
        assert v["env_id"] == str(env_id)

        assert e["frame_skip"] == int(frame_skip)
        assert v["frame_skip"] == int(frame_skip)

        assert e["discrete_actions"] is bool(discrete_actions)
        assert v["discrete_actions"] is bool(discrete_actions)

        assert e["car_action_set"] == car_action_set
        assert v["car_action_set"] == car_action_set

        assert e["domain_randomization"] is False
        assert v["domain_randomization"] is False

        assert e["num_envs"] == 1
        assert v["num_envs"] == 1

        assert e["render_mode"] is None
        assert v["render_mode"] == "rgb_array"

        assert e["seed"] == int(cfg_seed)
        assert v["seed"] == int(eval_seed)
