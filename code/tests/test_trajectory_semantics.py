from __future__ import annotations

import hashlib
import pickle

import gymnasium as gym
import numpy as np
import torch

from irl.evaluation.rollout import run_eval_episodes
from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.riac import RIAC
from irl.models.networks import PolicyNetwork


class _TinyEvalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._t = 0
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed=None, options=None):
        _ = options
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._t = 0
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(4,)).astype(np.float32)
        return obs, {}

    def step(self, action):
        _ = action
        self._t += 1
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(4,)).astype(np.float32)
        terminated = self._t >= 3
        return obs, 0.0, bool(terminated), False, {}

    def close(self) -> None:
        return


def _module_digest(mod: torch.nn.Module) -> str:
    sd = mod.state_dict()
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        h.update(str(k).encode("utf-8"))
        v = sd[k]
        if torch.is_tensor(v):
            t = v.detach().cpu()
            h.update(str(tuple(t.shape)).encode("utf-8"))
            h.update(str(t.dtype).encode("utf-8"))
            h.update(t.numpy().tobytes())
        else:
            h.update(pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL))
    return h.hexdigest()


def test_riac_eval_trajectory_does_not_mutate_state() -> None:
    torch.manual_seed(0)

    env = _TinyEvalEnv()
    try:
        obs_space = env.observation_space
        act_space = env.action_space
        device = torch.device("cpu")

        policy = PolicyNetwork(obs_space, act_space).to(device)
        policy.eval()

        riac = RIAC(
            obs_space,
            act_space,
            device="cpu",
            icm_cfg=ICMConfig(phi_dim=8, hidden=(16, 16)),
            region_capacity=64,
            depth_max=4,
            alpha_lp=0.5,
            checkpoint_include_points=True,
        )
        riac.eval()

        rng = np.random.default_rng(0)
        obs = rng.standard_normal((32, 4)).astype(np.float32)
        next_obs = rng.standard_normal((32, 4)).astype(np.float32)
        actions = rng.integers(0, int(act_space.n), size=(32,), endpoint=False, dtype=np.int64)
        _ = riac.compute_batch(obs, next_obs, actions, reduction="none")

        before = _module_digest(riac)

        def _norm(x: np.ndarray) -> np.ndarray:
            return np.asarray(x, dtype=np.float32)

        rr = run_eval_episodes(
            env=env,
            policy=policy,
            act_space=act_space,
            device=device,
            policy_mode="mode",
            episode_seeds=[123],
            normalize_obs=_norm,
            save_traj=True,
            is_image=False,
            intrinsic_module=riac,
            method="riac",
        )

        after = _module_digest(riac)
        assert before == after

        assert rr.trajectory is not None
        assert rr.trajectory.intrinsic_semantics == "frozen_checkpoint"
    finally:
        env.close()
