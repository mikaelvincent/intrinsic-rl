from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import register

from irl.envs.manager import EnvManager
from irl.intrinsic.regions.kdtree import KDTreeRegionStore
from irl.trainer.runtime_utils import _apply_final_observation
from irl.utils.checkpoint import CheckpointManager


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

    def reset(self, *, seed=None, options=None):
        _ = seed, options
        return np.zeros((4,), dtype=np.float32), {}

    def step(self, action):
        _ = action
        return np.zeros((4,), dtype=np.float32), 0.0, True, False, {}

    def close(self) -> None:
        return


try:
    register(id="CarRacingLikeStrict-v0", entry_point=_CarRacingLikeEnv)
except Exception:
    pass


def test_checkpoint_manager_prune_keeps_step0() -> None:
    def _payload(step: int) -> dict:
        return {"step": int(step), "meta": {"note": "test"}}

    with TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        cm = CheckpointManager(run_dir, interval_steps=10, max_to_keep=2)
        for step in (0, 10, 20, 30):
            cm.save(step=step, payload=_payload(step))

        kept = sorted(p.name for p in (run_dir / "checkpoints").glob("ckpt_step_*.pt"))
        assert "ckpt_step_0.pt" in kept
        assert "ckpt_step_30.pt" in kept
        assert "ckpt_step_20.pt" in kept
        assert "ckpt_step_10.pt" not in kept


def test_kdtree_bulk_insert_matches_sequential_and_dedup() -> None:
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((100, 3)).astype(np.float32)

    store_seq = KDTreeRegionStore(dim=3, capacity=4, depth_max=6)
    rids_seq = np.array([store_seq.insert(p) for p in pts], dtype=np.int64)

    store_bulk = KDTreeRegionStore(dim=3, capacity=4, depth_max=6)
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
