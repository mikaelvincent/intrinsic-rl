import random
from pathlib import Path

import numpy as np
import torch

from irl.evaluator import evaluate
from irl.envs import EnvManager
from irl.models import PolicyNetwork
from irl.utils.determinism import seed_everything


def test_seed_everything_produces_repeatable_streams():
    seed_everything(123, deterministic=True)
    py_1 = [random.random() for _ in range(3)]
    np_1 = np.random.rand(3).tolist()
    t_1 = torch.rand(3).tolist()

    seed_everything(123, deterministic=True)
    py_2 = [random.random() for _ in range(3)]
    np_2 = np.random.rand(3).tolist()
    t_2 = torch.rand(3).tolist()

    assert py_1 == py_2
    assert np.allclose(np_1, np_2)
    assert np.allclose(t_1, t_2)


def _make_dummy_ckpt(tmp_path: Path, seed: int) -> Path:
    env = EnvManager(env_id="MountainCar-v0", num_envs=1, seed=seed).make()
    try:
        obs_space = getattr(env, "single_observation_space", None) or env.observation_space
        act_space = getattr(env, "single_action_space", None) or env.action_space
    finally:
        env.close()

    policy = PolicyNetwork(obs_space, act_space)

    payload = {
        "step": 0,
        "policy": policy.state_dict(),
        "cfg": {"env": {"id": "MountainCar-v0"}, "seed": int(seed)},
        "obs_norm": None,
    }
    ckpt_path = tmp_path / "ckpt_eval_determinism.pt"
    torch.save(payload, ckpt_path)
    return ckpt_path


def test_evaluator_is_repeatable_with_same_seed(tmp_path: Path):
    ckpt = _make_dummy_ckpt(tmp_path, seed=321)
    s1 = evaluate(env="MountainCar-v0", ckpt=ckpt, episodes=2, device="cpu")
    s2 = evaluate(env="MountainCar-v0", ckpt=ckpt, episodes=2, device="cpu")
    assert s1["returns"] == s2["returns"]
    assert s1["lengths"] == s2["lengths"]
