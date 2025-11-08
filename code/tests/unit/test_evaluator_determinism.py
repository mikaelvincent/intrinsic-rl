from pathlib import Path

import torch

from irl.evaluator import evaluate
from irl.envs import EnvManager
from irl.models import PolicyNetwork, ValueNetwork


def _make_dummy_ckpt(tmp_path: Path, seed: int = 123) -> Path:
    # Build spaces from a single MountainCar env to ensure matching network shapes
    m = EnvManager(env_id="MountainCar-v0", num_envs=1, seed=seed)
    env = m.make()
    try:
        obs_space = getattr(env, "single_observation_space", None) or env.observation_space
        act_space = getattr(env, "single_action_space", None) or env.action_space
    finally:
        env.close()

    # Create a compatible policy/value, snapshot weights into a dummy checkpoint
    policy = PolicyNetwork(obs_space, act_space)
    value = ValueNetwork(obs_space)

    payload = {
        "step": 0,
        "policy": policy.state_dict(),
        "value": value.state_dict(),
        "cfg": {
            "env": {"id": "MountainCar-v0"},
            "seed": int(seed),
        },
        "obs_norm": None,
        "intrinsic_norm": {"r2_ema": 1.0, "beta": 0.99, "eps": 1e-8},
    }
    ckpt_path = tmp_path / "ckpt_eval_determinism.pt"
    torch.save(payload, ckpt_path)
    return ckpt_path


def test_evaluator_is_repeatable_with_same_seed(tmp_path):
    ckpt = _make_dummy_ckpt(tmp_path, seed=321)

    # Two back-to-back evaluations with identical inputs
    s1 = evaluate(env="MountainCar-v0", ckpt=ckpt, episodes=2, device="cpu")
    s2 = evaluate(env="MountainCar-v0", ckpt=ckpt, episodes=2, device="cpu")

    # Expect identical trajectories and aggregates
    assert s1["returns"] == s2["returns"]
    assert s1["lengths"] == s2["lengths"]
    assert s1["mean_return"] == s2["mean_return"]
    assert s1["mean_length"] == s2["mean_length"]
