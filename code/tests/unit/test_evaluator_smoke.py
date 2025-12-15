import torch

from irl.evaluator import evaluate
from irl.envs import EnvManager
from irl.models import PolicyNetwork, ValueNetwork


def test_evaluator_runs_one_episode(tmp_path):
    m = EnvManager(env_id="MountainCar-v0", num_envs=1, seed=123)
    env = m.make()
    try:
        obs_space = getattr(env, "single_observation_space", None) or env.observation_space
        act_space = getattr(env, "single_action_space", None) or env.action_space
    finally:
        env.close()

    policy = PolicyNetwork(obs_space, act_space)
    value = ValueNetwork(obs_space)

    payload = {
        "step": 0,
        "policy": policy.state_dict(),
        "value": value.state_dict(),
        "cfg": {"env": {"id": "MountainCar-v0"}, "seed": 1},
        "obs_norm": None,
        "intrinsic_norm": {"r2_ema": 1.0, "beta": 0.99, "eps": 1e-8},
    }
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(payload, ckpt_path)

    summary = evaluate(env="MountainCar-v0", ckpt=ckpt_path, episodes=1, device="cpu")
    assert summary["env_id"] == "MountainCar-v0"
    assert summary["episodes"] == 1
    assert isinstance(summary["mean_return"], float)
    assert isinstance(summary["mean_length"], float)
    assert len(summary["returns"]) == 1
    assert len(summary["lengths"]) == 1
