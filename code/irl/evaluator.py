from __future__ import annotations

"""Programmatic policy evaluator (no intrinsic).

Runs deterministic evaluation episodes using the policy's *mode* action,
returns aggregated statistics as a plain dict that is easy to serialize.

Typical usage (from Python):
    from pathlib import Path
    from irl.evaluator import evaluate
    summary = evaluate(env="MountainCar-v0",
                       ckpt=Path("runs/.../checkpoints/ckpt_step_100000.pt"),
                       episodes=10,
                       device="cpu")
    print(summary["mean_return"], summary["std_return"])
"""

from pathlib import Path
from statistics import mean, pstdev
from typing import Optional, Tuple

import numpy as np
import torch

from irl.envs import EnvManager
from irl.models import PolicyNetwork
from irl.utils.checkpoint import load_checkpoint
from irl.utils.determinism import seed_everything  # NEW


def _single_spaces(env) -> Tuple:
    """Return (obs_space, action_space) for both single and vector envs."""
    obs_space = getattr(env, "single_observation_space", None) or env.observation_space
    act_space = getattr(env, "single_action_space", None) or env.action_space
    return obs_space, act_space


def _is_image_space(space) -> bool:
    """Heuristic: Box with rank >= 2 is treated as image."""
    return hasattr(space, "shape") and len(space.shape) >= 2


def _build_normalizer(payload) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return (mean, std) if vector obs_norm was persisted; images return None."""
    on = payload.get("obs_norm")
    if on is None:
        return None
    mean_arr = np.asarray(on.get("mean"), dtype=np.float64)
    var_arr = np.asarray(on.get("var"), dtype=np.float64)
    std_arr = np.sqrt(var_arr + 1e-8)
    return mean_arr, std_arr


def evaluate(
    *,
    env: str,
    ckpt: Path,
    episodes: int = 10,
    device: str = "cpu",
) -> dict:
    """Evaluate a saved policy deterministically (no intrinsic), returning stats.

    Parameters
    ----------
    env:
        Gymnasium environment id (e.g., "MountainCar-v0").
    ckpt:
        Path to a training checkpoint (produced by the trainer).
    episodes:
        Number of episodes to run.
    device:
        Torch device string, e.g. "cpu" or "cuda:0".

    Returns
    -------
    dict
        {
          "env_id": str,
          "episodes": int,
          "seed": int,
          "checkpoint_step": int,
          "mean_return": float,
          "std_return": float,
          "min_return": float,
          "max_return": float,
          "mean_length": float,
          "std_length": float,
          "returns": [floats],
          "lengths": [ints],
        }
    """
    payload = load_checkpoint(ckpt, map_location=device)
    cfg = payload.get("cfg", {}) or {}
    seed = int(cfg.get("seed", 1))
    step = int(payload.get("step", -1))

    # NEW: apply uniform seeding to maximize repeatability in tests/CI
    seed_everything(seed, deterministic=True)

    # Build a single environment (no vectorization) for evaluation
    manager = EnvManager(env_id=env, num_envs=1, seed=seed)
    e = manager.make()
    obs_space, act_space = _single_spaces(e)

    # Policy network (same architecture as training) and weights
    policy = PolicyNetwork(obs_space, act_space).to(device)
    policy.load_state_dict(payload["policy"])
    policy.eval()

    # Observation normalization for vector observations if stored
    is_image = _is_image_space(obs_space)
    norm = None if is_image else _build_normalizer(payload)

    def _normalize(x: np.ndarray) -> np.ndarray:
        if norm is None:
            return x
        mean_arr, std_arr = norm
        return (x - mean_arr) / std_arr

    returns: list[float] = []
    lengths: list[int] = []

    for _ in range(int(episodes)):
        obs, _ = e.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0

        while not done:
            x = obs if isinstance(obs, np.ndarray) else np.asarray(obs, dtype=np.float32)
            x = _normalize(x)

            with torch.no_grad():
                obs_t = torch.as_tensor(x, dtype=torch.float32, device=device)
                dist = policy.distribution(obs_t)
                act = dist.mode()
                a_np = act.detach().cpu().numpy()

            # Discrete or continuous action routing
            if hasattr(act_space, "n"):
                action_for_env = int(a_np.item())
            else:
                action_for_env = a_np.reshape(-1)

            next_obs, r, term, trunc, _ = e.step(action_for_env)
            ep_ret += float(r)
            ep_len += 1
            obs = next_obs
            done = bool(term) or bool(trunc)

        returns.append(ep_ret)
        lengths.append(ep_len)

    summary = {
        "env_id": str(env),
        "episodes": int(episodes),
        "seed": seed,
        "checkpoint_step": step,
        "mean_return": float(mean(returns)) if returns else 0.0,
        "std_return": float(pstdev(returns)) if len(returns) > 1 else 0.0,
        "min_return": float(min(returns)) if returns else 0.0,
        "max_return": float(max(returns)) if returns else 0.0,
        "mean_length": float(mean(lengths)) if lengths else 0.0,
        "std_length": float(pstdev(lengths)) if len(lengths) > 1 else 0.0,
        "returns": [float(x) for x in returns],
        "lengths": [int(x) for x in lengths],
    }

    e.close()
    return summary
