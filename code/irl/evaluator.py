"""
Evaluate a saved policy deterministically (no intrinsic).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import typer
import torch

from irl.envs import EnvManager
from irl.models import PolicyNetwork
from irl.trainer.build import single_spaces  # use shared helper
from irl.utils.checkpoint import load_checkpoint
from irl.utils.determinism import seed_everything  # unified seeding helper

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


def _is_image_space(space) -> bool:
    """Heuristic: Box with rank >= 2 is treated as image-like."""
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
        Gymnasium environment id (for example "MountainCar-v0").
    ckpt:
        Path to a training checkpoint produced by the trainer.
    episodes:
        Number of episodes to run.
    device:
        Torch device string, such as "cpu" or "cuda:0".

    Returns
    -------
    dict
        Dictionary with keys:

        * env_id
        * episodes
        * seed
        * checkpoint_step
        * aggregate statistics for returns and episode lengths
        * per-episode returns and lengths lists
    """
    payload = load_checkpoint(ckpt, map_location=device)
    cfg = payload.get("cfg", {}) or {}
    seed = int(cfg.get("seed", 1))
    step = int(payload.get("step", -1))

    # Apply consistent seeding so evaluation matches training determinism.
    seed_everything(seed, deterministic=True)

    # Derive env settings from the checkpoint to ensure parity with training.
    env_cfg = (cfg.get("env") or {}) if isinstance(cfg, dict) else {}
    frame_skip = int(env_cfg.get("frame_skip", 1))
    # For CarRacing, discrete vs Box(3,) matters — use the original training setting.
    discrete_actions = bool(env_cfg.get("discrete_actions", True))
    # Deterministic evaluation: disable domain randomization regardless of training value.
    domain_randomization = False

    # Build a single environment (no vectorisation) for evaluation
    manager = EnvManager(
        env_id=env,
        num_envs=1,
        seed=seed,
        frame_skip=frame_skip,
        domain_randomization=domain_randomization,
        discrete_actions=discrete_actions,
    )
    e = manager.make()
    obs_space, act_space = single_spaces(e)

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
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns, ddof=0)) if len(returns) > 1 else 0.0,
        "min_return": float(min(returns)) if returns else 0.0,
        "max_return": float(max(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "std_length": float(np.std(lengths, ddof=0)) if len(lengths) > 1 else 0.0,
        "returns": [float(x) for x in returns],
        "lengths": [int(x) for x in lengths],
    }

    e.close()
    return summary


@app.command("eval")
def cli_eval(
    env: str = typer.Option(..., "--env", "-e", help="Gymnasium env id (e.g., MountainCar-v0)."),
    ckpt: Path = typer.Option(
        ..., "--ckpt", "-k", help="Path to a training checkpoint file.", exists=True
    ),
    episodes: int = typer.Option(10, "--episodes", "-n", help="Number of episodes to evaluate."),
    device: str = typer.Option(
        "cpu", "--device", "-d", help='Torch device, e.g., "cpu" or "cuda:0".'
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Optional path to write aggregated results as JSON.",
        dir_okay=False,
    ),
) -> None:
    """Run evaluation episodes using mode actions and report aggregate stats."""
    summary = evaluate(env=env, ckpt=ckpt, episodes=episodes, device=device)

    # Per-episode lines
    for i, (ret, length) in enumerate(zip(summary["returns"], summary["lengths"]), start=1):
        typer.echo(f"Episode {i}/{summary['episodes']}: return = {ret:.2f}, length = {length}")

    # Aggregate line
    typer.echo(
        f"\n[green]Eval complete[/green] — mean return {summary['mean_return']:.2f} "
        f"± {summary['std_return']:.2f} over {summary['episodes']} episodes"
    )

    # Optional JSON dump (atomic)
    if out is not None:
        text = json.dumps(summary, indent=2)
        from irl.utils.checkpoint import atomic_write_text  # lazy import (keeps module light)

        atomic_write_text(out, text)
        typer.echo(f"Saved summary to {out}")


def main() -> None:
    """Entry point for the ``irl-eval`` console script."""
    app()


if __name__ == "__main__":
    main()
