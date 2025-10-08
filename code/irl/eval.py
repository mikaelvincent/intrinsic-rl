"""Evaluation entry point (minimal CLI via Typer).

Runs a policy for N episodes *without* intrinsic rewards.

Examples
--------
python -m irl.eval --env MountainCar-v0 --ckpt runs/.../checkpoints/ckpt_step_10000.pt --episodes 5
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev
from typing import Optional

import numpy as np
import torch
import typer

from irl.envs import EnvManager
from irl.models import PolicyNetwork, ValueNetwork  # Value kept for parity
from irl.utils.checkpoint import load_checkpoint

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


def _single_spaces(env) -> tuple:
    obs_space = getattr(env, "single_observation_space", None) or env.observation_space
    act_space = getattr(env, "single_action_space", None) or env.action_space
    return obs_space, act_space


def _build_normalizer(payload) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return (mean, std) if obs_norm is present in checkpoint payload."""
    on = payload.get("obs_norm")
    if on is None:
        return None
    mean = np.asarray(on.get("mean"), dtype=np.float64)
    var = np.asarray(on.get("var"), dtype=np.float64)
    std = np.sqrt(var + 1e-8)
    return mean, std


@app.command("eval")
def cli_eval(
    env: str = typer.Option(
        ..., "--env", "-e", help="Gymnasium env id (e.g., MountainCar-v0)."
    ),
    ckpt: Path = typer.Option(
        ..., "--ckpt", "-k", help="Path to a training checkpoint file.", exists=True
    ),
    episodes: int = typer.Option(
        10, "--episodes", "-n", help="Number of episodes to evaluate."
    ),
    device: str = typer.Option(
        "cpu", "--device", "-d", help='Torch device, e.g., "cpu" or "cuda:0".'
    ),
) -> None:
    """Evaluate a saved policy deterministically (mode action)."""
    payload = load_checkpoint(ckpt, map_location=device)
    cfg = payload.get("cfg", {})
    seed = int(cfg.get("seed", 1))

    manager = EnvManager(env_id=env, num_envs=1, seed=seed)
    e = manager.make()
    obs_space, act_space = _single_spaces(e)

    policy = PolicyNetwork(obs_space, act_space).to(device)
    policy.load_state_dict(payload["policy"])
    policy.eval()

    norm = _build_normalizer(payload)

    def _normalize(x: np.ndarray) -> np.ndarray:
        if norm is None:
            return x
        mean, std = norm
        return (x - mean) / std

    returns: list[float] = []

    for ep in range(int(episodes)):
        obs, _ = e.reset()
        done = False
        ep_ret = 0.0
        while not done:
            x = _normalize(
                obs
                if isinstance(obs, np.ndarray)
                else np.asarray(obs, dtype=np.float32)
            )
            with torch.no_grad():
                obs_t = torch.as_tensor(x, dtype=torch.float32, device=device).view(
                    1, -1
                )
                dist = policy.distribution(obs_t)
                act = dist.mode()
                a = act.detach().cpu().numpy()
            next_obs, r, term, trunc, _ = e.step(
                int(a.item()) if hasattr(act_space, "n") else a[0]
            )
            ep_ret += float(r)
            obs = next_obs
            done = bool(term) or bool(trunc)
        returns.append(ep_ret)
        typer.echo(f"Episode {ep+1}/{episodes}: return = {ep_ret:.2f}")

    m = mean(returns)
    s = pstdev(returns) if len(returns) > 1 else 0.0
    typer.echo(
        f"\n[green]Eval complete[/green] — mean return {m:.2f} ± {s:.2f} over {episodes} episodes"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
