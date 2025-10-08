"""Training entry point (minimal PPO CLI via Typer).

This implements a *small* but functional training loop for Vanilla PPO on vector
Gymnasium environments (default: MountainCar-v0). It intentionally avoids any
intrinsic reward modules for Sprint 0.

Usage examples
--------------
# From a YAML config (recommended):
python -m irl.train --config configs/mountaincar_vanilla.yaml --total-steps 10000

# With defaults (runs MountainCar-v0 vanilla for 10k steps on CPU):
python -m irl.train --total-steps 10000

Notes
-----
* We rely on `irl.algo.advantage.compute_gae` and `irl.algo.ppo.ppo_update`.
* Vector envs are created via `irl.envs.EnvManager` with RecordEpisodeStatistics.
* Logging is handled by MetricLogger (CSV every `csv_interval`; TB if enabled).
* A checkpoint is saved at the end of training and at configured intervals.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer

from irl.algo.advantage import compute_gae
from irl.algo.ppo import ppo_update
from irl.cfg import Config, ConfigError, load_config, to_dict, validate_config
from irl.envs import EnvManager
from irl.models import PolicyNetwork, ValueNetwork
from irl.utils.checkpoint import CheckpointManager
from irl.utils.loggers import MetricLogger

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


# ----------------------------- helpers -----------------------------


def _single_spaces(env) -> tuple:
    """Return (obs_space, action_space) for both single and vector envs."""
    obs_space = getattr(env, "single_observation_space", None) or env.observation_space
    act_space = getattr(env, "single_action_space", None) or env.action_space
    return obs_space, act_space


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _default_run_dir(cfg: Config) -> Path:
    base = Path("runs")
    env_id = cfg.env.id.replace("/", "-")
    tag = f"{cfg.method}__{env_id}__seed{cfg.seed}__{_now_tag()}"
    return base / tag


def _ensure_device(dev_str: str) -> torch.device:
    d = dev_str.strip().lower()
    if d.startswith("cuda") and not torch.cuda.is_available():
        typer.echo("[yellow]CUDA requested but not available; falling back to CPU.[/yellow]")
        return torch.device("cpu")
    return torch.device(dev_str)


# ----------------------------- CLI -----------------------------


@app.command("train")
def cli_train(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML configuration (see configs/mountaincar_vanilla.yaml).",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    total_steps: int = typer.Option(
        10_000, "--total-steps", "-n", help="Total environment steps to run (across all envs)."
    ),
    run_dir: Optional[Path] = typer.Option(
        None, "--run-dir", help="Run directory for logs and checkpoints (auto if omitted)."
    ),
    method: Optional[str] = typer.Option(
        None,
        "--method",
        help="Override method in config (vanilla|icm|rnd|ride|riac|proposed). Defaults to 'vanilla' if no config.",
    ),
) -> None:
    """Run a minimal PPO training loop (Vanilla only for Sprint 0)."""
    # --- Load or synthesize config ---
    if config is not None:
        cfg = load_config(str(config))
    else:
        cfg = Config()  # use defaults from schema
    # If no config was supplied and no method override, force vanilla for Sprint 0
    if config is None and method is None:
        method = "vanilla"
    if method is not None:
        try:
            cfg = replace(cfg, method=str(method))
        except Exception:
            raise ConfigError(f"Invalid method override: {method!r}")
    # Validate ranges and divisibility
    validate_config(cfg)

    # --- Device & seeding ---
    device = _ensure_device(cfg.device)
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    # --- Env & models ---
    manager = EnvManager(
        env_id=cfg.env.id,
        num_envs=cfg.env.vec_envs,
        seed=cfg.seed,
        frame_skip=cfg.env.frame_skip,
        domain_randomization=cfg.env.domain_randomization,
        discrete_actions=cfg.env.discrete_actions,
        render_mode=None,
        async_vector=False,
    )
    env = manager.make()
    obs_space, act_space = _single_spaces(env)

    policy = PolicyNetwork(obs_space, act_space).to(device)
    value = ValueNetwork(obs_space).to(device)

    # --- Run directory, logging, checkpoints ---
    run_dir = Path(run_dir) if run_dir is not None else _default_run_dir(cfg)
    ml = MetricLogger(run_dir, cfg.logging)
    ml.log_hparams(to_dict(cfg))
    ckpt = CheckpointManager(run_dir, interval_steps=cfg.logging.checkpoint_interval, max_to_keep=3)

    # --- Reset env(s) ---
    obs, _ = env.reset()
    # shape helpers
    vec_envs = int(getattr(env, "num_envs", 1))
    obs_dim = int(obs_space.shape[0])  # Box flat observations (Sprint 0 scope)
    is_discrete = hasattr(act_space, "n")

    global_step = 0
    update_idx = 0

    try:
        while global_step < int(total_steps):
            # Steps per update (per-env); cap to remaining steps
            per_env_steps = min(
                int(cfg.ppo.steps_per_update),
                max(1, (int(total_steps) - int(global_step) + vec_envs - 1) // vec_envs),
            )

            # --- Rollout buffers (T, B, ...) ---
            T, B = per_env_steps, vec_envs
            obs_seq = np.zeros((T, B, obs_dim), dtype=np.float32)
            acts_seq = (
                np.zeros((T, B), dtype=np.int64)
                if is_discrete
                else np.zeros((T, B, int(act_space.shape[0])), dtype=np.float32)
            )
            rew_seq = np.zeros((T, B), dtype=np.float32)
            done_seq = np.zeros((T, B), dtype=np.float32)

            for t in range(T):
                # Store current obs
                if B == 1 and isinstance(obs, np.ndarray) and obs.ndim == 1:
                    obs_b = obs[None, :]
                else:
                    obs_b = obs
                obs_seq[t] = obs_b

                # Sample actions
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs_b, device=device, dtype=torch.float32)
                    a_tensor, _ = policy.act(obs_tensor)
                a_np = a_tensor.detach().cpu().numpy()
                if is_discrete:
                    a_np = a_np.astype(np.int64).reshape(B,)
                else:
                    a_np = a_np.reshape(B, -1).astype(np.float32)

                # Step env(s)
                next_obs, rewards, terms, truncs, infos = env.step(a_np if B > 1 else a_np[0])
                done_flags = np.asarray(terms, dtype=bool) | np.asarray(truncs, dtype=bool)

                # Record step
                acts_seq[t] = a_np if B > 1 else (a_np if is_discrete else a_np[None, :])
                rew_seq[t] = np.asarray(rewards, dtype=np.float32).reshape(B)
                done_seq[t] = np.asarray(done_flags, dtype=np.float32).reshape(B)

                obs = next_obs
                global_step += B

            # --- GAE (time-major) ---
            gae_batch = {
                "obs": obs_seq,  # (T, B, obs_dim)
                "rewards": rew_seq,  # (T, B)
                "dones": done_seq,  # (T, B)
            }
            adv, v_targets = compute_gae(
                gae_batch, value, gamma=float(cfg.ppo.gamma), lam=float(cfg.ppo.gae_lambda)
            )

            # --- PPO update (flatten to N = T*B) ---
            obs_flat = obs_seq.reshape(T * B, obs_dim)
            if is_discrete:
                acts_flat = acts_seq.reshape(T * B)
            else:
                acts_flat = acts_seq.reshape(T * B, -1)

            batch = {"obs": obs_flat, "actions": acts_flat, "rewards": rew_seq.reshape(T * B), "dones": done_seq.reshape(T * B)}
            ppo_update(policy, value, batch, adv, v_targets, cfg.ppo)
            update_idx += 1

            # --- Logging ---
            with torch.no_grad():
                last_obs = torch.as_tensor(obs_seq[-1], device=device, dtype=torch.float32).view(B, -1)
                ent = policy.entropy(last_obs).mean().item()
            ml.log(
                step=int(global_step),
                policy_entropy=float(ent),
                reward_mean=float(rew_seq.mean()),
            )

            # --- Checkpoint cadence ---
            if ckpt.should_save(int(global_step)):
                payload = {
                    "step": int(global_step),
                    "policy": policy.state_dict(),
                    "value": value.state_dict(),
                    "cfg": to_dict(cfg),
                    "meta": {"updates": update_idx},
                }
                ckpt.save(step=int(global_step), payload=payload)

        # Final checkpoint
        payload = {
            "step": int(global_step),
            "policy": policy.state_dict(),
            "value": value.state_dict(),
            "cfg": to_dict(cfg),
            "meta": {"updates": update_idx},
        }
        ckpt.save(step=int(global_step), payload=payload)

    finally:
        ml.close()
        env.close()

    typer.echo(f"[green]Training finished[/green] — steps={global_step}, updates={update_idx}\nRun dir: {run_dir}")


def main() -> None:
    """Entrypoint for console_scripts and `python -m irl.train`."""
    app()


if __name__ == "__main__":
    main()
