"""Training entry point (minimal PPO CLI via Typer).

This implements a *small* but functional training loop for Vanilla PPO on vector
Gymnasium environments (default: MountainCar-v0). It now supports plugging in
Sprint‑1 intrinsic modules (`icm`, `rnd`) via a tiny factory. Intrinsic rewards
are computed on collected batches and added to extrinsic rewards before GAE/PPO.

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
* If `cfg.method` is "icm" or "rnd" and `intrinsic.eta > 0`, intrinsic rewards
  are computed and added to extrinsic rewards prior to advantage estimation.
* Intrinsic rewards are globally normalized online by a running RMS (EMA)
  to stabilize scales across modules.

Sprint‑2 addition
-----------------
* For method `"ride"`, intrinsic rewards are computed **per step** using
  `RIDE.compute_impact_binned(...)` with episodic binning & counts (§5.5).

Sprint‑3 update
---------------
* For modules that already **return normalized** intrinsic (e.g., RIAC after
  Sprint 3 — Step 3), we **skip** the global RMS normalization to avoid double
  normalization. Such modules expose `outputs_normalized=True` and (optionally)
  `lp_rms` for logging.

Sprint‑3 — Step 4 (this change)
-------------------------------
* When method is `"riac"`, export diagnostics at the CSV logging cadence:
  `diagnostics/regions.jsonl` (JSONL) and `diagnostics/gates.csv` (CSV).
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

# Intrinsic factory & helpers (Sprint 1)
from irl.intrinsic import (  # type: ignore
    is_intrinsic_method,
    create_intrinsic_module,
    compute_intrinsic_batch,
    update_module,
    RunningRMS,
)

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


# ----------------------------- running obs norm -----------------------------


class RunningObsNorm:
    """Per-dimension running mean/variance with batch updates."""

    def __init__(self, shape: int):
        self.count = 0.0
        self.mean = np.zeros((shape,), dtype=np.float64)
        self.var = np.ones((shape,), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """x: [B, obs_dim] batch."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        b = float(x.shape[0])
        if b == 0:
            return
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)  # population variance

        if self.count == 0.0:
            self.mean = batch_mean
            self.var = batch_var
            self.count = b
            return

        delta = batch_mean - self.mean
        tot = self.count + b
        new_mean = self.mean + delta * (b / tot)

        # Combine variances (Chan et al.)
        m_a = self.var * self.count
        m_b = batch_var * b
        new_var = (m_a + m_b + (delta**2) * (self.count * b / tot)) / tot

        self.mean = np.maximum(new_mean, -1e12)  # clamp extremely for safety (theoretical)
        self.var = np.maximum(new_var, 1e-12)  # numeric safety
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var + 1e-8)
        return (x - self.mean) / std

    def state_dict(self) -> dict:
        return {
            "count": float(self.count),
            "mean": self.mean.astype(np.float64),
            "var": self.var.astype(np.float64),
        }


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
        10_000,
        "--total-steps",
        "-n",
        help="Total environment steps to run (across all envs).",
    ),
    run_dir: Optional[Path] = typer.Option(
        None,
        "--run-dir",
        help="Run directory for logs and checkpoints (auto if omitted).",
    ),
    method: Optional[str] = typer.Option(
        None,
        "--method",
        help="Override method in config (vanilla|icm|rnd|ride|riac|proposed). Defaults to 'vanilla' if no config.",
    ),
) -> None:
    """Run a minimal PPO training loop with optional intrinsic rewards (ICM/RND/RIDE/RIAC)."""
    # --- Load or synthesize config ---
    if config is not None:
        cfg = load_config(str(config))
    else:
        cfg = Config()  # use defaults from schema
    # If no config was supplied and no method override, force vanilla for Sprint 0/1
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

    # --- Intrinsic module (optional) ---
    method_l = str(cfg.method).lower()
    eta = float(cfg.intrinsic.eta)
    use_intrinsic = is_intrinsic_method(method_l) and eta > 0.0
    intrinsic_module = None
    if is_intrinsic_method(method_l):
        try:
            # For RIDE, plumb bin_size and alpha_impact from config.
            intrinsic_module = create_intrinsic_module(
                method_l,
                obs_space,
                act_space,
                device=device,
                bin_size=float(cfg.intrinsic.bin_size),
                alpha_impact=float(cfg.intrinsic.alpha_impact),
            )
            if not use_intrinsic:
                typer.echo(
                    f"[yellow]Method '{method_l}' selected but intrinsic.eta={eta:.3g};"
                    " intrinsic will be computed but ignored in total reward.[/yellow]"
                )
        except Exception as exc:
            typer.echo(
                f"[yellow]Failed to create intrinsic module '{method_l}': {exc}. "
                "Continuing without intrinsic.[/yellow]"
            )
            intrinsic_module = None
            use_intrinsic = False

    # --- Global intrinsic RMS normalizer ---
    int_rms = RunningRMS(beta=0.99, eps=1e-8)

    # --- Run directory, logging, checkpoints ---
    run_dir = Path(run_dir) if run_dir is not None else _default_run_dir(cfg)
    ml = MetricLogger(run_dir, cfg.logging)
    ml.log_hparams(to_dict(cfg))
    ckpt = CheckpointManager(run_dir, interval_steps=cfg.logging.checkpoint_interval, max_to_keep=3)

    # --- Reset env(s) ---
    obs, _ = env.reset()
    vec_envs = int(getattr(env, "num_envs", 1))
    obs_dim = int(obs_space.shape[0])  # Box flat observations (Sprint 0/1 scope)
    is_discrete = hasattr(act_space, "n")

    # Observation normalization
    obs_norm = RunningObsNorm(shape=obs_dim)
    first_batch = obs if vec_envs > 1 else obs[None, :]
    obs_norm.update(first_batch)

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
            next_obs_seq = np.zeros((T, B, obs_dim), dtype=np.float32)
            acts_seq = (
                np.zeros((T, B), dtype=np.int64)
                if is_discrete
                else np.zeros((T, B, int(act_space.shape[0])), dtype=np.float32)
            )
            rew_ext_seq = np.zeros((T, B), dtype=np.float32)
            done_seq = np.zeros((T, B), dtype=np.float32)

            # For RIDE episodic binning: store raw intrinsic per step
            r_int_raw_seq = None
            if intrinsic_module is not None and method_l == "ride":
                r_int_raw_seq = np.zeros((T, B), dtype=np.float32)

            for t in range(T):
                # Ensure [B, obs_dim]
                obs_b = obs if B > 1 else obs[None, :]
                # Update running stats and normalize
                obs_norm.update(obs_b)
                obs_b_norm = obs_norm.normalize(obs_b)

                # Store normalized obs
                obs_seq[t] = obs_b_norm.astype(np.float32)

                # Sample actions
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs_b_norm, device=device, dtype=torch.float32)
                    a_tensor, _ = policy.act(obs_tensor)
                a_np = a_tensor.detach().cpu().numpy()
                if is_discrete:
                    a_np = a_np.astype(np.int64).reshape(
                        B,
                    )
                else:
                    a_np = a_np.reshape(B, -1).astype(np.float32)

                # Step env(s)
                next_obs, rewards, terms, truncs, infos = env.step(a_np if B > 1 else a_np[0])
                done_flags = np.asarray(terms, dtype=bool) | np.asarray(truncs, dtype=bool)

                # Prepare normalized next_obs for GAE bootstrapping
                next_obs_b = next_obs if B > 1 else next_obs[None, :]
                obs_norm.update(next_obs_b)
                next_obs_b_norm = obs_norm.normalize(next_obs_b)

                # Record step
                acts_seq[t] = a_np if B > 1 else (a_np if is_discrete else a_np[None, :])
                rew_ext_seq[t] = np.asarray(rewards, dtype=np.float32).reshape(B)
                done_seq[t] = np.asarray(done_flags, dtype=np.float32).reshape(B)
                next_obs_seq[t] = next_obs_b_norm.astype(np.float32)

                # --- Intrinsic (RIDE episodic binning computed per step) ---
                if intrinsic_module is not None and method_l == "ride":
                    # Compute raw (un-normalized/un-scaled) impact with episodic counts
                    r_step = intrinsic_module.compute_impact_binned(
                        obs_b_norm,
                        next_obs_b_norm,
                        dones=done_flags,
                        reduction="none",
                    )
                    r_int_raw_seq[t] = r_step.detach().cpu().numpy().reshape(B).astype(np.float32)

                obs = next_obs
                global_step += B

            # --- Intrinsic rewards (optional) ---
            r_int_raw_flat = None
            r_int_scaled_flat = None
            mod_metrics = {}
            if intrinsic_module is not None:
                if method_l == "ride":
                    # Flatten the per-step raw values (already binned, includes alpha_impact)
                    r_int_raw_flat = r_int_raw_seq.reshape(T * B).astype(np.float32)
                else:
                    # ICM / RND / RIAC path: compute on flattened batch
                    obs_flat = obs_seq.reshape(T * B, obs_dim)
                    next_obs_flat = next_obs_seq.reshape(T * B, obs_dim)
                    if is_discrete:
                        acts_flat = acts_seq.reshape(T * B)
                    else:
                        acts_flat = acts_seq.reshape(T * B, -1)

                    # Compute per-sample intrinsic [N]; module may already normalize
                    r_int_raw_t = compute_intrinsic_batch(
                        intrinsic_module, method_l, obs_flat, next_obs_flat, acts_flat
                    )
                    r_int_raw_flat = r_int_raw_t.detach().cpu().numpy().astype(np.float32)

                # Scale & (maybe) normalize
                r_clip = float(cfg.intrinsic.r_clip)
                outputs_norm = bool(getattr(intrinsic_module, "outputs_normalized", False))
                if outputs_norm:
                    # Module already normalized (e.g., RIAC). Skip global RMS to avoid double normalization.
                    r_int_scaled_flat = eta * np.clip(r_int_raw_flat, -r_clip, r_clip)
                else:
                    int_rms.update(r_int_raw_flat)
                    r_int_norm_flat = int_rms.normalize(r_int_raw_flat)
                    r_int_scaled_flat = eta * np.clip(r_int_norm_flat, -r_clip, r_clip)

                # Optionally train intrinsic module on the same batch (ICM/RND/RIDE encoder/RIAC ICM)
                try:
                    if method_l == "ride":
                        obs_flat = obs_seq.reshape(T * B, obs_dim)
                        next_obs_flat = next_obs_seq.reshape(T * B, obs_dim)
                        if is_discrete:
                            acts_flat = acts_seq.reshape(T * B)
                        else:
                            acts_flat = acts_seq.reshape(T * B, -1)
                    mod_metrics = update_module(
                        intrinsic_module, method_l, obs_flat, next_obs_flat, acts_flat
                    )
                except Exception:
                    mod_metrics = {}

            # --- Rewards for GAE: extrinsic + (optional) intrinsic ---
            if r_int_scaled_flat is not None:
                rew_total_seq = rew_ext_seq + r_int_scaled_flat.reshape(T, B)
            else:
                rew_total_seq = rew_ext_seq

            # --- GAE (time-major) with proper bootstrap from v(s_{t+1}) ---
            gae_batch = {
                "obs": obs_seq,  # (T, B, obs_dim) normalized
                "next_observations": next_obs_seq,
                "rewards": rew_total_seq,  # (T, B)
                "dones": done_seq,  # (T, B)
            }
            adv, v_targets = compute_gae(
                gae_batch,
                value,
                gamma=float(cfg.ppo.gamma),
                lam=float(cfg.ppo.gae_lambda),
            )

            # --- PPO update (flatten to N = T*B) ---
            obs_flat = obs_seq.reshape(T * B, obs_dim)
            if is_discrete:
                acts_flat = acts_seq.reshape(T * B)
            else:
                acts_flat = acts_seq.reshape(T * B, -1)

            batch = {
                "obs": obs_flat,
                "actions": acts_flat,
                "rewards": rew_total_seq.reshape(T * B),
                "dones": done_seq.reshape(T * B),
            }
            ppo_update(policy, value, batch, adv, v_targets, cfg.ppo)
            update_idx += 1

            # --- Logging ---
            with torch.no_grad():
                last_obs = torch.as_tensor(obs_seq[-1], device=device, dtype=torch.float32).view(
                    B, -1
                )
                ent = policy.entropy(last_obs).mean().item()

            log_payload = {
                "policy_entropy": float(ent),
                "reward_mean": float(rew_ext_seq.mean()),
                "reward_total_mean": float(rew_total_seq.mean()),
            }
            if r_int_raw_flat is not None and r_int_scaled_flat is not None:
                # Prefer module-provided RMS if outputs are pre-normalized (e.g., RIAC)
                outputs_norm = (
                    bool(getattr(intrinsic_module, "outputs_normalized", False))
                    if intrinsic_module is not None
                    else False
                )
                if outputs_norm and hasattr(intrinsic_module, "lp_rms"):
                    r_int_rms_val = float(getattr(intrinsic_module, "lp_rms"))
                else:
                    r_int_rms_val = float(int_rms.rms)
                log_payload.update(
                    {
                        "r_int_raw_mean": float(np.mean(r_int_raw_flat)),
                        "r_int_mean": float(
                            np.mean(r_int_scaled_flat)
                        ),  # normalized (module or global) + scaled (η)
                        "r_int_rms": r_int_rms_val,
                    }
                )
                # Prefix module metrics by method to avoid collisions across methods
                for k, v in (mod_metrics or {}).items():
                    try:
                        log_payload[f"{method_l}_{k}"] = float(v)
                    except Exception:
                        pass

            ml.log(step=int(global_step), **log_payload)

            # --- Diagnostics export for RIAC (regions.jsonl & gates.csv) ---
            try:
                if (
                    intrinsic_module is not None
                    and method_l == "riac"
                    and int(global_step) % int(cfg.logging.csv_interval) == 0
                    and hasattr(intrinsic_module, "export_diagnostics")
                ):
                    diag_dir = run_dir / "diagnostics"
                    intrinsic_module.export_diagnostics(diag_dir, step=int(global_step))
            except Exception:
                # Diagnostics must never break training; ignore any I/O issues.
                pass

            # --- Checkpoint cadence ---
            if ckpt.should_save(int(global_step)):
                payload = {
                    "step": int(global_step),
                    "policy": policy.state_dict(),
                    "value": value.state_dict(),
                    "cfg": to_dict(cfg),
                    "obs_norm": {
                        "count": obs_norm.count,
                        "mean": obs_norm.mean,
                        "var": obs_norm.var,
                    },
                    "intrinsic_norm": int_rms.state_dict(),
                    "meta": {"updates": update_idx},
                }
                # Save intrinsic module weights if present
                if intrinsic_module is not None and hasattr(intrinsic_module, "state_dict"):
                    try:
                        payload["intrinsic"] = {
                            "method": method_l,
                            "state_dict": intrinsic_module.state_dict(),
                        }
                    except Exception:
                        pass
                ckpt.save(step=int(global_step), payload=payload)

        # Final checkpoint
        payload = {
            "step": int(global_step),
            "policy": policy.state_dict(),
            "value": value.state_dict(),
            "cfg": to_dict(cfg),
            "obs_norm": {
                "count": obs_norm.count,
                "mean": obs_norm.mean,
                "var": obs_norm.var,
            },
            "intrinsic_norm": int_rms.state_dict(),
            "meta": {"updates": update_idx},
        }
        if intrinsic_module is not None and hasattr(intrinsic_module, "state_dict"):
            try:
                payload["intrinsic"] = {
                    "method": method_l,
                    "state_dict": intrinsic_module.state_dict(),
                }
            except Exception:
                pass
        ckpt.save(step=int(global_step), payload=payload)

    finally:
        ml.close()
        env.close()

    typer.echo(
        f"[green]Training finished[/green] — steps={global_step}, updates={update_idx}\nRun dir: {run_dir}"
    )


def main() -> None:
    """Entrypoint for console_scripts and `python -m irl.train`."""
    app()


if __name__ == "__main__":
    main()
