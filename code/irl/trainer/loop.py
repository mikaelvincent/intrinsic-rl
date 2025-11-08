from __future__ import annotations

import dataclasses
from dataclasses import replace
from math import ceil
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from irl.algo.advantage import compute_gae
from irl.algo.ppo import ppo_update
from irl.cfg import Config, ConfigError, to_dict, validate_config
from irl.envs import EnvManager
from irl.models import PolicyNetwork, ValueNetwork
from irl.utils.checkpoint import CheckpointManager
from irl.utils.loggers import MetricLogger
from irl.intrinsic import (  # type: ignore
    is_intrinsic_method,
    create_intrinsic_module,
    compute_intrinsic_batch,
    update_module,
    RunningRMS,
)

from .build import ensure_device, default_run_dir, single_spaces, ensure_mujoco_gl
from .obs_norm import RunningObsNorm


def _is_image_space(space) -> bool:
    return hasattr(space, "shape") and len(space.shape) >= 2


def train(
    cfg: Config,
    *,
    total_steps: int = 10_000,
    run_dir: Optional[Path] = None,
) -> Path:
    """Run PPO + optional intrinsic rewards; returns the run directory."""
    validate_config(cfg)
    device = ensure_device(cfg.device)
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    # Ensure sensible MUJOCO_GL for MuJoCo tasks (no-op for others)
    try:
        ensure_mujoco_gl(cfg.env.id)
    except Exception:
        # Defensive: never hard-fail on an advisory utility
        pass

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
    obs_space, act_space = single_spaces(env)

    is_image = _is_image_space(obs_space)

    policy = PolicyNetwork(obs_space, act_space).to(device)
    value = ValueNetwork(obs_space).to(device)

    # --- Intrinsic module (optional) ---
    method_l = str(cfg.method).lower()
    eta = float(cfg.intrinsic.eta)
    use_intrinsic = is_intrinsic_method(method_l) and eta > 0.0
    intrinsic_module = None
    if is_intrinsic_method(method_l):
        try:
            intrinsic_module = create_intrinsic_module(
                method_l,
                obs_space,
                act_space,
                device=device,
                bin_size=float(cfg.intrinsic.bin_size),
                alpha_impact=float(cfg.intrinsic.alpha_impact),
                alpha_lp=float(cfg.intrinsic.alpha_lp),
                region_capacity=int(cfg.intrinsic.region_capacity),
                depth_max=int(cfg.intrinsic.depth_max),
                ema_beta_long=float(cfg.intrinsic.ema_beta_long),
                ema_beta_short=float(cfg.intrinsic.ema_beta_short),
                gate_tau_lp_mult=float(cfg.intrinsic.gate.tau_lp_mult),
                gate_tau_s=float(cfg.intrinsic.gate.tau_s),
                gate_hysteresis_up_mult=float(cfg.intrinsic.gate.hysteresis_up_mult),
                gate_min_consec_to_gate=int(cfg.intrinsic.gate.min_consec_to_gate),
            )
            if not use_intrinsic:
                print(
                    f"[warning] Method '{method_l}' selected but intrinsic.eta={eta:.3g}; "
                    "intrinsic will be computed but ignored in total reward."
                )
        except Exception as exc:
            print(
                f"[warning] Failed to create intrinsic module '{method_l}': {exc}. "
                "Continuing without intrinsic."
            )
            intrinsic_module = None
            use_intrinsic = False

    int_rms = RunningRMS(beta=0.99, eps=1e-8)

    # --- Run dir, logging, checkpoints ---
    run_dir = Path(run_dir) if run_dir is not None else default_run_dir(cfg)
    ml = MetricLogger(run_dir, cfg.logging)
    ml.log_hparams(to_dict(cfg))
    ckpt = CheckpointManager(run_dir, interval_steps=cfg.logging.checkpoint_interval, max_to_keep=3)

    # --- Reset env(s) & init norm ---
    obs, _ = env.reset()
    B = int(getattr(env, "num_envs", 1))

    obs_norm = None if is_image else RunningObsNorm(shape=int(obs_space.shape[0]))
    if not is_image:
        first_batch = obs if B > 1 else obs[None, :]
        obs_norm.update(first_batch)  # type: ignore[union-attr]

    global_step = 0
    update_idx = 0

    try:
        while global_step < int(total_steps):
            # Steps per update (per-env), capped by remaining
            per_env_steps = min(
                int(cfg.ppo.steps_per_update),
                max(1, ceil((int(total_steps) - int(global_step)) / B)),
            )
            T = per_env_steps

            # Allocate storage depending on observation type
            if not is_image:
                obs_dim = int(obs_space.shape[0])
                obs_seq = np.zeros((T, B, obs_dim), dtype=np.float32)
                next_obs_seq = np.zeros((T, B, obs_dim), dtype=np.float32)
            else:
                obs_seq_list = []
                next_obs_seq_list = []

            is_discrete = hasattr(act_space, "n")
            acts_seq = (
                np.zeros((T, B), dtype=np.int64)
                if is_discrete
                else np.zeros((T, B, int(act_space.shape[0])), dtype=np.float32)
            )
            rew_ext_seq = np.zeros((T, B), dtype=np.float32)
            done_seq = np.zeros((T, B), dtype=np.float32)
            r_int_raw_seq = np.zeros((T, B), dtype=np.float32) if (
                intrinsic_module is not None and method_l == "ride"
            ) else None

            for t in range(T):
                obs_b = obs if B > 1 else obs[None, ...]
                if not is_image:
                    obs_norm.update(obs_b)  # type: ignore[union-attr]
                    obs_b_norm = obs_norm.normalize(obs_b)  # type: ignore[union-attr]
                else:
                    obs_b_norm = obs_b  # images: no vector obs-norm (handled inside nets)

                if not is_image:
                    obs_seq[t] = obs_b_norm.astype(np.float32)
                else:
                    obs_seq_list.append(obs_b_norm.astype(np.float32))

                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs_b_norm, device=device, dtype=torch.float32)
                    a_tensor, _ = policy.act(obs_tensor)
                a_np = a_tensor.detach().cpu().numpy()
                if is_discrete:
                    a_np = a_np.astype(np.int64).reshape(B)
                else:
                    a_np = a_np.reshape(B, -1).astype(np.float32)

                next_obs, rewards, terms, truncs, _ = env.step(a_np if B > 1 else a_np[0])
                done_flags = np.asarray(terms, dtype=bool) | np.asarray(truncs, dtype=bool)

                next_obs_b = next_obs if B > 1 else next_obs[None, ...]
                if not is_image:
                    obs_norm.update(next_obs_b)  # type: ignore[union-attr]
                    next_obs_b_norm = obs_norm.normalize(next_obs_b)  # type: ignore[union-attr]
                else:
                    next_obs_b_norm = next_obs_b

                acts_seq[t] = a_np if B > 1 else (a_np if is_discrete else a_np[0:1, :])
                rew_ext_seq[t] = np.asarray(rewards, dtype=np.float32).reshape(B)
                done_seq[t] = np.asarray(done_flags, dtype=np.float32).reshape(B)

                if not is_image:
                    next_obs_seq[t] = next_obs_b_norm.astype(np.float32)
                else:
                    next_obs_seq_list.append(next_obs_b_norm.astype(np.float32))

                if r_int_raw_seq is not None:
                    r_step = intrinsic_module.compute_impact_binned(  # type: ignore[union-attr]
                        obs_b_norm,
                        next_obs_b_norm,
                        dones=done_flags,
                        reduction="none",
                    )
                    r_int_raw_seq[t] = (
                        r_step.detach().cpu().numpy().reshape(B).astype(np.float32)
                    )

                obs = next_obs
                global_step += B

            # Build obs arrays
            if not is_image:
                obs_seq_final = obs_seq
                next_obs_seq_final = next_obs_seq
                obs_shape = (int(obs_space.shape[0]),)
            else:
                obs_seq_final = np.stack(obs_seq_list, axis=0)
                next_obs_seq_final = np.stack(next_obs_seq_list, axis=0)
                obs_shape = tuple(int(s) for s in obs_space.shape)

            # --- Intrinsic compute/update (optional) ---
            r_int_raw_flat = None
            r_int_scaled_flat = None
            mod_metrics = {}
            if intrinsic_module is not None:
                if method_l == "ride":
                    r_int_raw_flat = r_int_raw_seq.reshape(T * B).astype(np.float32)  # type: ignore[union-attr]
                else:
                    obs_flat = obs_seq_final.reshape((T * B,) + obs_shape)
                    next_obs_flat = next_obs_seq_final.reshape((T * B,) + obs_shape)
                    acts_flat = acts_seq.reshape(T * B) if is_discrete else acts_seq.reshape(T * B, -1)
                    r_int_raw_t = compute_intrinsic_batch(
                        intrinsic_module, method_l, obs_flat, next_obs_flat, acts_flat
                    )
                    r_int_raw_flat = r_int_raw_t.detach().cpu().numpy().astype(np.float32)

                r_clip = float(cfg.intrinsic.r_clip)
                outputs_norm = bool(getattr(intrinsic_module, "outputs_normalized", False))
                if outputs_norm:
                    r_int_scaled_flat = eta * np.clip(r_int_raw_flat, -r_clip, r_clip)
                else:
                    int_rms.update(r_int_raw_flat)
                    r_int_norm_flat = int_rms.normalize(r_int_raw_flat)
                    r_int_scaled_flat = eta * np.clip(r_int_norm_flat, -r_clip, r_clip)

                try:
                    if method_l == "ride":
                        obs_flat = obs_seq_final.reshape((T * B,) + obs_shape)
                        next_obs_flat = next_obs_seq_final.reshape((T * B,) + obs_shape)
                        acts_flat = acts_seq.reshape(T * B) if is_discrete else acts_seq.reshape(T * B, -1)
                    mod_metrics = update_module(
                        intrinsic_module, method_l, obs_flat, next_obs_flat, acts_flat  # type: ignore[arg-type]
                    )
                except Exception:
                    mod_metrics = {}

            # --- Rewards & GAE ---
            if r_int_scaled_flat is not None:
                rew_total_seq = rew_ext_seq + r_int_scaled_flat.reshape(T, B)
            else:
                rew_total_seq = rew_ext_seq

            gae_batch = {
                "obs": obs_seq_final,
                "next_observations": next_obs_seq_final,
                "rewards": rew_total_seq,
                "dones": done_seq,
            }
            adv, v_targets = compute_gae(
                gae_batch, value, gamma=float(cfg.ppo.gamma), lam=float(cfg.ppo.gae_lambda)
            )

            obs_flat_for_ppo = (
                obs_seq_final.reshape(T * B, -1) if not is_image else obs_seq_final.reshape((T * B,) + obs_shape)
            )
            acts_flat_for_ppo = acts_seq.reshape(T * B) if is_discrete else acts_seq.reshape(T * B, -1)
            batch = {
                "obs": obs_flat_for_ppo,
                "actions": acts_flat_for_ppo,
                "rewards": rew_total_seq.reshape(T * B),
                "dones": done_seq.reshape(T * B),
            }
            ppo_update(policy, value, batch, adv, v_targets, cfg.ppo)
            update_idx += 1

            # --- Logging ---
            with torch.no_grad():
                last_obs = torch.as_tensor(
                    obs_seq_final[-1], device=device, dtype=torch.float32
                )  # [B,...]
                ent = float(policy.entropy(last_obs).mean().item())

            log_payload = {
                "policy_entropy": float(ent),
                "reward_mean": float(rew_ext_seq.mean()),
                "reward_total_mean": float(rew_total_seq.mean()),
            }
            if r_int_raw_flat is not None and r_int_scaled_flat is not None:
                outputs_norm = bool(getattr(intrinsic_module, "outputs_normalized", False)) if intrinsic_module else False
                if outputs_norm and hasattr(intrinsic_module, "lp_rms"):
                    r_int_rms_val = float(getattr(intrinsic_module, "lp_rms"))
                else:
                    r_int_rms_val = float(int_rms.rms)
                log_payload.update(
                    {
                        "r_int_raw_mean": float(np.mean(r_int_raw_flat)),
                        "r_int_mean": float(np.mean(r_int_scaled_flat)),
                        "r_int_rms": r_int_rms_val,
                    }
                )
                for k, v in (mod_metrics or {}).items():
                    try:
                        log_payload[f"{method_l}_{k}"] = float(v)
                    except Exception:
                        pass

            if intrinsic_module is not None and method_l == "proposed" and hasattr(intrinsic_module, "gate_rate"):
                try:
                    gr = float(getattr(intrinsic_module, "gate_rate"))
                    log_payload["gate_rate"] = gr
                    log_payload["gate_rate_pct"] = 100.0 * gr
                except Exception:
                    pass

            ml.log(step=int(global_step), **log_payload)

            # RIAC diagnostics cadence
            try:
                if (
                    intrinsic_module is not None
                    and method_l == "riac"
                    and int(global_step) % int(cfg.logging.csv_interval) == 0
                    and hasattr(intrinsic_module, "export_diagnostics")
                ):
                    diag_dir = run_dir / "diagnostics"
                    intrinsic_module.export_diagnostics(diag_dir, step=int(global_step))  # type: ignore[union-attr]
            except Exception:
                pass

            # --- Checkpoint cadence ---
            if ckpt.should_save(int(global_step)):
                payload = {
                    "step": int(global_step),
                    "policy": policy.state_dict(),
                    "value": value.state_dict(),
                    "cfg": to_dict(cfg),
                    # Only persist obs_norm for vector observations
                    "obs_norm": None if is_image else {"count": obs_norm.count, "mean": obs_norm.mean, "var": obs_norm.var},  # type: ignore[union-attr]
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

        # Final checkpoint
        payload = {
            "step": int(global_step),
            "policy": policy.state_dict(),
            "value": value.state_dict(),
            "cfg": to_dict(cfg),
            "obs_norm": None if is_image else {"count": obs_norm.count, "mean": obs_norm.mean, "var": obs_norm.var},  # type: ignore[union-attr]
            "intrinsic_norm": int_rms.state_dict(),
            "meta": {"updates": update_idx},
        }
        if intrinsic_module is not None and hasattr(intrinsic_module, "state_dict"):
            try:
                payload["intrinsic"] = {"method": method_l, "state_dict": intrinsic_module.state_dict()}
            except Exception:
                pass
        ckpt.save(step=int(global_step), payload=payload)

    finally:
        ml.close()
        env.close()

    return run_dir
