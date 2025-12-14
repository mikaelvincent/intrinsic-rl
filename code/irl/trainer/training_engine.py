"""PPO training engine (rollouts, updates, logging, checkpointing).

This module hosts the main training loop that was previously implemented
inline in ``irl.trainer.loop``. The implementation is logic-identical;
only the project structure has changed.
"""

from __future__ import annotations

import time
from math import ceil
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch

from irl.algo.advantage import compute_gae
from irl.algo.ppo import ppo_update
from irl.cfg import Config, to_dict
from irl.utils.checkpoint import compute_cfg_hash
from irl.intrinsic import (  # type: ignore
    compute_intrinsic_batch,
    update_module,
)

from .runtime_utils import _apply_final_observation, _ensure_time_major_np
from .training_setup import TrainingSession


def _build_checkpoint_payload(
    cfg: Config,
    *,
    global_step: int,
    update_idx: int,
    policy: Any,
    value: Any,
    is_image: bool,
    obs_norm: Any,
    int_rms: Any,
    pol_opt: Any,
    val_opt: Any,
    intrinsic_module: Optional[Any],
    method_l: str,
) -> dict:
    """Create a checkpoint payload matching the original trainer."""
    payload = {
        "step": int(global_step),
        "policy": policy.state_dict(),
        "value": value.state_dict(),
        "cfg": to_dict(cfg),
        "cfg_hash": compute_cfg_hash(to_dict(cfg)),  # store config hash
        # Only persist obs_norm for vector observations
        "obs_norm": None
        if is_image
        else {
            "count": obs_norm.count,
            "mean": obs_norm.mean,
            "var": obs_norm.var,
        },
        "intrinsic_norm": int_rms.state_dict(),
        "meta": {"updates": int(update_idx)},
        # persist PPO optimizers
        "optimizers": {
            "policy": pol_opt.state_dict(),
            "value": val_opt.state_dict(),
        },
    }
    if intrinsic_module is not None and hasattr(intrinsic_module, "state_dict"):
        try:
            payload["intrinsic"] = {
                "method": method_l,
                "state_dict": intrinsic_module.state_dict(),
            }
        except Exception:
            pass
    return payload


def run_training_loop(
    cfg: Config,
    session: TrainingSession,
    *,
    total_steps: int,
    logger,
    log_every_updates: int = 10,
) -> None:
    """Run PPO updates until `total_steps` is reached (in env steps)."""
    run_dir = session.run_dir
    env = session.env
    obs_space = session.obs_space
    act_space = session.act_space
    is_image = session.is_image
    B = int(session.num_envs)

    policy = session.policy
    value = session.value
    pol_opt = session.optimizers.policy
    val_opt = session.optimizers.value

    intrinsic_module = session.intrinsic.module
    method_l = str(session.intrinsic.method)
    eta = float(session.intrinsic.eta)
    use_intrinsic = bool(session.intrinsic.use_intrinsic)
    intrinsic_norm_mode = str(session.intrinsic.norm_mode)
    intrinsic_outputs_normalized_flag = session.intrinsic.outputs_normalized
    int_rms = session.intrinsic.rms

    obs_norm = session.obs_norm
    ml = session.metric_logger
    device = session.device
    ckpt = session.ckpt

    obs = session.obs
    global_step = int(session.global_step)
    update_idx = int(session.update_idx)

    # Wall-clock tracking for SPS and periodic console logging
    start_wall = time.time()
    last_log_time = start_wall
    last_log_step = global_step

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
        # new: separate terminal and truncation masks for GAE
        terms_seq = np.zeros((T, B), dtype=np.float32)
        truncs_seq = np.zeros((T, B), dtype=np.float32)
        r_int_raw_seq = (
            np.zeros((T, B), dtype=np.float32)
            if (intrinsic_module is not None and method_l == "ride" and use_intrinsic)
            else None
        )

        for t in range(T):
            obs_b = obs if B > 1 else obs[None, ...]
            if not is_image:
                obs_norm.update(obs_b)  # type: ignore[union-attr]
                obs_b_norm = obs_norm.normalize(obs_b)  # type: ignore[union-attr]
            else:
                # images: keep raw dtype (e.g., uint8) for proper scaling downstream
                obs_b_norm = obs_b

            if not is_image:
                obs_seq[t] = obs_b_norm.astype(np.float32)
            else:
                # IMPORTANT: explicitly copy image observations before buffering.
                # Some VectorEnv implementations may reuse internal buffers (especially when copy=False),
                # which can silently corrupt rollouts if we store references.
                obs_seq_list.append(np.array(obs_b_norm, copy=True))

            with torch.no_grad():
                if is_image:
                    # Pass raw images; the policy will preprocess (layout + scaling)
                    a_tensor, _ = policy.act(obs_b_norm)
                else:
                    obs_tensor = torch.as_tensor(obs_b_norm, device=device, dtype=torch.float32)
                    a_tensor, _ = policy.act(obs_tensor)
            a_np = a_tensor.detach().cpu().numpy()
            if is_discrete:
                a_np = a_np.astype(np.int64).reshape(B)
            else:
                a_np = a_np.reshape(B, -1).astype(np.float32)

            # Step the env (vector envs may auto-reset; terminal obs can be in infos["final_observation"])
            next_obs_env, rewards, terms, truncs, infos = env.step(a_np if B > 1 else a_np[0])

            # Normalize masks to (B,) arrays
            terms_b = np.asarray(terms, dtype=bool).reshape(B)
            truncs_b = np.asarray(truncs, dtype=bool).reshape(B)
            done_flags = terms_b | truncs_b

            # For rollouts (GAE bootstrapping + intrinsic), use terminal observations when provided.
            # Keep stepping with the environment-provided next_obs (which may already be reset).
            next_obs_rollout = _apply_final_observation(next_obs_env, done_flags, infos)

            next_obs_b = next_obs_rollout if B > 1 else next_obs_rollout[None, ...]
            if not is_image:
                obs_norm.update(next_obs_b)  # type: ignore[union-attr]
                next_obs_b_norm = obs_norm.normalize(next_obs_b)  # type: ignore[union-attr]
            else:
                next_obs_b_norm = next_obs_b

            acts_seq[t] = a_np if B > 1 else (a_np if is_discrete else a_np[0:1, :])
            rew_ext_seq[t] = np.asarray(rewards, dtype=np.float32).reshape(B)
            done_seq[t] = np.asarray(done_flags, dtype=np.float32).reshape(B)
            # store terminals and truncations separately for timeout-aware GAE
            terms_seq[t] = terms_b.astype(np.float32, copy=False)
            truncs_seq[t] = truncs_b.astype(np.float32, copy=False)

            if not is_image:
                next_obs_seq[t] = next_obs_b_norm.astype(np.float32)
            else:
                # IMPORTANT: explicitly copy next observations before buffering for the same reason.
                next_obs_seq_list.append(np.array(next_obs_b_norm, copy=True))

            if r_int_raw_seq is not None:
                r_step = intrinsic_module.compute_impact_binned(  # type: ignore[union-attr]
                    obs_b_norm,
                    next_obs_b_norm,
                    dones=done_flags,
                    reduction="none",
                )
                r_step_np = r_step.detach().cpu().numpy().reshape(B).astype(np.float32)
                # Recommended safety: do not grant intrinsic reward on done transitions.
                r_step_np[done_flags] = 0.0
                r_int_raw_seq[t] = r_step_np

            # Continue stepping with the env-returned observation (may be an auto-reset obs).
            obs = next_obs_env
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

        # --- Image rollout integrity check (best-effort, warning only) ---
        if is_image:
            try:
                if T >= 2:
                    # Sample a small number of consecutive pairs for env 0 to keep cost bounded.
                    sample_pairs = int(min(16, T - 1))
                    idxs = np.linspace(0, T - 2, sample_pairs, dtype=np.int64)
                    b0 = 0
                    same = 0
                    for ti in idxs:
                        if np.array_equal(obs_seq_final[ti, b0], obs_seq_final[ti + 1, b0]):
                            same += 1
                    frac_same = float(same) / float(sample_pairs)
                    if frac_same > 0.9:
                        logger.warning(
                            "Image rollout sanity check: %.0f%% of sampled consecutive frames are identical. "
                            "If unexpected, check VectorEnv copy semantics and rollout buffering.",
                            100.0 * frac_same,
                        )
            except Exception:
                # Never fail training due to a diagnostic-only check.
                pass

        # enforce time-major (T,B,...) everywhere
        obs_seq_final = _ensure_time_major_np(obs_seq_final, T, B, "obs_seq")
        next_obs_seq_final = _ensure_time_major_np(next_obs_seq_final, T, B, "next_obs_seq")
        rew_ext_seq = _ensure_time_major_np(rew_ext_seq, T, B, "rewards")
        done_seq = _ensure_time_major_np(done_seq, T, B, "dones")
        terms_seq = _ensure_time_major_np(terms_seq, T, B, "terminals")
        truncs_seq = _ensure_time_major_np(truncs_seq, T, B, "truncations")
        if r_int_raw_seq is not None:
            r_int_raw_seq = _ensure_time_major_np(r_int_raw_seq, T, B, "r_int_raw")

        # --- Intrinsic compute/update (optional) ---
        r_int_raw_flat = None
        r_int_scaled_flat = None
        mod_metrics = {}
        if intrinsic_module is not None and use_intrinsic:
            if method_l == "ride":
                r_int_raw_flat = r_int_raw_seq.reshape(T * B).astype(np.float32)  # type: ignore[union-attr]
            else:
                obs_flat = obs_seq_final.reshape((T * B,) + obs_shape)
                next_obs_flat = next_obs_seq_final.reshape((T * B,) + obs_shape)
                acts_flat = acts_seq.reshape(T * B) if is_discrete else acts_seq.reshape(T * B, -1)
                r_int_raw_t = compute_intrinsic_batch(
                    intrinsic_module,
                    method_l,
                    obs_flat,
                    next_obs_flat,
                    acts_flat,
                )
                r_int_raw_flat = r_int_raw_t.detach().cpu().numpy().astype(np.float32)

            # Recommended safety: no intrinsic reward on done transitions (terminated OR truncated).
            done_mask_flat = (done_seq.reshape(T * B) > 0.0)
            r_int_raw_flat = np.asarray(r_int_raw_flat, dtype=np.float32)
            r_int_raw_flat[done_mask_flat] = 0.0

            r_clip = float(cfg.intrinsic.r_clip)
            outputs_norm = bool(getattr(intrinsic_module, "outputs_normalized", False))
            if outputs_norm:
                # Module already normalized -> only clip + scale
                r_int_scaled_flat = eta * np.clip(r_int_raw_flat, -r_clip, r_clip)
            else:
                # Global RMS path
                int_rms.update(r_int_raw_flat)
                r_int_norm_flat = int_rms.normalize(r_int_raw_flat)
                r_int_scaled_flat = eta * np.clip(r_int_norm_flat, -r_clip, r_clip)

            try:
                if method_l == "ride":
                    obs_flat = obs_seq_final.reshape((T * B,) + obs_shape)
                    next_obs_flat = next_obs_seq_final.reshape((T * B,) + obs_shape)
                    acts_flat = acts_seq.reshape(T * B) if is_discrete else acts_seq.reshape(T * B, -1)
                mod_metrics = update_module(
                    intrinsic_module,
                    method_l,
                    obs_flat,  # type: ignore[arg-type]
                    next_obs_flat,
                    acts_flat,
                )
            except Exception:
                mod_metrics = {}

        # --- Rewards & GAE ---
        if r_int_scaled_flat is not None:
            rew_total_seq = rew_ext_seq + r_int_scaled_flat.reshape(T, B)
        else:
            rew_total_seq = rew_ext_seq

        # Time-limit-aware GAE: separate terminals vs truncations and bootstrap on timeouts.
        gae_batch = {
            "obs": obs_seq_final,
            "next_observations": next_obs_seq_final,
            "rewards": rew_total_seq,
            "terminals": terms_seq,
            "truncations": truncs_seq,
        }
        adv, v_targets = compute_gae(
            gae_batch,
            value,
            gamma=float(cfg.ppo.gamma),
            lam=float(cfg.ppo.gae_lambda),
            bootstrap_on_timeouts=True,
        )

        obs_flat_for_ppo = (
            obs_seq_final.reshape(T * B, -1)
            if not is_image
            else obs_seq_final.reshape((T * B,) + obs_shape)
        )
        acts_flat_for_ppo = acts_seq.reshape(T * B) if is_discrete else acts_seq.reshape(T * B, -1)
        batch = {
            "obs": obs_flat_for_ppo,
            "actions": acts_flat_for_ppo,
            "rewards": rew_total_seq.reshape(T * B),
            "dones": done_seq.reshape(T * B),
        }
        # pass persistent optimizers; capture optional stats for logging
        ppo_stats = ppo_update(
            policy,
            value,
            batch,
            adv,
            v_targets,
            cfg.ppo,
            optimizers=(pol_opt, val_opt),
            return_stats=True,
        )
        update_idx += 1

        # --- Logging ---
        with torch.no_grad():
            # Last-step entropy (fast to compute).
            last_obs = obs_seq_final[-1]  # [B,...]
            if is_image:
                ent_last = float(policy.entropy(last_obs).mean().item())
            else:
                last_obs_t = torch.as_tensor(last_obs, device=device, dtype=torch.float32)
                ent_last = float(policy.entropy(last_obs_t).mean().item())

            # Update-wide entropy: compute only if TB enabled or small batch; subsample large batches.
            ENTROPY_MAX_SAMPLES = 1024
            total_samples = int(T * B)
            want_update_entropy = (ml.tb is not None) or (total_samples <= ENTROPY_MAX_SAMPLES)

            ent_mean_update = float("nan")
            if want_update_entropy:
                if total_samples > ENTROPY_MAX_SAMPLES:
                    # Evenly spaced deterministic subsample to bound cost
                    idx = np.linspace(0, total_samples - 1, ENTROPY_MAX_SAMPLES, dtype=np.int64)
                    obs_subset = obs_flat_for_ppo[idx]
                else:
                    obs_subset = obs_flat_for_ppo

                if is_image:
                    ent_mean_update = float(policy.entropy(obs_subset).mean().item())
                else:
                    obs_flat_t = torch.as_tensor(obs_subset, device=device, dtype=torch.float32)
                    ent_mean_update = float(policy.entropy(obs_flat_t).mean().item())

        log_payload = {
            # Renamed for clarity: entropy on last obs vs mean over update
            "entropy_last": float(ent_last),
            "entropy_update_mean": float(ent_mean_update),
            "reward_mean": float(rew_ext_seq.mean()),
            "reward_total_mean": float(rew_total_seq.mean()),
        }

        # Record intrinsic normalization mode in scalars/TB for downstream analysis.
        log_payload["intrinsic_norm_mode"] = intrinsic_norm_mode
        if intrinsic_outputs_normalized_flag is not None:
            log_payload["intrinsic_outputs_normalized"] = bool(intrinsic_outputs_normalized_flag)

        # add PPO monitor stats when available (+ percentage for clip_frac)
        if isinstance(ppo_stats, dict):
            try:
                clip_frac = float(ppo_stats.get("clip_frac", float("nan")))
                log_payload.update(
                    {
                        "approx_kl": float(ppo_stats.get("approx_kl", float("nan"))),
                        "clip_frac": clip_frac,
                        "clip_frac_pct": (100.0 * clip_frac) if np.isfinite(clip_frac) else float("nan"),
                        # Return update-wide means for losses (not just last minibatch)
                        "entropy_minibatch_mean": float(ppo_stats.get("entropy", float("nan"))),
                        "ppo_policy_loss": float(ppo_stats.get("policy_loss", float("nan"))),
                        "ppo_value_loss": float(ppo_stats.get("value_loss", float("nan"))),
                        # KL early-stop + epochs ran
                        "ppo_early_stop": float(ppo_stats.get("early_stop", 0.0)),
                        "ppo_epochs_ran": float(ppo_stats.get("epochs_ran", float("nan"))),
                    }
                )
            except Exception:
                pass

        if r_int_raw_flat is not None and r_int_scaled_flat is not None:
            outputs_norm = bool(getattr(intrinsic_module, "outputs_normalized", False)) if intrinsic_module else False

            # Prefer module-provided RMS diagnostics to avoid double-normalization ambiguity.
            r_int_rms_val = float(int_rms.rms)
            if outputs_norm and intrinsic_module is not None:
                # Single-RMS modules (e.g., RND with internal normalization)
                if hasattr(intrinsic_module, "rms"):
                    try:
                        r_int_rms_val = float(getattr(intrinsic_module, "rms"))
                    except Exception:
                        pass
                # Multi-component modules (Proposed): log both and average for r_int_rms
                if hasattr(intrinsic_module, "impact_rms") and hasattr(intrinsic_module, "lp_rms"):
                    try:
                        imp_rms_val = float(getattr(intrinsic_module, "impact_rms"))
                        lp_rms_val = float(getattr(intrinsic_module, "lp_rms"))
                        # Explicitly log both RMS tracks for Proposed
                        log_payload["impact_rms"] = imp_rms_val
                        log_payload["lp_rms"] = lp_rms_val
                        r_int_rms_val = 0.5 * (imp_rms_val + lp_rms_val)
                    except Exception:
                        pass
                # Fallback: some modules expose only lp_rms (RIAC)
                elif hasattr(intrinsic_module, "lp_rms"):
                    try:
                        r_int_rms_val = float(getattr(intrinsic_module, "lp_rms"))
                    except Exception:
                        pass

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

        if (
            intrinsic_module is not None
            and use_intrinsic
            and method_l == "proposed"
            and hasattr(intrinsic_module, "gate_rate")
        ):
            try:
                gr = float(getattr(intrinsic_module, "gate_rate"))
                log_payload["gate_rate"] = gr
                log_payload["gate_rate_pct"] = 100.0 * gr
            except Exception:
                pass

        ml.log(step=int(global_step), **log_payload)

        # --- Periodic console progress log ---
        if update_idx % int(log_every_updates) == 0 or global_step >= int(total_steps):
            now = time.time()
            elapsed = max(now - start_wall, 1e-6)
            avg_sps = float(global_step) / elapsed if global_step > 0 else 0.0
            delta_steps = int(global_step - last_log_step)
            delta_t = max(now - last_log_time, 1e-6)
            recent_sps = float(delta_steps) / delta_t if delta_steps > 0 else 0.0

            approx_kl = float(log_payload.get("approx_kl", float("nan")))
            clip_frac = float(log_payload.get("clip_frac", float("nan")))
            r_total = float(log_payload.get("reward_total_mean", float("nan")))
            r_int_mean = float(log_payload.get("r_int_mean", 0.0)) if "r_int_mean" in log_payload else 0.0

            logger.info(
                "Train progress: step=%d update=%d avg_sps=%.1f recent_sps=%.1f "
                "reward_total_mean=%.3f r_int_mean=%.3f approx_kl=%.4f clip_frac=%.3f",
                int(global_step),
                int(update_idx),
                avg_sps,
                recent_sps,
                r_total,
                r_int_mean,
                approx_kl,
                clip_frac,
            )
            last_log_time = now
            last_log_step = global_step

        # RIAC diagnostics cadence
        try:
            if (
                intrinsic_module is not None
                and use_intrinsic
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
            payload = _build_checkpoint_payload(
                cfg,
                global_step=int(global_step),
                update_idx=int(update_idx),
                policy=policy,
                value=value,
                is_image=is_image,
                obs_norm=obs_norm,
                int_rms=int_rms,
                pol_opt=pol_opt,
                val_opt=val_opt,
                intrinsic_module=intrinsic_module,
                method_l=method_l,
            )
            ckpt_path = ckpt.save(step=int(global_step), payload=payload)
            logger.info("Saved checkpoint at step=%d to %s", int(global_step), ckpt_path)

    # Final checkpoint
    payload = _build_checkpoint_payload(
        cfg,
        global_step=int(global_step),
        update_idx=int(update_idx),
        policy=policy,
        value=value,
        is_image=is_image,
        obs_norm=obs_norm,
        int_rms=int_rms,
        pol_opt=pol_opt,
        val_opt=val_opt,
        intrinsic_module=intrinsic_module,
        method_l=method_l,
    )
    final_ckpt_path = ckpt.save(step=int(global_step), payload=payload)
    logger.info("Saved final checkpoint at step=%d to %s", int(global_step), final_ckpt_path)

    # Persist updated counters/obs back into the session object.
    session.obs = obs
    session.global_step = int(global_step)
    session.update_idx = int(update_idx)
