from __future__ import annotations

import dataclasses
from dataclasses import replace
from math import ceil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim import Adam  # NEW: persistent PPO optimizers

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

# NEW: config hash helper for resume verification
from irl.utils.checkpoint import compute_cfg_hash


def _is_image_space(space) -> bool:
    return hasattr(space, "shape") and len(space.shape) >= 2


# NEW: ensure loaded optimizer state tensors are on the right device
def _move_optimizer_state_to_device(opt: Adam, device: torch.device) -> None:
    for state in opt.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def train(
    cfg: Config,
    *,
    total_steps: int = 10_000,
    run_dir: Optional[Path] = None,
    resume: bool = False,
) -> Path:
    """Run PPO + optional intrinsic rewards; returns the run directory.

    Parameters
    ----------
    cfg:
        Validated Config object.
    total_steps:
        Absolute target environment steps for this run. If resuming from a checkpoint with
        step=S, training continues until step reaches `total_steps` (no extra offset).
    run_dir:
        Directory for logs/checkpoints. If omitted, a fresh timestamped directory is created.
        When provided with `resume=True`, the latest checkpoint in this directory is loaded.
    resume:
        If True and a latest checkpoint exists in `run_dir`, restore state and continue.
        A config-hash mismatch aborts with a clear error to avoid accidental cross-run resumes.
    """
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

    # ---- Resolve run directory & (optional) resume payload ----
    run_dir = Path(run_dir) if run_dir is not None else default_run_dir(cfg)
    ckpt = CheckpointManager(run_dir, interval_steps=cfg.logging.checkpoint_interval, max_to_keep=3)

    resume_payload: Optional[dict] = None
    resume_step: int = 0

    if resume:
        try:
            # Load on CPU first (cheaper / device-agnostic), verify config hash early.
            payload_cpu, step_cpu = ckpt.load_latest(map_location="cpu")
            # Verify config hash (prefer stored cfg_hash; fallback to hash of embedded cfg).
            current_hash = compute_cfg_hash(to_dict(cfg))
            stored_hash = payload_cpu.get("cfg_hash")
            if stored_hash is None:
                stored_hash = compute_cfg_hash(payload_cpu.get("cfg", {}) or {})
            if str(stored_hash) != str(current_hash):
                raise RuntimeError(
                    "Config hash mismatch when resuming:\n"
                    f"  checkpoint: {stored_hash}\n"
                    f"  current   : {current_hash}\n"
                    "Refuse to resume with a different configuration. "
                    "Supply a matching config or run with --no-resume."
                )
            resume_payload = payload_cpu
            resume_step = int(step_cpu)
            print(f"[resume] Loaded latest checkpoint at step={resume_step} from {ckpt.latest_path}")
        except FileNotFoundError:
            print("[resume] No checkpoint found; starting a new run.")
        except Exception as exc:
            # Fail loudly on intentional config mismatches; user can opt-out via resume=False.
            raise

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

    # NEW: persistent PPO optimizers (reused each update, saved in checkpoints)
    pol_opt = Adam(policy.parameters(), lr=float(cfg.ppo.learning_rate))
    val_opt = Adam(value.parameters(), lr=float(cfg.ppo.learning_rate))

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
    ml = MetricLogger(run_dir, cfg.logging)
    ml.log_hparams(to_dict(cfg))

    # --- Reset env(s) & init norm ---
    obs, _ = env.reset()
    B = int(getattr(env, "num_envs", 1))

    obs_norm = None if is_image else RunningObsNorm(shape=int(obs_space.shape[0]))
    if not is_image:
        first_batch = obs if B > 1 else obs[None, :]
        obs_norm.update(first_batch)  # type: ignore[union-attr]

    # --- If resuming: restore state from checkpoint payload ---
    global_step = 0
    update_idx = 0
    if resume_payload is not None:
        try:
            # Policy / Value
            policy.load_state_dict(resume_payload["policy"])
            value.load_state_dict(resume_payload["value"])
        except Exception:
            print("[resume] Warning: could not load policy/value weights from checkpoint.")
        # Intrinsic global normalizer
        try:
            int_rms.load_state_dict(resume_payload.get("intrinsic_norm", {}))
        except Exception:
            pass
        # Observation normalizer (only for vector observations)
        try:
            if not is_image and resume_payload.get("obs_norm") is not None:
                on = resume_payload["obs_norm"]
                # Ensure dtype/shape correctness
                import numpy as _np

                obs_norm.count = float(on.get("count", obs_norm.count))  # type: ignore[union-attr]
                obs_norm.mean = _np.asarray(on.get("mean", obs_norm.mean), dtype=_np.float64)  # type: ignore[union-attr]
                obs_norm.var = _np.asarray(on.get("var", obs_norm.var), dtype=_np.float64)  # type: ignore[union-attr]
        except Exception:
            pass
        # Intrinsic module state (if present & compatible)
        try:
            intr = resume_payload.get("intrinsic")
            if intrinsic_module is not None and isinstance(intr, dict):
                if intr.get("method") == method_l and "state_dict" in intr:
                    intrinsic_module.load_state_dict(intr["state_dict"])  # type: ignore[attr-defined]
        except Exception:
            print("[resume] Warning: intrinsic module state not restored.")

        # NEW: Load optimizer states (if present) and move tensors to correct device
        try:
            opt_payload = resume_payload.get("optimizers", {})
            pol_state = opt_payload.get("policy", None)
            val_state = opt_payload.get("value", None)
            if pol_state is not None:
                pol_opt.load_state_dict(pol_state)
                _move_optimizer_state_to_device(pol_opt, next(policy.parameters()).device)
            if val_state is not None:
                val_opt.load_state_dict(val_state)
                _move_optimizer_state_to_device(val_opt, next(value.parameters()).device)
        except Exception:
            print("[resume] Warning: PPO optimizer state not restored; continuing with fresh optimizers.")

        # Counters
        global_step = int(resume_step)
        try:
            update_idx = int((resume_payload.get("meta") or {}).get("updates", update_idx))
        except Exception:
            update_idx = update_idx

        print(f"[resume] State restored. Continuing from global_step={global_step}.")

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
            acts_flat_for_ppo = (
                acts_seq.reshape(T * B) if is_discrete else acts_seq.reshape(T * B, -1)
            )
            batch = {
                "obs": obs_flat_for_ppo,
                "actions": acts_flat_for_ppo,
                "rewards": rew_total_seq.reshape(T * B),
                "dones": done_seq.reshape(T * B),
            }
            # NEW: pass persistent optimizers; capture optional stats for logging
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
                last_obs = torch.as_tensor(
                    obs_seq_final[-1], device=device, dtype=torch.float32
                )  # [B,...]
                ent = float(policy.entropy(last_obs).mean().item())

            log_payload = {
                "policy_entropy": float(ent),
                "reward_mean": float(rew_ext_seq.mean()),
                "reward_total_mean": float(rew_total_seq.mean()),
            }
            # NEW: add PPO monitor stats when available
            if isinstance(ppo_stats, dict):
                try:
                    log_payload.update(
                        {
                            "approx_kl": float(ppo_stats.get("approx_kl", float("nan"))),
                            "clip_frac": float(ppo_stats.get("clip_frac", float("nan"))),
                            "ppo_entropy": float(ppo_stats.get("entropy", float("nan"))),
                            "ppo_policy_loss": float(ppo_stats.get("policy_loss", float("nan"))),
                            "ppo_value_loss": float(ppo_stats.get("value_loss", float("nan"))),
                            # NEW: KL early-stop + epochs ran
                            "ppo_early_stop": float(ppo_stats.get("early_stop", 0.0)),
                            "ppo_epochs_ran": float(ppo_stats.get("epochs_ran", float("nan"))),
                        }
                    )
                except Exception:
                    pass

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
                    "cfg_hash": compute_cfg_hash(to_dict(cfg)),  # NEW: store config hash
                    # Only persist obs_norm for vector observations
                    "obs_norm": None if is_image else {"count": obs_norm.count, "mean": obs_norm.mean, "var": obs_norm.var},  # type: ignore[union-attr]
                    "intrinsic_norm": int_rms.state_dict(),
                    "meta": {"updates": update_idx},
                    # NEW: persist PPO optimizers
                    "optimizers": {"policy": pol_opt.state_dict(), "value": val_opt.state_dict()},
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
            "cfg_hash": compute_cfg_hash(to_dict(cfg)),  # NEW: store config hash
            "obs_norm": None if is_image else {"count": obs_norm.count, "mean": obs_norm.mean, "var": obs_norm.var},  # type: ignore[union-attr]
            "intrinsic_norm": int_rms.state_dict(),
            "meta": {"updates": update_idx},
            # NEW: persist PPO optimizers
            "optimizers": {"policy": pol_opt.state_dict(), "value": val_opt.state_dict()},
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
