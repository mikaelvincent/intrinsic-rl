"""End-to-end PPO training loop with intrinsic rewards.

This module wires together environments, PPO models, intrinsic reward
modules, logging, and checkpointing into a single training entry point.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from math import ceil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim import Adam  # persistent PPO optimizers

from irl.algo.advantage import compute_gae
from irl.algo.ppo import ppo_update
from irl.cfg import Config, ConfigError, to_dict, validate_config
from irl.envs import EnvManager
from irl.models import PolicyNetwork, ValueNetwork
from irl.utils.checkpoint import CheckpointManager
from irl.utils.loggers import (
    MetricLogger,
    get_logger,
    log_domain_randomization,
    log_intrinsic_norm_hint,
    log_resume_intrinsic_warning,
    log_resume_loaded,
    log_resume_no_checkpoint,
    log_resume_optimizer_warning,
    log_resume_state_restored,
)
from irl.intrinsic import (  # type: ignore
    is_intrinsic_method,
    create_intrinsic_module,
    compute_intrinsic_batch,
    update_module,
    RunningRMS,
)

from .build import ensure_device, default_run_dir, single_spaces, ensure_mujoco_gl
from .obs_norm import RunningObsNorm

# config hash helper for resume verification
from irl.utils.checkpoint import compute_cfg_hash
# Unified seeding helper (Python, NumPy, and PyTorch)
from irl.utils.determinism import seed_everything


def _is_image_space(space) -> bool:
    return hasattr(space, "shape") and len(space.shape) >= 2


# Ensure we have a module-level logger for this trainer.
_LOG = get_logger(__name__)


def _move_optimizer_state_to_device(opt: Adam, device: torch.device) -> None:
    """Move all optimizer state tensors in ``opt`` onto ``device``."""
    for state in opt.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


# Helper: enforce (T, B, ...) time-major layout for NumPy arrays.


def _ensure_time_major_np(x: np.ndarray, T: int, B: int, name: str) -> np.ndarray:
    """Return array with leading dims (T,B,...) from (T,B,...) or (B,T,...).

    Raises ValueError with a clear message if shapes are inconsistent.
    """
    if x.ndim < 2:
        raise ValueError(f"{name}: expected at least 2 dims (T,B,...), got shape={x.shape}")
    t0, b0 = int(x.shape[0]), int(x.shape[1])
    if t0 == T and b0 == B:
        return x
    if t0 == B and b0 == T:
        # auto-fix common mistake: batch-major provided instead of time-major
        return np.swapaxes(x, 0, 1)
    raise ValueError(
        f"{name}: inconsistent leading dims. Expected (T,B)=({T},{B}); got {tuple(x.shape[:2])}. "
        "Ensure time is the first axis and batch is second."
    )


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

    # Unified seeding: Python, NumPy, and PyTorch.
    deterministic = False
    try:
        exp_cfg = getattr(cfg, "exp", None)
        if exp_cfg is not None:
            deterministic = bool(getattr(exp_cfg, "deterministic", False))
    except Exception:
        deterministic = False
    seed_everything(int(cfg.seed), deterministic=deterministic)

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
            log_resume_loaded(resume_step, ckpt.latest_path)
        except FileNotFoundError:
            log_resume_no_checkpoint()
        except Exception:
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
        car_action_set=cfg.env.car_discrete_action_set,
        render_mode=None,
        async_vector=False,
        make_kwargs=None,
    )
    env = manager.make()
    obs_space, act_space = single_spaces(env)

    is_image = _is_image_space(obs_space)

    policy = PolicyNetwork(obs_space, act_space).to(device)
    value = ValueNetwork(obs_space).to(device)

    # Persistent PPO optimizers (reused each update, saved in checkpoints)
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
                # gating thresholds
                gate_tau_lp_mult=float(cfg.intrinsic.gate.tau_lp_mult),
                gate_tau_s=float(cfg.intrinsic.gate.tau_s),
                gate_hysteresis_up_mult=float(cfg.intrinsic.gate.hysteresis_up_mult),
                gate_min_consec_to_gate=int(cfg.intrinsic.gate.min_consec_to_gate),
                gate_min_regions_for_gating=int(cfg.intrinsic.gate.min_regions_for_gating),
                # normalization & gating toggles (Proposed)
                normalize_inside=bool(cfg.intrinsic.normalize_inside),
                gating_enabled=bool(cfg.intrinsic.gate.enabled),
            )
            if not use_intrinsic:
                _LOG.warning(
                    "Method %r selected but intrinsic.eta=%.3g; intrinsic will be computed "
                    "but ignored in the total reward.",
                    method_l,
                    eta,
                )
        except Exception as exc:
            _LOG.warning(
                "Failed to create intrinsic module %r (%s). Continuing without intrinsic.",
                method_l,
                exc,
            )
            intrinsic_module = None
            use_intrinsic = False

    int_rms = RunningRMS(beta=0.99, eps=1e-8)

    # --- Run dir, logging, checkpoints ---
    ml = MetricLogger(run_dir, cfg.logging)
    ml.log_hparams(to_dict(cfg))

    # --- Reset env(s) & init norm ---
    printed_dr_hint = False  # one-time DR diagnostics notice (if provided by wrapper)
    printed_intr_norm_hint = False  # one-time intrinsic normalization hint

    obs, info = env.reset()
    try:
        if isinstance(info, dict) and ("dr_applied" in info) and not printed_dr_hint:
            diag = info.get("dr_applied")
            msg = ""
            if isinstance(diag, dict):
                mj = int(diag.get("mujoco", 0))
                b2 = int(diag.get("box2d", 0))
                msg = f"mujoco={mj}, box2d={b2}"
            elif isinstance(diag, (list, tuple)):
                # Vector envs may return per-env diagnostics; aggregate counts
                mj = 0
                b2 = 0
                n = 0
                for d in diag:
                    if isinstance(d, dict):
                        mj += int(d.get("mujoco", 0))
                        b2 += int(d.get("box2d", 0))
                        n += 1
                msg = f"mujoco={mj}, box2d={b2} (across {n} envs)"
            else:
                msg = str(diag)
            log_domain_randomization(msg)
            printed_dr_hint = True
    except Exception:
        # Diagnostics are best-effort; ignore unexpected info formats
        pass

    # Print once if intrinsic module emits raw outputs (trainer will normalize)
    if intrinsic_module is not None and not printed_intr_norm_hint:
        try:
            outputs_norm_flag = bool(
                getattr(intrinsic_module, "outputs_normalized", False)
            )
            if not outputs_norm_flag:
                log_intrinsic_norm_hint(method_l, outputs_norm_flag)
                printed_intr_norm_hint = True
        except Exception:
            pass

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
            _LOG.warning(
                "Could not load policy/value weights from checkpoint; continuing with "
                "fresh initialization."
            )

        # Intrinsic global normalizer
        try:
            int_rms.load_state_dict(resume_payload.get("intrinsic_norm", {}))
        except Exception:
            pass

        # Observation normalizer (only for vector observations)
        try:
            if not is_image and resume_payload.get("obs_norm") is not None:
                on = resume_payload["obs_norm"]
                import numpy as _np

                obs_norm.count = float(on.get("count", obs_norm.count))  # type: ignore[union-attr]
                obs_norm.mean = _np.asarray(
                    on.get("mean", obs_norm.mean), dtype=_np.float64
                )  # type: ignore[union-attr]
                obs_norm.var = _np.asarray(
                    on.get("var", obs_norm.var), dtype=_np.float64
                )  # type: ignore[union-attr]
        except Exception:
            pass

        # Intrinsic module state (if present & compatible)
        try:
            intr = resume_payload.get("intrinsic")
            if intrinsic_module is not None and isinstance(intr, dict):
                if intr.get("method") == method_l and "state_dict" in intr:
                    intrinsic_module.load_state_dict(intr["state_dict"])  # type: ignore[attr-defined]
                    intrinsic_module.to(device)  # type: ignore[attr-defined]
        except Exception:
            log_resume_intrinsic_warning(method_l)

        # Load optimizer states (if present) and move tensors to correct device
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
            log_resume_optimizer_warning()

        # Counters
        global_step = int(resume_step)
        try:
            update_idx = int((resume_payload.get("meta") or {}).get("updates", update_idx))
        except Exception:
            update_idx = update_idx

        log_resume_state_restored(global_step)

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
                else np.zeros(
                    (T, B, int(act_space.shape[0])), dtype=np.float32
                )
            )
            rew_ext_seq = np.zeros((T, B), dtype=np.float32)
            done_seq = np.zeros((T, B), dtype=np.float32)
            # new: separate terminal and truncation masks for GAE
            terms_seq = np.zeros((T, B), dtype=np.float32)
            truncs_seq = np.zeros((T, B), dtype=np.float32)
            r_int_raw_seq = (
                np.zeros((T, B), dtype=np.float32)
                if (intrinsic_module is not None and method_l == "ride")
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
                    obs_seq_list.append(obs_b_norm)

                with torch.no_grad():
                    if is_image:
                        # Pass raw images; the policy will preprocess (layout + scaling)
                        a_tensor, _ = policy.act(obs_b_norm)
                    else:
                        obs_tensor = torch.as_tensor(
                            obs_b_norm, device=device, dtype=torch.float32
                        )
                        a_tensor, _ = policy.act(obs_tensor)
                a_np = a_tensor.detach().cpu().numpy()
                if is_discrete:
                    a_np = a_np.astype(np.int64).reshape(B)
                else:
                    a_np = a_np.reshape(B, -1).astype(np.float32)

                next_obs, rewards, terms, truncs, _ = env.step(
                    a_np if B > 1 else a_np[0]
                )
                done_flags = np.asarray(terms, dtype=bool) | np.asarray(
                    truncs, dtype=bool
                )

                next_obs_b = next_obs if B > 1 else next_obs[None, ...]
                if not is_image:
                    obs_norm.update(next_obs_b)  # type: ignore[union-attr]
                    next_obs_b_norm = obs_norm.normalize(next_obs_b)  # type: ignore[union-attr]
                else:
                    next_obs_b_norm = next_obs_b

                acts_seq[t] = (
                    a_np
                    if B > 1
                    else (a_np if is_discrete else a_np[0:1, :])
                )
                rew_ext_seq[t] = np.asarray(
                    rewards, dtype=np.float32
                ).reshape(B)
                done_seq[t] = np.asarray(
                    done_flags, dtype=np.float32
                ).reshape(B)
                # store terminals and truncations separately for timeout-aware GAE
                terms_seq[t] = np.asarray(terms, dtype=np.float32).reshape(B)
                truncs_seq[t] = np.asarray(truncs, dtype=np.float32).reshape(B)

                if not is_image:
                    next_obs_seq[t] = next_obs_b_norm.astype(np.float32)
                else:
                    next_obs_seq_list.append(
                        next_obs_b_norm.astype(
                            next_obs_b_norm.dtype, copy=False
                        )
                    )

                if r_int_raw_seq is not None:
                    r_step = intrinsic_module.compute_impact_binned(  # type: ignore[union-attr]
                        obs_b_norm,
                        next_obs_b_norm,
                        dones=done_flags,
                        reduction="none",
                    )
                    r_int_raw_seq[t] = (
                        r_step.detach()
                        .cpu()
                        .numpy()
                        .reshape(B)
                        .astype(np.float32)
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

            # enforce time-major (T,B,...) everywhere
            obs_seq_final = _ensure_time_major_np(obs_seq_final, T, B, "obs_seq")
            next_obs_seq_final = _ensure_time_major_np(
                next_obs_seq_final, T, B, "next_obs_seq"
            )
            rew_ext_seq = _ensure_time_major_np(
                rew_ext_seq, T, B, "rewards"
            )
            done_seq = _ensure_time_major_np(done_seq, T, B, "dones")
            terms_seq = _ensure_time_major_np(terms_seq, T, B, "terminals")
            truncs_seq = _ensure_time_major_np(truncs_seq, T, B, "truncations")
            if r_int_raw_seq is not None:
                r_int_raw_seq = _ensure_time_major_np(
                    r_int_raw_seq, T, B, "r_int_raw"
                )

            # --- Intrinsic compute/update (optional) ---
            r_int_raw_flat = None
            r_int_scaled_flat = None
            mod_metrics = {}
            if intrinsic_module is not None:
                if method_l == "ride":
                    r_int_raw_flat = r_int_raw_seq.reshape(T * B).astype(  # type: ignore[union-attr]
                        np.float32
                    )
                else:
                    obs_flat = obs_seq_final.reshape((T * B,) + obs_shape)
                    next_obs_flat = next_obs_seq_final.reshape(
                        (T * B,) + obs_shape
                    )
                    acts_flat = (
                        acts_seq.reshape(T * B)
                        if is_discrete
                        else acts_seq.reshape(T * B, -1)
                    )
                    r_int_raw_t = compute_intrinsic_batch(
                        intrinsic_module,
                        method_l,
                        obs_flat,
                        next_obs_flat,
                        acts_flat,
                    )
                    r_int_raw_flat = (
                        r_int_raw_t.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                r_clip = float(cfg.intrinsic.r_clip)
                outputs_norm = bool(
                    getattr(intrinsic_module, "outputs_normalized", False)
                )
                if outputs_norm:
                    # Module already normalized -> only clip + scale
                    r_int_scaled_flat = eta * np.clip(
                        r_int_raw_flat, -r_clip, r_clip
                    )
                else:
                    # Global RMS path
                    int_rms.update(r_int_raw_flat)
                    r_int_norm_flat = int_rms.normalize(r_int_raw_flat)
                    r_int_scaled_flat = eta * np.clip(
                        r_int_norm_flat, -r_clip, r_clip
                    )

                try:
                    if method_l == "ride":
                        obs_flat = obs_seq_final.reshape((T * B,) + obs_shape)
                        next_obs_flat = next_obs_seq_final.reshape(
                            (T * B,) + obs_shape
                        )
                        acts_flat = (
                            acts_seq.reshape(T * B)
                            if is_discrete
                            else acts_seq.reshape(T * B, -1)
                        )
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
                rew_total_seq = (
                    rew_ext_seq + r_int_scaled_flat.reshape(T, B)
                )
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
            acts_flat_for_ppo = (
                acts_seq.reshape(T * B)
                if is_discrete
                else acts_seq.reshape(T * B, -1)
            )
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
                    last_obs_t = torch.as_tensor(
                        last_obs, device=device, dtype=torch.float32
                    )
                    ent_last = float(
                        policy.entropy(last_obs_t).mean().item()
                    )

                # Update-wide entropy: compute only if TB enabled or small batch; subsample large batches.
                ENTROPY_MAX_SAMPLES = 1024
                total_samples = int(T * B)
                want_update_entropy = (ml.tb is not None) or (
                    total_samples <= ENTROPY_MAX_SAMPLES
                )

                ent_mean_update = float("nan")
                if want_update_entropy:
                    if total_samples > ENTROPY_MAX_SAMPLES:
                        # Evenly spaced deterministic subsample to bound cost
                        idx = np.linspace(
                            0,
                            total_samples - 1,
                            ENTROPY_MAX_SAMPLES,
                            dtype=np.int64,
                        )
                        obs_subset = obs_flat_for_ppo[idx]
                    else:
                        obs_subset = obs_flat_for_ppo

                    if is_image:
                        ent_mean_update = float(
                            policy.entropy(obs_subset).mean().item()
                        )
                    else:
                        obs_flat_t = torch.as_tensor(
                            obs_subset,
                            device=device,
                            dtype=torch.float32,
                        )
                        ent_mean_update = float(
                            policy.entropy(obs_flat_t).mean().item()
                        )

            log_payload = {
                # Renamed for clarity: entropy on last obs vs mean over update
                "entropy_last": float(ent_last),
                "entropy_update_mean": float(ent_mean_update),
                "reward_mean": float(rew_ext_seq.mean()),
                "reward_total_mean": float(rew_total_seq.mean()),
            }
            # add PPO monitor stats when available (+ percentage for clip_frac)
            if isinstance(ppo_stats, dict):
                try:
                    clip_frac = float(ppo_stats.get("clip_frac", float("nan")))
                    log_payload.update(
                        {
                            "approx_kl": float(
                                ppo_stats.get("approx_kl", float("nan"))
                            ),
                            "clip_frac": clip_frac,
                            "clip_frac_pct": (
                                (100.0 * clip_frac)
                                if np.isfinite(clip_frac)
                                else float("nan")
                            ),
                            # Return update-wide means for losses (not just last minibatch)
                            "entropy_minibatch_mean": float(
                                ppo_stats.get("entropy", float("nan"))
                            ),
                            "ppo_policy_loss": float(
                                ppo_stats.get("policy_loss", float("nan"))
                            ),
                            "ppo_value_loss": float(
                                ppo_stats.get("value_loss", float("nan"))
                            ),
                            # KL early-stop + epochs ran
                            "ppo_early_stop": float(
                                ppo_stats.get("early_stop", 0.0)
                            ),
                            "ppo_epochs_ran": float(
                                ppo_stats.get("epochs_ran", float("nan"))
                            ),
                        }
                    )
                except Exception:
                    pass

            if r_int_raw_flat is not None and r_int_scaled_flat is not None:
                outputs_norm = (
                    bool(
                        getattr(intrinsic_module, "outputs_normalized", False)
                    )
                    if intrinsic_module
                    else False
                )

                # Prefer module-provided RMS diagnostics to avoid double-normalization ambiguity.
                r_int_rms_val = float(int_rms.rms)
                if outputs_norm and intrinsic_module is not None:
                    # Single-RMS modules (e.g., RND with internal normalization)
                    if hasattr(intrinsic_module, "rms"):
                        try:
                            r_int_rms_val = float(
                                getattr(intrinsic_module, "rms")
                            )
                        except Exception:
                            pass
                    # Multi-component modules (Proposed): log both and average for r_int_rms
                    if hasattr(intrinsic_module, "impact_rms") and hasattr(
                        intrinsic_module, "lp_rms"
                    ):
                        try:
                            imp_rms_val = float(
                                getattr(intrinsic_module, "impact_rms")
                            )
                            lp_rms_val = float(
                                getattr(intrinsic_module, "lp_rms")
                            )
                            # Explicitly log both RMS tracks for Proposed
                            log_payload["impact_rms"] = imp_rms_val
                            log_payload["lp_rms"] = lp_rms_val
                            r_int_rms_val = 0.5 * (imp_rms_val + lp_rms_val)
                        except Exception:
                            pass
                    # Fallback: some modules expose only lp_rms (RIAC)
                    elif hasattr(intrinsic_module, "lp_rms"):
                        try:
                            r_int_rms_val = float(
                                getattr(intrinsic_module, "lp_rms")
                            )
                        except Exception:
                            pass

                log_payload.update(
                    {
                        "r_int_raw_mean": float(
                            np.mean(r_int_raw_flat)
                        ),
                        "r_int_mean": float(
                            np.mean(r_int_scaled_flat)
                        ),
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

            # RIAC diagnostics cadence
            try:
                if (
                    intrinsic_module is not None
                    and method_l == "riac"
                    and int(global_step) % int(cfg.logging.csv_interval)
                    == 0
                    and hasattr(
                        intrinsic_module, "export_diagnostics"
                    )
                ):
                    diag_dir = run_dir / "diagnostics"
                    intrinsic_module.export_diagnostics(  # type: ignore[union-attr]
                        diag_dir, step=int(global_step)
                    )
            except Exception:
                pass

            # --- Checkpoint cadence ---
            if ckpt.should_save(int(global_step)):
                payload = {
                    "step": int(global_step),
                    "policy": policy.state_dict(),
                    "value": value.state_dict(),
                    "cfg": to_dict(cfg),
                    "cfg_hash": compute_cfg_hash(
                        to_dict(cfg)
                    ),  # store config hash
                    # Only persist obs_norm for vector observations
                    "obs_norm": None
                    if is_image
                    else {
                        "count": obs_norm.count,
                        "mean": obs_norm.mean,
                        "var": obs_norm.var,
                    },  # type: ignore[union-attr]
                    "intrinsic_norm": int_rms.state_dict(),
                    "meta": {"updates": update_idx},
                    # persist PPO optimizers
                    "optimizers": {
                        "policy": pol_opt.state_dict(),
                        "value": val_opt.state_dict(),
                    },
                }
                if intrinsic_module is not None and hasattr(
                    intrinsic_module, "state_dict"
                ):
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
            "cfg_hash": compute_cfg_hash(
                to_dict(cfg)
            ),  # store config hash
            "obs_norm": None
            if is_image
            else {
                "count": obs_norm.count,
                "mean": obs_norm.mean,
                "var": obs_norm.var,
            },  # type: ignore[union-attr]
            "intrinsic_norm": int_rms.state_dict(),
            "meta": {"updates": update_idx},
            "optimizers": {
                "policy": pol_opt.state_dict(),
                "value": val_opt.state_dict(),
            },
        }
        if intrinsic_module is not None and hasattr(
            intrinsic_module, "state_dict"
        ):
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

    return run_dir
