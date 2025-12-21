from __future__ import annotations

import time

import numpy as np
import torch

from irl.cfg import Config

from .actor_sync import ActorPolicySync
from .checkpointing import (
    maybe_save_baseline_checkpoint,
    maybe_save_periodic_checkpoint,
    save_final_checkpoint,
)
from .diagnostics import maybe_export_intrinsic_diagnostics
from .metrics import build_log_payload
from .rollout import collect_rollout
from .telemetry import ProgressLogger
from .training_setup import TrainingSession
from .update_steps import compute_advantages, compute_intrinsic_rewards, ppo_step


def _maybe_cuda_sync(device: torch.device, enabled: bool) -> None:
    if not enabled:
        return
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass


def run_training_loop(
    cfg: Config,
    session: TrainingSession,
    *,
    total_steps: int,
    logger,
    log_every_updates: int = 10,
) -> None:
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

    profile_cuda_sync = False
    try:
        profile_cuda_sync = bool(getattr(getattr(cfg, "exp", None), "profile_cuda_sync", False))
    except Exception:
        profile_cuda_sync = False

    taper_start = getattr(getattr(cfg, "intrinsic", None), "taper_start_frac", None)
    taper_end = getattr(getattr(cfg, "intrinsic", None), "taper_end_frac", None)
    taper_active = (
        str(method_l).startswith("glpe") and taper_start is not None and taper_end is not None
    )
    if taper_active and hasattr(logger, "info"):
        try:
            start_step = int(float(taper_start) * float(int(total_steps)))
            end_step = int(float(taper_end) * float(int(total_steps)))
        except Exception:
            start_step = -1
            end_step = -1

        logger.info(
            "GLPE intrinsic taper active: start_frac=%.3f end_frac=%.3f (steps %d..%d of %d).",
            float(taper_start),
            float(taper_end),
            int(start_step),
            int(end_step),
            int(total_steps),
        )

    obs = session.obs
    global_step = int(session.global_step)
    update_idx = int(session.update_idx)

    actor_sync = ActorPolicySync.maybe_create(
        obs_space=obs_space,
        act_space=act_space,
        device=device,
        logger=logger,
    )

    maybe_save_baseline_checkpoint(
        cfg,
        ckpt=ckpt,
        policy=policy,
        value=value,
        is_image=bool(is_image),
        obs_norm=obs_norm,
        int_rms=int_rms,
        pol_opt=pol_opt,
        val_opt=val_opt,
        intrinsic_module=intrinsic_module,
        method_l=method_l,
        global_step=int(global_step),
        update_idx=int(update_idx),
        logger=logger,
    )

    progress = ProgressLogger.start(initial_step=int(global_step))

    ep_buf_returns: list[float] = []
    ep_buf_lengths: list[int] = []
    ep_buf_successes: list[float] = []

    while global_step < int(total_steps):
        global_step_start = int(global_step)
        t_update_start = time.perf_counter()

        t_intrinsic_compute = 0.0
        t_intrinsic_update = 0.0
        t_gae = 0.0
        t_ppo = 0.0

        t_logging_compute = 0.0
        t_ml_log = 0.0

        actor_sync.sync_from(policy)

        remaining_steps = int(total_steps) - int(global_step)
        per_env_steps = min(int(cfg.ppo.rollout_steps_per_env), remaining_steps // B)
        if per_env_steps <= 0:
            break

        rollout = collect_rollout(
            env=env,
            policy=policy,
            actor_policy=actor_sync.actor_policy,
            obs=obs,
            obs_space=obs_space,
            act_space=act_space,
            is_image=bool(is_image),
            obs_norm=obs_norm,
            intrinsic_module=intrinsic_module,
            use_intrinsic=bool(use_intrinsic),
            method_l=str(method_l),
            T=int(per_env_steps),
            B=int(B),
            device=device,
            logger=logger,
        )

        if rollout.episode_returns:
            ep_buf_returns.extend(rollout.episode_returns)
        if rollout.episode_lengths:
            ep_buf_lengths.extend(rollout.episode_lengths)
        if rollout.episode_successes:
            ep_buf_successes.extend(rollout.episode_successes)

        obs = rollout.final_env_obs
        global_step += int(rollout.steps_collected)

        t_rollout_total = float(rollout.time_rollout_s)
        t_rollout_policy = float(rollout.time_rollout_policy_s)
        t_rollout_env_step = float(rollout.time_rollout_env_step_s)
        t_rollout_intrinsic_step = float(rollout.time_rollout_intrinsic_step_s)
        t_rollout_other = float(rollout.time_rollout_other_s)

        intrinsic_out = compute_intrinsic_rewards(
            rollout=rollout,
            intrinsic_module=intrinsic_module,
            use_intrinsic=bool(use_intrinsic),
            method_l=str(method_l),
            eta=float(eta),
            r_clip=float(cfg.intrinsic.r_clip),
            int_rms=int_rms,
            device=device,
            profile_cuda_sync=bool(profile_cuda_sync),
            maybe_cuda_sync=_maybe_cuda_sync,
            total_steps=int(total_steps),
            global_step_start=int(global_step_start),
            taper_start_frac=taper_start,
            taper_end_frac=taper_end,
        )
        rewards_total_seq = intrinsic_out.rewards_total_seq
        r_int_raw_flat = intrinsic_out.r_int_raw_flat
        r_int_scaled_flat = intrinsic_out.r_int_scaled_flat
        mod_metrics = intrinsic_out.module_metrics
        t_intrinsic_compute = float(intrinsic_out.time_compute_s)
        t_intrinsic_update = float(intrinsic_out.time_update_s)

        adv_out = compute_advantages(
            rollout=rollout,
            rewards_total_seq=rewards_total_seq,
            value_fn=value,
            gamma=float(cfg.ppo.gamma),
            lam=float(cfg.ppo.gae_lambda),
            device=device,
            profile_cuda_sync=bool(profile_cuda_sync),
            maybe_cuda_sync=_maybe_cuda_sync,
        )
        adv = adv_out.advantages
        v_targets = adv_out.value_targets
        t_gae = float(adv_out.time_s)

        T, B2 = int(rollout.T), int(rollout.B)

        obs_flat_for_ppo = (
            rollout.obs_seq.reshape(T * B2, -1)
            if not is_image
            else rollout.obs_seq.reshape((T * B2,) + tuple(rollout.obs_shape))
        )
        acts_flat_for_ppo = (
            rollout.actions_seq.reshape(T * B2)
            if bool(rollout.is_discrete)
            else rollout.actions_seq.reshape(T * B2, -1)
        )
        batch = {
            "obs": obs_flat_for_ppo,
            "actions": acts_flat_for_ppo,
            "old_log_probs": rollout.old_log_probs_seq.reshape(T * B2),
            "rewards": rewards_total_seq.reshape(T * B2),
            "dones": rollout.dones_seq.reshape(T * B2),
        }

        ppo_out = ppo_step(
            policy=policy,
            value=value,
            batch=batch,
            advantages=adv,
            value_targets=v_targets,
            cfg_ppo=cfg.ppo,
            optimizers=(pol_opt, val_opt),
            device=device,
            profile_cuda_sync=bool(profile_cuda_sync),
            maybe_cuda_sync=_maybe_cuda_sync,
        )
        ppo_stats = ppo_out.stats
        t_ppo = float(ppo_out.time_s)

        update_idx += 1

        log_compute_t0 = time.perf_counter()
        log_payload = build_log_payload(
            policy=policy,
            rollout=rollout,
            rewards_total_seq=rewards_total_seq,
            obs_flat_for_ppo=obs_flat_for_ppo,
            ppo_stats=ppo_stats,
            intrinsic_module=intrinsic_module,
            method_l=str(method_l),
            intrinsic_norm_mode=str(intrinsic_norm_mode),
            intrinsic_taper_weight=float(intrinsic_out.intrinsic_taper_weight_mean),
            intrinsic_eta_effective=float(intrinsic_out.intrinsic_eta_effective_mean),
            intrinsic_outputs_normalized_flag=intrinsic_outputs_normalized_flag,
            int_rms=int_rms,
            r_int_raw_flat=r_int_raw_flat,
            r_int_scaled_flat=r_int_scaled_flat,
            mod_metrics=mod_metrics,
            device=device,
            t_intrinsic_compute=float(t_intrinsic_compute),
            t_intrinsic_update=float(t_intrinsic_update),
            t_gae=float(t_gae),
            t_ppo=float(t_ppo),
        )
        t_logging_compute = time.perf_counter() - log_compute_t0

        if ep_buf_returns:
            ret = np.asarray(ep_buf_returns, dtype=np.float64).reshape(-1)
            lens = (
                np.asarray(ep_buf_lengths, dtype=np.float64).reshape(-1)
                if ep_buf_lengths
                else ret * 0.0
            )
            log_payload["episode_count"] = int(ret.size)
            log_payload["episode_return_mean"] = float(ret.mean()) if ret.size else 0.0
            log_payload["episode_return_std"] = float(ret.std(ddof=0)) if ret.size > 1 else 0.0
            log_payload["episode_length_mean"] = float(lens.mean()) if lens.size else 0.0
            log_payload["episode_length_std"] = float(lens.std(ddof=0)) if lens.size > 1 else 0.0
            log_payload["success_rate"] = (
                float(np.mean(ep_buf_successes)) if ep_buf_successes else 0.0
            )
        else:
            log_payload["episode_count"] = 0
            log_payload["episode_return_mean"] = float("nan")
            log_payload["episode_return_std"] = float("nan")
            log_payload["episode_length_mean"] = float("nan")
            log_payload["episode_length_std"] = float("nan")
            log_payload["success_rate"] = float("nan")

        ml_t0 = time.perf_counter()
        wrote_csv = ml.log(step=int(global_step), **log_payload)
        t_ml_log = time.perf_counter() - ml_t0

        if wrote_csv:
            ep_buf_returns.clear()
            ep_buf_lengths.clear()
            ep_buf_successes.clear()

        t_update_total = time.perf_counter() - t_update_start

        progress.maybe_log(
            logger=logger,
            update_idx=int(update_idx),
            global_step=int(global_step),
            total_steps=int(total_steps),
            log_every_updates=int(log_every_updates),
            log_payload=log_payload,
            t_update_total=float(t_update_total),
            t_rollout_total=float(t_rollout_total),
            t_rollout_policy=float(t_rollout_policy),
            t_rollout_env_step=float(t_rollout_env_step),
            t_rollout_intrinsic_step=float(t_rollout_intrinsic_step),
            t_rollout_other=float(t_rollout_other),
            t_intrinsic_compute=float(t_intrinsic_compute),
            t_intrinsic_update=float(t_intrinsic_update),
            t_gae=float(t_gae),
            t_ppo=float(t_ppo),
            t_logging_compute=float(t_logging_compute),
            t_ml_log=float(t_ml_log),
        )

        maybe_export_intrinsic_diagnostics(
            run_dir=run_dir,
            intrinsic_module=intrinsic_module,
            method_l=str(method_l),
            use_intrinsic=bool(use_intrinsic),
            step=int(global_step),
            csv_interval=int(cfg.logging.csv_interval),
        )

        maybe_save_periodic_checkpoint(
            cfg,
            ckpt=ckpt,
            policy=policy,
            value=value,
            is_image=bool(is_image),
            obs_norm=obs_norm,
            int_rms=int_rms,
            pol_opt=pol_opt,
            val_opt=val_opt,
            intrinsic_module=intrinsic_module,
            method_l=method_l,
            global_step=int(global_step),
            update_idx=int(update_idx),
            logger=logger,
        )

    save_final_checkpoint(
        cfg,
        ckpt=ckpt,
        policy=policy,
        value=value,
        is_image=bool(is_image),
        obs_norm=obs_norm,
        int_rms=int_rms,
        pol_opt=pol_opt,
        val_opt=val_opt,
        intrinsic_module=intrinsic_module,
        method_l=method_l,
        global_step=int(global_step),
        update_idx=int(update_idx),
        logger=logger,
    )

    session.obs = obs
    session.global_step = int(global_step)
    session.update_idx = int(update_idx)
