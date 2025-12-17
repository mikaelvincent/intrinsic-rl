from __future__ import annotations

import time
from math import ceil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from irl.cfg import Config
from irl.models import PolicyNetwork
from irl.utils.checkpoint_schema import build_checkpoint_payload

from .metrics import build_log_payload
from .rollout import RolloutBatch, collect_rollout
from .training_setup import TrainingSession
from .update_steps import compute_advantages, compute_intrinsic_rewards, ppo_step


# CUDA kernels launch asynchronously; synchronize only for timing/profiling.
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

    obs = session.obs
    global_step = int(session.global_step)
    update_idx = int(session.update_idx)

    actor_policy: Optional[PolicyNetwork] = None
    if device.type != "cpu":
        actor_policy = PolicyNetwork(obs_space, act_space).to(torch.device("cpu"))
        actor_policy.eval()
        for p in actor_policy.parameters():
            p.requires_grad = False
        logger.info("Using CPU actor policy for env stepping; syncing each PPO update.")

    def _sync_actor_from_learner() -> None:
        if actor_policy is None:
            return
        with torch.no_grad():
            sd = policy.state_dict()
            cpu_sd = {k: v.detach().to("cpu") for k, v in sd.items()}
            actor_policy.load_state_dict(cpu_sd, strict=True)

    if int(global_step) == 0 and int(update_idx) == 0 and not ckpt.latest_path.exists():
        payload0 = build_checkpoint_payload(
            cfg,
            global_step=0,
            update_idx=0,
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
        ckpt_path0 = ckpt.save(step=0, payload=payload0)
        logger.info("Saved baseline checkpoint at step=0 to %s", ckpt_path0)

    start_wall = time.time()
    last_log_time = start_wall
    last_log_step = global_step

    while global_step < int(total_steps):
        t_update_start = time.perf_counter()

        t_rollout_total = 0.0
        t_rollout_policy = 0.0
        t_rollout_env_step = 0.0
        t_rollout_intrinsic_step = 0.0
        t_rollout_other = 0.0

        t_intrinsic_compute = 0.0
        t_intrinsic_update = 0.0
        t_gae = 0.0
        t_ppo = 0.0

        t_logging_compute = 0.0
        t_ml_log = 0.0

        _sync_actor_from_learner()

        per_env_steps = min(
            int(cfg.ppo.steps_per_update),
            max(1, ceil((int(total_steps) - int(global_step)) / B)),
        )

        rollout: RolloutBatch = collect_rollout(
            env=env,
            policy=policy,
            actor_policy=actor_policy,
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

        ml_t0 = time.perf_counter()
        ml.log(step=int(global_step), **log_payload)
        t_ml_log = time.perf_counter() - ml_t0

        t_update_total = time.perf_counter() - t_update_start

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
            r_int_mean = (
                float(log_payload.get("r_int_mean", 0.0)) if "r_int_mean" in log_payload else 0.0
            )

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

            intrinsic_total = float(
                t_rollout_intrinsic_step + t_intrinsic_compute + t_intrinsic_update
            )
            logging_total = float(t_logging_compute + t_ml_log)
            logger.info(
                "Timings (s) update=%d: total=%.3f | rollout=%.3f (policy=%.3f, env_step=%.3f, "
                "intr_step=%.3f, other=%.3f) | intrinsic_total=%.3f (batch_compute=%.3f, update=%.3f) | "
                "gae=%.3f | ppo=%.3f | logging=%.3f (compute=%.3f, io=%.3f)",
                int(update_idx),
                float(t_update_total),
                float(t_rollout_total),
                float(t_rollout_policy),
                float(t_rollout_env_step),
                float(t_rollout_intrinsic_step),
                float(t_rollout_other),
                intrinsic_total,
                float(t_intrinsic_compute),
                float(t_intrinsic_update),
                float(t_gae),
                float(t_ppo),
                logging_total,
                float(t_logging_compute),
                float(t_ml_log),
            )

            last_log_time = now
            last_log_step = global_step

        try:
            if (
                intrinsic_module is not None
                and bool(use_intrinsic)
                and str(method_l) == "riac"
                and int(global_step) % int(cfg.logging.csv_interval) == 0
                and hasattr(intrinsic_module, "export_diagnostics")
            ):
                diag_dir = Path(run_dir) / "diagnostics"
                intrinsic_module.export_diagnostics(diag_dir, step=int(global_step))
        except Exception:
            pass

        if ckpt.should_save(int(global_step)):
            payload = build_checkpoint_payload(
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

    payload = build_checkpoint_payload(
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

    session.obs = obs
    session.global_step = int(global_step)
    session.update_idx = int(update_idx)
