from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.optim import Adam

from irl.cfg import Config, to_dict
from irl.models import PolicyNetwork, ValueNetwork
from irl.utils.determinism import seed_everything
from irl.utils.spaces import is_image_space

from .build import ensure_mujoco_gl, single_spaces
from .obs_norm import RunningObsNorm
from .resume import _init_run_dir_and_ckpt, _maybe_load_resume_payload, _restore_from_checkpoint
from .session_types import IntrinsicContext, PPOOptimizers, TrainingSession
from .setup_env import _build_env, _log_reset_diagnostics
from .setup_intrinsic import _build_intrinsic
from irl.utils.loggers import MetricLogger


def _resolve_deterministic_flag(cfg: Config) -> bool:
    deterministic = False
    try:
        exp_cfg = getattr(cfg, "exp", None)
        if exp_cfg is not None:
            deterministic = bool(getattr(exp_cfg, "deterministic", False))
    except Exception:
        deterministic = False
    return deterministic


def build_training_session(
    cfg: Config,
    *,
    device: torch.device,
    run_dir: Optional[Path],
    resume: bool,
    logger,
) -> TrainingSession:
    deterministic = _resolve_deterministic_flag(cfg)
    seed_everything(int(cfg.seed), deterministic=deterministic)

    try:
        ensure_mujoco_gl(cfg.env.id)
    except Exception:
        pass

    run_dir_resolved, ckpt = _init_run_dir_and_ckpt(cfg, run_dir)
    resume_payload, resume_step = _maybe_load_resume_payload(cfg, ckpt, resume)

    env = _build_env(cfg, logger=logger)
    obs_space, act_space = single_spaces(env)

    is_image = bool(is_image_space(obs_space))

    policy = PolicyNetwork(obs_space, act_space).to(device)
    value = ValueNetwork(obs_space).to(device)

    pol_opt = Adam(policy.parameters(), lr=float(cfg.ppo.learning_rate))
    val_opt = Adam(value.parameters(), lr=float(cfg.ppo.learning_rate))

    (
        intrinsic_module,
        use_intrinsic,
        intrinsic_norm_mode,
        intrinsic_outputs_normalized_flag,
        method_l,
        eta,
        int_rms,
    ) = _build_intrinsic(
        cfg,
        obs_space=obs_space,
        act_space=act_space,
        device=device,
        logger=logger,
    )

    ml = MetricLogger(run_dir_resolved, cfg.logging)
    ml.log_hparams(to_dict(cfg))

    obs = _log_reset_diagnostics(
        env=env,
        intrinsic_module=intrinsic_module,
        method_l=method_l,
        intrinsic_outputs_normalized_flag=intrinsic_outputs_normalized_flag,
        seed=int(cfg.seed),
    )

    B = int(getattr(env, "num_envs", 1))

    obs_norm = None if is_image else RunningObsNorm(shape=int(obs_space.shape[0]))

    global_step, update_idx = _restore_from_checkpoint(
        resume_payload=resume_payload,
        resume_step=resume_step,
        policy=policy,
        value=value,
        pol_opt=pol_opt,
        val_opt=val_opt,
        intrinsic_module=intrinsic_module,
        method_l=method_l,
        int_rms=int_rms,
        obs_norm=obs_norm,
        is_image=is_image,
        device=device,
        logger=logger,
    )

    return TrainingSession(
        run_dir=run_dir_resolved,
        device=device,
        env=env,
        obs_space=obs_space,
        act_space=act_space,
        is_image=is_image,
        num_envs=B,
        policy=policy,
        value=value,
        optimizers=PPOOptimizers(policy=pol_opt, value=val_opt),
        intrinsic=IntrinsicContext(
            module=intrinsic_module,
            method=method_l,
            eta=float(eta),
            use_intrinsic=bool(use_intrinsic),
            norm_mode=str(intrinsic_norm_mode),
            outputs_normalized=intrinsic_outputs_normalized_flag,
            rms=int_rms,
        ),
        obs_norm=obs_norm,
        ckpt=ckpt,
        metric_logger=ml,
        obs=obs,
        global_step=int(global_step),
        update_idx=int(update_idx),
    )
