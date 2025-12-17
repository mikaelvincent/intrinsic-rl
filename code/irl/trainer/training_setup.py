from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.optim import Adam

from irl.cfg import Config, to_dict
from irl.models import PolicyNetwork, ValueNetwork
from irl.utils.determinism import seed_everything
from irl.utils.spaces import is_image_space
from irl.utils.loggers import MetricLogger

from .build import default_run_dir, ensure_mujoco_gl, single_spaces
from .obs_norm import RunningObsNorm
from .resume import _init_run_dir_and_ckpt, _maybe_load_resume_payload, _restore_from_checkpoint
from .session_types import IntrinsicContext, PPOOptimizers, TrainingSession
from .setup_env import _build_env, _log_reset_diagnostics
from .setup_intrinsic import _build_intrinsic


def _existing_run_artifacts(run_dir: Path) -> list[Path]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return []
    if not run_dir.is_dir():
        return [run_dir]

    artifacts: list[Path] = []

    ckpt_dir = run_dir / "checkpoints"
    latest = ckpt_dir / "ckpt_latest.pt"
    if latest.exists():
        artifacts.append(latest)
    if ckpt_dir.exists():
        try:
            artifacts.extend(sorted(p for p in ckpt_dir.glob("ckpt_step_*.pt") if p.is_file()))
        except Exception:
            pass

    scalars = run_dir / "logs" / "scalars.csv"
    if scalars.exists():
        try:
            if scalars.is_file() and scalars.stat().st_size > 0:
                artifacts.append(scalars)
        except Exception:
            artifacts.append(scalars)

    return artifacts


def _format_artifacts(run_dir: Path, artifacts: list[Path]) -> str:
    bits: list[str] = []
    for p in artifacts:
        try:
            bits.append(str(p.resolve().relative_to(run_dir.resolve())))
        except Exception:
            bits.append(str(p))
    return ", ".join(bits)


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
    run_dir_candidate = Path(run_dir) if run_dir is not None else default_run_dir(cfg)

    # A fresh run into an existing directory can corrupt checkpoint cadence and mix logs.
    if not bool(resume):
        artifacts = _existing_run_artifacts(run_dir_candidate)
        if artifacts:
            found = _format_artifacts(run_dir_candidate, artifacts)
            raise RuntimeError(
                "Refusing to start a fresh run in a directory that already contains run artifacts: "
                f"{run_dir_candidate} (found: {found}). Delete the directory or run with resume=True."
            )

    deterministic = _resolve_deterministic_flag(cfg)
    seed_everything(int(cfg.seed), deterministic=deterministic)

    try:
        ensure_mujoco_gl(cfg.env.id)
    except Exception:
        pass

    run_dir_resolved, ckpt = _init_run_dir_and_ckpt(cfg, run_dir_candidate)
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
