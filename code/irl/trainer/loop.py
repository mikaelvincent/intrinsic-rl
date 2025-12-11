"""End-to-end PPO training loop with intrinsic rewards.

This module wires together environments, PPO models, intrinsic reward
modules, logging, and checkpointing into a single training entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from irl.cfg import Config, validate_config
from irl.utils.loggers import get_logger

from .build import ensure_device
from .training_engine import run_training_loop
from .training_setup import build_training_session


# Ensure we have a module-level logger for this trainer.
_LOG = get_logger(__name__)
# How often (in PPO updates) to emit a console progress line.
_LOG_TRAIN_EVERY_UPDATES = 10


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
    resume:
        If True and a latest checkpoint exists in `run_dir`, restore state and continue.
        A config-hash mismatch aborts with a clear error to avoid accidental cross-run resumes.

    Notes
    -----
    The trainer collects nominal rollouts of length ``T = cfg.ppo.steps_per_update``
    per environment, but the final PPO update in a run may use a shorter rollout
    if fewer than ``T * cfg.env.vec_envs`` steps remain before reaching
    ``total_steps``. That partial batch is still split into minibatches and
    passed to :func:`irl.algo.ppo.ppo_update`, which is designed to handle
    non-divisible batch sizes robustly.
    """
    validate_config(cfg)
    device = ensure_device(cfg.device)

    session = build_training_session(
        cfg,
        device=device,
        run_dir=run_dir,
        resume=resume,
        logger=_LOG,
    )

    try:
        run_training_loop(
            cfg,
            session,
            total_steps=int(total_steps),
            logger=_LOG,
            log_every_updates=int(_LOG_TRAIN_EVERY_UPDATES),
        )
    finally:
        session.metric_logger.close()
        session.env.close()

    return session.run_dir
