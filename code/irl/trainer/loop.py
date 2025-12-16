from __future__ import annotations

from pathlib import Path
from typing import Optional

from irl.cfg import Config, validate_config
from irl.utils.loggers import get_logger

from .build import ensure_device
from .training_engine import run_training_loop
from .training_setup import build_training_session

_LOG = get_logger(__name__)
_LOG_TRAIN_EVERY_UPDATES = 10


def train(
    cfg: Config,
    *,
    total_steps: Optional[int] = None,
    run_dir: Optional[Path] = None,
    resume: bool = False,
) -> Path:
    validate_config(cfg)
    device = ensure_device(cfg.device)

    cfg_steps = None
    try:
        cfg_steps = getattr(getattr(cfg, "exp", None), "total_steps", None)
    except Exception:
        cfg_steps = None

    if total_steps is None:
        if cfg_steps is not None:
            effective_total_steps = int(cfg_steps)
            _LOG.info(
                "Training total_steps=%d (from cfg.exp.total_steps).",
                effective_total_steps,
            )
        else:
            effective_total_steps = 10_000
            _LOG.info(
                "Training total_steps=%d (default; cfg.exp.total_steps is unset).",
                effective_total_steps,
            )
    else:
        effective_total_steps = int(total_steps)
        if cfg_steps is not None and int(cfg_steps) != effective_total_steps:
            _LOG.info(
                "Training total_steps=%d (override; cfg.exp.total_steps=%d).",
                effective_total_steps,
                int(cfg_steps),
            )
        else:
            _LOG.info("Training total_steps=%d.", effective_total_steps)

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
            total_steps=int(effective_total_steps),
            logger=_LOG,
            log_every_updates=int(_LOG_TRAIN_EVERY_UPDATES),
        )
    finally:
        session.metric_logger.close()
        session.env.close()

    return session.run_dir
