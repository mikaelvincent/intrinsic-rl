from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import irl.utils.checkpoint_schema as ckpt_schema
from irl.cfg import Config, to_dict
from irl.intrinsic import RunningRMS
from irl.models import PolicyNetwork, ValueNetwork
from irl.utils.checkpoint import CheckpointManager, compute_cfg_hash
from irl.utils.loggers import (
    log_resume_intrinsic_warning,
    log_resume_loaded,
    log_resume_no_checkpoint,
    log_resume_optimizer_warning,
    log_resume_state_restored,
)

from .build import default_run_dir
from .obs_norm import RunningObsNorm
from .runtime_utils import _move_optimizer_state_to_device


def _init_run_dir_and_ckpt(cfg: Config, run_dir: Optional[Path]) -> Tuple[Path, CheckpointManager]:
    resolved = Path(run_dir) if run_dir is not None else default_run_dir(cfg)
    ckpt = CheckpointManager(
        resolved,
        interval_steps=cfg.logging.checkpoint_interval,
        max_to_keep=getattr(cfg.logging, "checkpoint_max_to_keep", None),
    )
    return resolved, ckpt


def _maybe_load_resume_payload(
    cfg: Config, ckpt: CheckpointManager, resume: bool
) -> Tuple[Optional[dict], int]:
    resume_payload: Optional[dict] = None
    resume_step: int = 0

    if resume:
        try:
            payload_cpu, step_cpu = ckpt.load_latest(map_location="cpu")
            current_hash = compute_cfg_hash(to_dict(cfg))
            stored_hash = payload_cpu.get(ckpt_schema.KEY_CFG_HASH)
            if stored_hash is None:
                stored_hash = compute_cfg_hash(payload_cpu.get(ckpt_schema.KEY_CFG, {}) or {})
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
            raise

    return resume_payload, resume_step


def _restore_from_checkpoint(
    *,
    resume_payload: Optional[dict],
    resume_step: int,
    policy: PolicyNetwork,
    value: ValueNetwork,
    pol_opt,
    val_opt,
    intrinsic_module: Optional[object],
    method_l: str,
    int_rms: RunningRMS,
    obs_norm: Optional[RunningObsNorm],
    is_image: bool,
    device,
    logger,
) -> Tuple[int, int]:
    global_step = 0
    update_idx = 0

    if resume_payload is None:
        return global_step, update_idx

    try:
        policy.load_state_dict(resume_payload[ckpt_schema.KEY_POLICY])
        value.load_state_dict(resume_payload[ckpt_schema.KEY_VALUE])
    except Exception:
        logger.warning("Could not load policy/value weights from checkpoint; using fresh init.")

    try:
        int_rms.load_state_dict(resume_payload.get(ckpt_schema.KEY_INTRINSIC_NORM, {}))
    except Exception:
        pass

    try:
        if (
            not is_image
            and resume_payload.get(ckpt_schema.KEY_OBS_NORM) is not None
            and obs_norm is not None
        ):
            on = resume_payload[ckpt_schema.KEY_OBS_NORM]
            import numpy as _np

            obs_norm.count = float(on.get("count", obs_norm.count))
            obs_norm.mean = _np.asarray(on.get("mean", obs_norm.mean), dtype=_np.float64)
            obs_norm.var = _np.asarray(on.get("var", obs_norm.var), dtype=_np.float64)
    except Exception:
        pass

    try:
        intr = resume_payload.get(ckpt_schema.KEY_INTRINSIC)
        if (
            intrinsic_module is not None
            and isinstance(intr, dict)
            and intr.get(ckpt_schema.INTRINSIC_METHOD) == method_l
        ):
            sd = intr.get(ckpt_schema.INTRINSIC_STATE_DICT, None)
            extra_state = intr.get(ckpt_schema.INTRINSIC_EXTRA_STATE, None)

            if isinstance(sd, dict):
                res = intrinsic_module.load_state_dict(sd, strict=False)
                missing = [k for k in res.missing_keys if k != "_extra_state"]
                unexpected = [k for k in res.unexpected_keys if k != "_extra_state"]
                if missing or unexpected:
                    logger.warning(
                        "Intrinsic state mismatch on resume for method=%s (missing=%s, unexpected=%s).",
                        method_l,
                        missing,
                        unexpected,
                    )

            if extra_state is None:
                extra_state = resume_payload.get(ckpt_schema.KEY_INTRINSIC_EXTRA_STATE_COMPAT, None)
                if extra_state is None:
                    extra_state = resume_payload.get(ckpt_schema.KEY_INTRINSIC_STATE_COMPAT, None)

            if (
                extra_state is not None
                and hasattr(intrinsic_module, "set_extra_state")
                and callable(getattr(intrinsic_module, "set_extra_state"))
                and not (isinstance(sd, dict) and "_extra_state" in sd)
            ):
                intrinsic_module.set_extra_state(extra_state)

            intrinsic_module.to(device)
    except Exception:
        log_resume_intrinsic_warning(method_l)

    try:
        opt_payload = resume_payload.get(ckpt_schema.KEY_OPTIMIZERS, {})
        pol_state = opt_payload.get(ckpt_schema.OPT_POLICY, None)
        val_state = opt_payload.get(ckpt_schema.OPT_VALUE, None)
        if pol_state is not None:
            pol_opt.load_state_dict(pol_state)
            _move_optimizer_state_to_device(pol_opt, next(policy.parameters()).device)
        if val_state is not None:
            val_opt.load_state_dict(val_state)
            _move_optimizer_state_to_device(val_opt, next(value.parameters()).device)
    except Exception:
        log_resume_optimizer_warning()

    global_step = int(resume_step)
    try:
        update_idx = int(
            (resume_payload.get(ckpt_schema.KEY_META) or {}).get(ckpt_schema.META_UPDATES, update_idx)
        )
    except Exception:
        update_idx = update_idx

    log_resume_state_restored(global_step)
    return global_step, update_idx
