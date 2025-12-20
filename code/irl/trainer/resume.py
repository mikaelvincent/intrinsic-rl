from __future__ import annotations

import os
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

_ALLOW_RESUME_WO_POINTS_ENV = "IRL_ALLOW_RESUME_WITHOUT_KDTREE_POINTS"


def _truthy_env(name: str) -> bool:
    v = os.environ.get(name)
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_kdtree_intrinsic_method(method_l: str) -> bool:
    m = str(method_l).strip().lower()
    return m == "riac" or m.startswith("glpe")


def _extract_store_include_points(extra: object) -> bool | None:
    if not isinstance(extra, dict):
        return None
    store = extra.get("store")
    if not isinstance(store, dict):
        return None
    if "include_points" not in store:
        return None
    try:
        return bool(store.get("include_points"))
    except Exception:
        return None


def _payload_extra_state(payload: dict, intr: dict) -> object | None:
    extra_state = intr.get(ckpt_schema.INTRINSIC_EXTRA_STATE, None)
    if extra_state is not None:
        return extra_state
    compat = payload.get(ckpt_schema.KEY_INTRINSIC_EXTRA_STATE_COMPAT, None)
    if compat is not None:
        return compat
    return payload.get(ckpt_schema.KEY_INTRINSIC_STATE_COMPAT, None)


def _intrinsic_optimizer_objects(intrinsic_module: object) -> dict[str, object]:
    out: dict[str, object] = {}

    opt_main = getattr(intrinsic_module, "_opt", None)
    if opt_main is not None and hasattr(opt_main, "load_state_dict"):
        out["main"] = opt_main

    icm = getattr(intrinsic_module, "icm", None)
    opt_icm = getattr(icm, "_opt", None) if icm is not None else None
    if opt_icm is not None and opt_icm is not opt_main and hasattr(opt_icm, "load_state_dict"):
        out["icm"] = opt_icm

    return out


def _restore_intrinsic_optimizers(
    *,
    intrinsic_module: object,
    opt_states: object,
    device: object,
    logger: object,
) -> None:
    if not isinstance(opt_states, dict):
        return

    opts = _intrinsic_optimizer_objects(intrinsic_module)
    if not opts:
        return

    try:
        param_dev = next(getattr(intrinsic_module, "parameters")()).device
    except Exception:
        param_dev = device

    for k, state in opt_states.items():
        name = str(k)
        opt = opts.get(name)
        if opt is None or not isinstance(state, dict):
            continue
        try:
            opt.load_state_dict(state)
            _move_optimizer_state_to_device(opt, param_dev)
        except Exception as exc:
            if hasattr(logger, "warning"):
                logger.warning(
                    "Could not restore intrinsic optimizer %s (%s).",
                    name,
                    type(exc).__name__,
                )


def _guard_resume_kdtree_points(
    payload: dict,
    *,
    intrinsic_module: object | None,
    method_l: str,
    logger: object,
) -> None:
    if intrinsic_module is None:
        return
    if not _is_kdtree_intrinsic_method(method_l):
        return

    intr = payload.get(ckpt_schema.KEY_INTRINSIC)
    if not isinstance(intr, dict) or intr.get(ckpt_schema.INTRINSIC_METHOD) != method_l:
        return

    sd = intr.get(ckpt_schema.INTRINSIC_STATE_DICT, None)
    extra = sd.get("_extra_state") if isinstance(sd, dict) and "_extra_state" in sd else None
    if extra is None:
        extra = _payload_extra_state(payload, intr)

    includes_points = _extract_store_include_points(extra)
    if includes_points is not False:
        return

    if _truthy_env(_ALLOW_RESUME_WO_POINTS_ENV):
        if hasattr(logger, "warning"):
            logger.warning(
                "Resuming with KDTree points omitted for method=%s; region splitting is disabled and "
                "training may diverge from an uninterrupted run.",
                str(method_l),
            )
        return

    raise RuntimeError(
        "Refusing to resume: intrinsic checkpoint for method "
        f"{str(method_l)!r} was saved without KDTree points (include_points=False), "
        "which changes training semantics on restore (region splitting is disabled). "
        "Resume with a checkpoint that includes points, or set "
        f"{_ALLOW_RESUME_WO_POINTS_ENV}=1 to resume anyway."
    )


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
                    "Supply a matching config or start a fresh run directory."
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

    _guard_resume_kdtree_points(
        resume_payload,
        intrinsic_module=intrinsic_module,
        method_l=str(method_l),
        logger=logger,
    )

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

            opt_states = intr.get(ckpt_schema.INTRINSIC_OPTIMIZERS, None)
            if opt_states is not None:
                _restore_intrinsic_optimizers(
                    intrinsic_module=intrinsic_module,
                    opt_states=opt_states,
                    device=device,
                    logger=logger,
                )
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
