from __future__ import annotations

from pathlib import Path
from typing import Any

from irl.cfg import Config
from irl.utils.checkpoint_schema import build_checkpoint_payload
from irl.utils.run_meta import read_run_meta


def _load_run_meta(ckpt: Any) -> dict[str, object] | None:
    run_dir = getattr(ckpt, "run_dir", None)
    if run_dir is None:
        return None
    meta = read_run_meta(Path(run_dir) / "run_meta.json")
    return meta if meta is not None else None


def maybe_save_baseline_checkpoint(
    cfg: Config,
    *,
    ckpt: Any,
    policy: Any,
    value: Any,
    is_image: bool,
    obs_norm: Any,
    int_rms: Any,
    pol_opt: Any,
    val_opt: Any,
    intrinsic_module: Any | None,
    method_l: str,
    global_step: int,
    update_idx: int,
    logger: Any,
) -> None:
    if int(global_step) != 0 or int(update_idx) != 0:
        return
    if ckpt.latest_path.exists():
        return

    payload0 = build_checkpoint_payload(
        cfg,
        global_step=0,
        update_idx=0,
        policy=policy,
        value=value,
        is_image=bool(is_image),
        obs_norm=obs_norm,
        int_rms=int_rms,
        pol_opt=pol_opt,
        val_opt=val_opt,
        intrinsic_module=intrinsic_module,
        method_l=str(method_l),
        run_meta=_load_run_meta(ckpt),
    )
    ckpt_path0 = ckpt.save(step=0, payload=payload0)
    logger.info("Saved baseline checkpoint at step=0 to %s", ckpt_path0)


def maybe_save_periodic_checkpoint(
    cfg: Config,
    *,
    ckpt: Any,
    policy: Any,
    value: Any,
    is_image: bool,
    obs_norm: Any,
    int_rms: Any,
    pol_opt: Any,
    val_opt: Any,
    intrinsic_module: Any | None,
    method_l: str,
    global_step: int,
    update_idx: int,
    logger: Any,
) -> None:
    if not ckpt.should_save(int(global_step)):
        return

    payload = build_checkpoint_payload(
        cfg,
        global_step=int(global_step),
        update_idx=int(update_idx),
        policy=policy,
        value=value,
        is_image=bool(is_image),
        obs_norm=obs_norm,
        int_rms=int_rms,
        pol_opt=pol_opt,
        val_opt=val_opt,
        intrinsic_module=intrinsic_module,
        method_l=str(method_l),
        run_meta=_load_run_meta(ckpt),
    )
    ckpt_path = ckpt.save(step=int(global_step), payload=payload)
    logger.info("Saved checkpoint at step=%d to %s", int(global_step), ckpt_path)


def save_final_checkpoint(
    cfg: Config,
    *,
    ckpt: Any,
    policy: Any,
    value: Any,
    is_image: bool,
    obs_norm: Any,
    int_rms: Any,
    pol_opt: Any,
    val_opt: Any,
    intrinsic_module: Any | None,
    method_l: str,
    global_step: int,
    update_idx: int,
    logger: Any,
) -> None:
    payload = build_checkpoint_payload(
        cfg,
        global_step=int(global_step),
        update_idx=int(update_idx),
        policy=policy,
        value=value,
        is_image=bool(is_image),
        obs_norm=obs_norm,
        int_rms=int_rms,
        pol_opt=pol_opt,
        val_opt=val_opt,
        intrinsic_module=intrinsic_module,
        method_l=str(method_l),
        run_meta=_load_run_meta(ckpt),
    )
    final_ckpt_path = ckpt.save(step=int(global_step), payload=payload)
    logger.info("Saved final checkpoint at step=%d to %s", int(global_step), final_ckpt_path)
