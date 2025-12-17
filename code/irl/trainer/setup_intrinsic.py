from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from irl.cfg import Config
from irl.intrinsic import RunningRMS, create_intrinsic_module, is_intrinsic_method
from irl.intrinsic.config import build_intrinsic_kwargs


def _build_intrinsic(
    cfg: Config,
    *,
    obs_space: Any,
    act_space: Any,
    device: torch.device,
    logger,
) -> Tuple[Optional[Any], bool, str, Optional[bool], str, float, RunningRMS]:
    method_l = str(cfg.method).lower()
    eta = float(cfg.intrinsic.eta)
    use_intrinsic = is_intrinsic_method(method_l) and eta > 0.0
    intrinsic_module = None
    intrinsic_norm_mode = "none"
    intrinsic_outputs_normalized_flag: Optional[bool] = None

    checkpoint_include_points = True
    try:
        checkpoint_include_points = bool(getattr(cfg.intrinsic, "checkpoint_include_points", True))
    except Exception:
        checkpoint_include_points = True

    if is_intrinsic_method(method_l):
        fail_on_intrinsic_error = bool(getattr(cfg.intrinsic, "fail_on_error", True))
        try:
            intrinsic_kwargs = build_intrinsic_kwargs(cfg)
            intrinsic_kwargs["checkpoint_include_points"] = bool(checkpoint_include_points)

            intrinsic_module = create_intrinsic_module(
                method_l,
                obs_space,
                act_space,
                device=device,
                **intrinsic_kwargs,
            )
            if intrinsic_module is not None and hasattr(intrinsic_module, "checkpoint_include_points"):
                try:
                    setattr(
                        intrinsic_module,
                        "checkpoint_include_points",
                        bool(checkpoint_include_points),
                    )
                except Exception:
                    pass

            if not use_intrinsic:
                logger.warning(
                    "Method %r selected but intrinsic.eta=%.3g; intrinsic rewards disabled (eta=0).",
                    method_l,
                    eta,
                )
        except Exception as exc:
            logger.error("Failed to create intrinsic module %r (%s).", method_l, exc)
            if fail_on_intrinsic_error:
                raise RuntimeError(
                    "Intrinsic module construction failed for method "
                    f"{method_l!r}. Set intrinsic.fail_on_error=False to continue without "
                    "intrinsic rewards."
                ) from exc
            logger.error("intrinsic.fail_on_error is False; continuing without intrinsic rewards.")
            intrinsic_module = None
            use_intrinsic = False

    if intrinsic_module is not None:
        try:
            intrinsic_outputs_normalized_flag = bool(
                getattr(intrinsic_module, "outputs_normalized", False)
            )
        except Exception:
            intrinsic_outputs_normalized_flag = None

        if intrinsic_outputs_normalized_flag is True:
            intrinsic_norm_mode = "module_rms"
        elif intrinsic_outputs_normalized_flag is False:
            intrinsic_norm_mode = "trainer_rms"
        else:
            intrinsic_norm_mode = "unknown"
    else:
        intrinsic_norm_mode = "none"
        intrinsic_outputs_normalized_flag = None

    int_rms = RunningRMS(beta=0.99, eps=1e-8)
    return (
        intrinsic_module,
        use_intrinsic,
        intrinsic_norm_mode,
        intrinsic_outputs_normalized_flag,
        method_l,
        eta,
        int_rms,
    )
