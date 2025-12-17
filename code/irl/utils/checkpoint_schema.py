from __future__ import annotations

from typing import Any, Mapping

from irl.cfg import to_dict
from irl.utils.checkpoint import compute_cfg_hash

KEY_STEP = "step"
KEY_POLICY = "policy"
KEY_VALUE = "value"
KEY_CFG = "cfg"
KEY_CFG_HASH = "cfg_hash"
KEY_OBS_NORM = "obs_norm"
KEY_INTRINSIC_NORM = "intrinsic_norm"
KEY_META = "meta"
KEY_OPTIMIZERS = "optimizers"
KEY_INTRINSIC = "intrinsic"

META_UPDATES = "updates"

OPT_POLICY = "policy"
OPT_VALUE = "value"

OBS_NORM_COUNT = "count"
OBS_NORM_MEAN = "mean"
OBS_NORM_VAR = "var"

INTRINSIC_METHOD = "method"
INTRINSIC_STATE_DICT = "state_dict"
INTRINSIC_EXTRA_STATE = "extra_state"

KEY_INTRINSIC_EXTRA_STATE_COMPAT = "intrinsic_extra_state"
KEY_INTRINSIC_STATE_COMPAT = "intrinsic_state"


def _obs_norm_state(obs_norm: Any) -> dict[str, Any] | None:
    if obs_norm is None:
        return None
    return {
        OBS_NORM_COUNT: float(getattr(obs_norm, "count")),
        OBS_NORM_MEAN: getattr(obs_norm, "mean"),
        OBS_NORM_VAR: getattr(obs_norm, "var"),
    }


def build_checkpoint_payload(
    cfg: Any,
    *,
    global_step: int,
    update_idx: int,
    policy: Any,
    value: Any,
    is_image: bool,
    obs_norm: Any,
    int_rms: Any,
    pol_opt: Any,
    val_opt: Any,
    intrinsic_module: Any | None,
    method_l: str,
) -> dict[str, Any]:
    cfg_dict = to_dict(cfg)
    payload: dict[str, Any] = {
        KEY_STEP: int(global_step),
        KEY_POLICY: policy.state_dict(),
        KEY_VALUE: value.state_dict(),
        KEY_CFG: cfg_dict,
        KEY_CFG_HASH: compute_cfg_hash(cfg_dict),
        KEY_OBS_NORM: None if bool(is_image) else _obs_norm_state(obs_norm),
        KEY_INTRINSIC_NORM: int_rms.state_dict(),
        KEY_META: {META_UPDATES: int(update_idx)},
        KEY_OPTIMIZERS: {
            OPT_POLICY: pol_opt.state_dict(),
            OPT_VALUE: val_opt.state_dict(),
        },
    }

    if intrinsic_module is not None and hasattr(intrinsic_module, "state_dict"):
        try:
            payload[KEY_INTRINSIC] = {
                INTRINSIC_METHOD: str(method_l),
                INTRINSIC_STATE_DICT: intrinsic_module.state_dict(),
            }
        except Exception:
            pass

    return payload


def extract_cfg(payload: Mapping[str, Any]) -> dict[str, Any]:
    cfg = payload.get(KEY_CFG)
    return cfg if isinstance(cfg, dict) else {}


def extract_env_cfg(payload: Mapping[str, Any]) -> dict[str, Any]:
    cfg = extract_cfg(payload)
    env = cfg.get("env")
    return env if isinstance(env, dict) else {}


def extract_obs_norm(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    obs_norm = payload.get(KEY_OBS_NORM)
    return obs_norm if isinstance(obs_norm, dict) else None


def extract_intrinsic(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    intr = payload.get(KEY_INTRINSIC)
    return intr if isinstance(intr, dict) else None
