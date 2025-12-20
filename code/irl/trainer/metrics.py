from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from .metrics_schema import (
    COL_EPISODE_RETURN_MEAN,
    COL_GATE_RATE,
    COL_GATE_RATE_PCT,
    COL_IMPACT_RMS,
    COL_LP_RMS,
    COL_REWARD_MEAN,
    COL_REWARD_TOTAL_MEAN,
)
from .rollout import RolloutBatch


def build_log_payload(
    *,
    policy: Any,
    rollout: RolloutBatch,
    rewards_total_seq: np.ndarray,
    obs_flat_for_ppo: np.ndarray,
    ppo_stats: Mapping[str, float] | None,
    intrinsic_module: Any | None,
    method_l: str,
    intrinsic_norm_mode: str,
    intrinsic_outputs_normalized_flag: bool | None,
    int_rms: Any,
    r_int_raw_flat: np.ndarray | None,
    r_int_scaled_flat: np.ndarray | None,
    mod_metrics: Mapping[str, float] | None,
    device: torch.device,
    t_intrinsic_compute: float,
    t_intrinsic_update: float,
    t_gae: float,
    t_ppo: float,
) -> dict[str, object]:
    T, B = int(rollout.T), int(rollout.B)

    ep_returns = rollout.episode_returns
    ep_lengths = rollout.episode_lengths
    ep_successes = rollout.episode_successes

    ep_count = int(len(ep_returns))
    ep_return_mean = float(np.mean(ep_returns)) if ep_returns else float("nan")
    ep_return_std = float(np.std(ep_returns, ddof=0)) if ep_returns else float("nan")
    ep_length_mean = float(np.mean(ep_lengths)) if ep_lengths else float("nan")
    ep_length_std = float(np.std(ep_lengths, ddof=0)) if ep_lengths else float("nan")
    ep_success_rate = float(np.mean(ep_successes)) if ep_successes else float("nan")

    with torch.no_grad():
        last_obs = rollout.obs_seq[-1]
        ent_last = float("nan")
        try:
            if len(rollout.obs_shape) >= 2:
                ent_last = float(policy.entropy(last_obs).mean().item())
            else:
                last_obs_t = torch.as_tensor(last_obs, device=device, dtype=torch.float32)
                ent_last = float(policy.entropy(last_obs_t).mean().item())
            if not np.isfinite(float(ent_last)):
                ent_last = float("nan")
        except Exception:
            ent_last = float("nan")

        ENTROPY_MAX_SAMPLES = 1024
        total_samples = int(obs_flat_for_ppo.shape[0])

        ent_mean_update = float("nan")
        if total_samples > 0:
            try:
                if total_samples > ENTROPY_MAX_SAMPLES:
                    idx = np.linspace(
                        0, total_samples - 1, ENTROPY_MAX_SAMPLES, dtype=np.int64
                    )
                    obs_sub = obs_flat_for_ppo[idx]
                else:
                    obs_sub = obs_flat_for_ppo

                if len(rollout.obs_shape) >= 2:
                    ent_mean_update = float(policy.entropy(obs_sub).mean().item())
                else:
                    obs_sub_t = torch.as_tensor(obs_sub, device=device, dtype=torch.float32)
                    ent_mean_update = float(policy.entropy(obs_sub_t).mean().item())

                if not np.isfinite(float(ent_mean_update)):
                    ent_mean_update = float("nan")
            except Exception:
                ent_mean_update = float("nan")

    log_payload: dict[str, object] = {
        "entropy_last": float(ent_last),
        "entropy_update_mean": float(ent_mean_update),
        COL_REWARD_MEAN: float(rollout.rewards_ext_seq.mean()),
        COL_REWARD_TOTAL_MEAN: float(rewards_total_seq.mean()),
        "episode_count": int(ep_count),
        COL_EPISODE_RETURN_MEAN: float(ep_return_mean),
        "episode_return_std": float(ep_return_std),
        "episode_length_mean": float(ep_length_mean),
        "episode_length_std": float(ep_length_std),
        "success_rate": float(ep_success_rate),
        "intrinsic_norm_mode": str(intrinsic_norm_mode),
    }
    if intrinsic_outputs_normalized_flag is not None:
        log_payload["intrinsic_outputs_normalized"] = bool(intrinsic_outputs_normalized_flag)

    if isinstance(ppo_stats, Mapping):
        try:
            clip_frac = float(ppo_stats.get("clip_frac", float("nan")))
            log_payload.update(
                {
                    "approx_kl": float(ppo_stats.get("approx_kl", float("nan"))),
                    "clip_frac": clip_frac,
                    "clip_frac_pct": (100.0 * clip_frac) if np.isfinite(clip_frac) else float("nan"),
                    "entropy_minibatch_mean": float(ppo_stats.get("entropy", float("nan"))),
                    "ppo_policy_loss": float(ppo_stats.get("policy_loss", float("nan"))),
                    "ppo_value_loss": float(ppo_stats.get("value_loss", float("nan"))),
                    "ppo_early_stop": float(ppo_stats.get("early_stop", 0.0)),
                    "ppo_epochs_ran": float(ppo_stats.get("epochs_ran", float("nan"))),
                }
            )
        except Exception:
            pass

    if r_int_raw_flat is not None and r_int_scaled_flat is not None:
        outputs_norm = (
            bool(getattr(intrinsic_module, "outputs_normalized", False))
            if intrinsic_module is not None
            else False
        )

        r_int_rms_val = float(int_rms.rms)
        if outputs_norm and intrinsic_module is not None:
            if hasattr(intrinsic_module, "rms"):
                try:
                    r_int_rms_val = float(getattr(intrinsic_module, "rms"))
                except Exception:
                    pass
            if hasattr(intrinsic_module, "impact_rms") and hasattr(intrinsic_module, "lp_rms"):
                try:
                    imp_rms_val = float(getattr(intrinsic_module, "impact_rms"))
                    lp_rms_val = float(getattr(intrinsic_module, "lp_rms"))
                    log_payload[COL_IMPACT_RMS] = imp_rms_val
                    log_payload[COL_LP_RMS] = lp_rms_val
                    r_int_rms_val = 0.5 * (imp_rms_val + lp_rms_val)
                except Exception:
                    pass
            elif hasattr(intrinsic_module, "lp_rms"):
                try:
                    r_int_rms_val = float(getattr(intrinsic_module, "lp_rms"))
                except Exception:
                    pass

        log_payload.update(
            {
                "r_int_raw_mean": float(np.mean(r_int_raw_flat)),
                "r_int_mean": float(np.mean(r_int_scaled_flat)),
                "r_int_rms": float(r_int_rms_val),
            }
        )
        for k, v in (mod_metrics or {}).items():
            try:
                log_payload[f"{method_l}_{k}"] = float(v)
            except Exception:
                pass

    if (
        intrinsic_module is not None
        and str(method_l).startswith("glpe")
        and hasattr(intrinsic_module, "gate_rate")
    ):
        try:
            gr = float(getattr(intrinsic_module, "gate_rate"))
            log_payload[COL_GATE_RATE] = gr
            log_payload[COL_GATE_RATE_PCT] = 100.0 * gr
        except Exception:
            pass

    log_payload.update(
        {
            "time_rollout_s": float(rollout.time_rollout_s),
            "time_rollout_policy_s": float(rollout.time_rollout_policy_s),
            "time_rollout_env_step_s": float(rollout.time_rollout_env_step_s),
            "time_rollout_intrinsic_step_s": float(rollout.time_rollout_intrinsic_step_s),
            "time_rollout_other_s": float(rollout.time_rollout_other_s),
            "time_intrinsic_compute_s": float(t_intrinsic_compute),
            "time_intrinsic_update_s": float(t_intrinsic_update),
            "time_gae_s": float(t_gae),
            "time_ppo_s": float(t_ppo),
        }
    )

    return log_payload
