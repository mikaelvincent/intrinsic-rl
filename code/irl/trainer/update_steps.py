from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from irl.algo.advantage import compute_gae
from irl.algo.ppo import ppo_update
from irl.intrinsic import compute_intrinsic_batch, update_module
from irl.methods.spec import canonical_method as _canonical_method

from .rollout import RolloutBatch


def _canonical_intrinsic_method(method_l: str) -> str:
    return _canonical_method(method_l)


def _expected_intrinsic_update_keys(method_l: str) -> tuple[str, ...]:
    m = _canonical_intrinsic_method(method_l)
    if m in {"icm", "ride", "riac", "glpe"}:
        return ("loss_total", "loss_forward", "loss_inverse", "intrinsic_mean")
    if m == "rnd":
        return ("loss_total", "loss_intrinsic_mean", "rms")
    return ()


def _coerce_metrics_to_floats(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in metrics.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            out[str(k)] = float("nan")
    return out


@dataclass(frozen=True)
class IntrinsicRewards:
    rewards_total_seq: np.ndarray
    r_int_raw_flat: np.ndarray | None
    r_int_scaled_flat: np.ndarray | None
    module_metrics: dict[str, float]
    time_compute_s: float
    time_update_s: float


def compute_intrinsic_rewards(
    *,
    rollout: RolloutBatch,
    intrinsic_module: Any | None,
    use_intrinsic: bool,
    method_l: str,
    eta: float,
    r_clip: float,
    int_rms: Any,
    device: torch.device,
    profile_cuda_sync: bool,
    maybe_cuda_sync: Callable[[torch.device, bool], None],
) -> IntrinsicRewards:
    T, B = int(rollout.T), int(rollout.B)

    r_int_raw_flat = None
    r_int_scaled_flat = None
    mod_metrics: dict[str, float] = {}

    t_intrinsic_compute = 0.0
    t_intrinsic_update = 0.0

    if intrinsic_module is not None and bool(use_intrinsic):
        intrinsic_compute_t0 = time.perf_counter()

        if str(method_l) == "ride":
            r_seq = rollout.r_int_raw_seq
            r_int_raw_flat = r_seq.reshape(T * B).astype(np.float32)
            obs_flat = rollout.obs_seq.reshape((T * B,) + tuple(rollout.obs_shape))
            next_obs_flat = rollout.next_obs_seq.reshape((T * B,) + tuple(rollout.obs_shape))
            acts_flat = (
                rollout.actions_seq.reshape(T * B)
                if bool(rollout.is_discrete)
                else rollout.actions_seq.reshape(T * B, -1)
            )
        else:
            maybe_cuda_sync(device, bool(profile_cuda_sync))

            obs_flat = rollout.obs_seq.reshape((T * B,) + tuple(rollout.obs_shape))
            next_obs_flat = rollout.next_obs_seq.reshape((T * B,) + tuple(rollout.obs_shape))
            acts_flat = (
                rollout.actions_seq.reshape(T * B)
                if bool(rollout.is_discrete)
                else rollout.actions_seq.reshape(T * B, -1)
            )

            r_int_raw_t = compute_intrinsic_batch(
                intrinsic_module,
                str(method_l),
                obs_flat,
                next_obs_flat,
                acts_flat,
            )
            r_int_raw_flat = r_int_raw_t.detach().cpu().numpy().astype(np.float32)

        term_mask_flat = rollout.terminals_seq.reshape(T * B) > 0.0
        r_int_raw_flat = np.asarray(r_int_raw_flat, dtype=np.float32)
        r_int_raw_flat[term_mask_flat] = 0.0

        outputs_norm = bool(getattr(intrinsic_module, "outputs_normalized", False))
        if outputs_norm:
            r_int_scaled_flat = float(eta) * np.clip(r_int_raw_flat, -float(r_clip), float(r_clip))
        else:
            int_rms.update(r_int_raw_flat)
            r_int_norm_flat = int_rms.normalize(r_int_raw_flat)
            r_int_scaled_flat = float(eta) * np.clip(
                r_int_norm_flat, -float(r_clip), float(r_clip)
            )

        maybe_cuda_sync(device, bool(profile_cuda_sync))
        t_intrinsic_compute = time.perf_counter() - intrinsic_compute_t0

        intrinsic_update_t0 = time.perf_counter()
        try:
            maybe_cuda_sync(device, bool(profile_cuda_sync))
            raw_metrics = dict(
                update_module(
                    intrinsic_module,
                    str(method_l),
                    obs_flat,
                    next_obs_flat,
                    acts_flat,
                )
            )
            maybe_cuda_sync(device, bool(profile_cuda_sync))
            mod_metrics = _coerce_metrics_to_floats(raw_metrics)
        except Exception:
            mod_metrics = {}

        expected = _expected_intrinsic_update_keys(str(method_l))
        if expected:
            for k in expected:
                if k not in mod_metrics:
                    mod_metrics[k] = float("nan")

        t_intrinsic_update = time.perf_counter() - intrinsic_update_t0

    if r_int_scaled_flat is not None:
        rewards_total_seq = rollout.rewards_ext_seq + r_int_scaled_flat.reshape(T, B)
    else:
        rewards_total_seq = rollout.rewards_ext_seq

    return IntrinsicRewards(
        rewards_total_seq=rewards_total_seq,
        r_int_raw_flat=r_int_raw_flat,
        r_int_scaled_flat=r_int_scaled_flat,
        module_metrics=mod_metrics,
        time_compute_s=float(t_intrinsic_compute),
        time_update_s=float(t_intrinsic_update),
    )


@dataclass(frozen=True)
class AdvantageBatch:
    advantages: torch.Tensor
    value_targets: torch.Tensor
    time_s: float


def compute_advantages(
    *,
    rollout: RolloutBatch,
    rewards_total_seq: np.ndarray,
    value_fn: Any,
    gamma: float,
    lam: float,
    device: torch.device,
    profile_cuda_sync: bool,
    maybe_cuda_sync: Callable[[torch.device, bool], None],
) -> AdvantageBatch:
    T, B = int(rollout.T), int(rollout.B)

    terminals_gae: np.ndarray | object = rollout.terminals_seq
    truncations_gae: np.ndarray | object = rollout.truncations_seq

    no_final = None
    try:
        no_final = getattr(rollout, "timeouts_no_final_obs_seq", None)
    except Exception:
        no_final = None

    if no_final is not None:
        try:
            nf = np.asarray(no_final, dtype=np.float32)
            if nf.shape[:2] == (T, B):
                mask = nf > 0.0
                if bool(mask.any()):
                    m = mask.astype(np.float32, copy=False)
                    terminals_gae = np.maximum(
                        np.array(rollout.terminals_seq, dtype=np.float32, copy=True),
                        m,
                    )
                    truncations_gae = np.array(rollout.truncations_seq, dtype=np.float32, copy=True)
                    truncations_gae = truncations_gae * (1.0 - m)
        except Exception:
            terminals_gae = rollout.terminals_seq
            truncations_gae = rollout.truncations_seq

    gae_batch = {
        "obs": rollout.obs_seq,
        "next_observations": rollout.next_obs_seq,
        "rewards": rewards_total_seq,
        "terminals": terminals_gae,
        "truncations": truncations_gae,
    }

    gae_t0 = time.perf_counter()
    maybe_cuda_sync(device, bool(profile_cuda_sync))
    adv, v_targets = compute_gae(
        gae_batch,
        value_fn,
        gamma=float(gamma),
        lam=float(lam),
        bootstrap_on_timeouts=True,
    )
    maybe_cuda_sync(device, bool(profile_cuda_sync))
    t_gae = time.perf_counter() - gae_t0

    return AdvantageBatch(advantages=adv, value_targets=v_targets, time_s=float(t_gae))


@dataclass(frozen=True)
class PPOUpdateOutput:
    stats: dict[str, float] | None
    time_s: float


def ppo_step(
    *,
    policy: Any,
    value: Any,
    batch: Any,
    advantages: Any,
    value_targets: Any,
    cfg_ppo: Any,
    optimizers: tuple[Any, Any],
    device: torch.device,
    profile_cuda_sync: bool,
    maybe_cuda_sync: Callable[[torch.device, bool], None],
) -> PPOUpdateOutput:
    ppo_t0 = time.perf_counter()
    maybe_cuda_sync(device, bool(profile_cuda_sync))
    stats = ppo_update(
        policy,
        value,
        batch,
        advantages,
        value_targets,
        cfg_ppo,
        optimizers=optimizers,
        return_stats=True,
    )
    maybe_cuda_sync(device, bool(profile_cuda_sync))
    t_ppo = time.perf_counter() - ppo_t0

    return PPOUpdateOutput(stats=stats, time_s=float(t_ppo))
