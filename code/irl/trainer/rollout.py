from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from .runtime_utils import _apply_final_observation, _ensure_time_major_np


def _try_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _try_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _coerce_vec(x: Any, B: int, *, dtype: Any | None = None) -> np.ndarray | None:
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)
    except Exception:
        return None
    arr = arr.reshape(-1)
    if int(arr.size) == int(B):
        return arr
    if int(arr.size) == 1 and int(B) == 1:
        return arr
    return None


def _episode_stats_from_info(info: Mapping[str, Any]) -> tuple[float, int, float | None] | None:
    ep = info.get("episode")
    if not isinstance(ep, Mapping):
        return None

    r = _try_float(ep.get("r"))
    l = _try_int(ep.get("l"))
    if r is None or l is None:
        return None

    success = None
    if "is_success" in info:
        try:
            success = 1.0 if bool(info.get("is_success")) else 0.0
        except Exception:
            success = None
    elif "success" in info:
        try:
            success = 1.0 if bool(info.get("success")) else 0.0
        except Exception:
            success = None

    return float(r), int(l), success


def _extract_completed_episodes(
    infos: Any, done_flags: np.ndarray
) -> tuple[list[float], list[int], list[float]]:
    done = np.asarray(done_flags, dtype=bool).reshape(-1)
    B = int(done.shape[0])
    idxs = np.flatnonzero(done)
    if idxs.size == 0:
        return [], [], []

    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []

    def _append(info: Mapping[str, Any]) -> None:
        rec = _episode_stats_from_info(info)
        if rec is None:
            return
        r, l, s = rec
        returns.append(float(r))
        lengths.append(int(l))
        if s is not None:
            successes.append(float(s))

    if isinstance(infos, dict):
        final_info = infos.get("final_info") or infos.get("final_infos")
        if final_info is not None:
            if isinstance(final_info, np.ndarray):
                if final_info.shape[:1] == (B,):
                    for i in idxs:
                        fi = final_info[int(i)]
                        if isinstance(fi, dict):
                            _append(fi)
                elif B == 1:
                    try:
                        fi0 = final_info.reshape(-1)[0]
                        if isinstance(fi0, dict):
                            _append(fi0)
                    except Exception:
                        pass
            elif isinstance(final_info, (list, tuple)) and len(final_info) == B:
                for i in idxs:
                    fi = final_info[int(i)]
                    if isinstance(fi, dict):
                        _append(fi)
            elif B == 1 and isinstance(final_info, dict):
                _append(final_info)

            if returns:
                return returns, lengths, successes

        ep = infos.get("episode")
        if isinstance(ep, dict):
            r_arr = _coerce_vec(ep.get("r"), B, dtype=np.float64)
            l_arr = _coerce_vec(ep.get("l"), B, dtype=None)
            if r_arr is not None and l_arr is not None:
                used: list[int] = []
                for i in idxs:
                    ri = _try_float(r_arr[int(i)])
                    li = _try_int(l_arr[int(i)])
                    if ri is None or li is None:
                        continue
                    returns.append(float(ri))
                    lengths.append(int(li))
                    used.append(int(i))

                if used:
                    s_arr = None
                    for k in ("is_success", "success"):
                        if k in infos:
                            s_arr = _coerce_vec(infos.get(k), B, dtype=None)
                            break
                    if s_arr is not None:
                        for i in used:
                            try:
                                successes.append(1.0 if bool(s_arr[int(i)]) else 0.0)
                            except Exception:
                                pass

        if not returns and B == 1 and isinstance(infos.get("episode"), Mapping):
            _append(infos)

        return returns, lengths, successes

    if isinstance(infos, (list, tuple)) and len(infos) == B:
        for i in idxs:
            fi = infos[int(i)]
            if isinstance(fi, dict):
                _append(fi)

    return returns, lengths, successes


@dataclass(frozen=True)
class RolloutBatch:
    T: int
    B: int
    obs_seq: np.ndarray
    next_obs_seq: np.ndarray
    actions_seq: np.ndarray
    rewards_ext_seq: np.ndarray
    dones_seq: np.ndarray
    terminals_seq: np.ndarray
    truncations_seq: np.ndarray
    obs_shape: tuple[int, ...]
    is_discrete: bool
    r_int_raw_seq: np.ndarray | None

    episode_returns: list[float]
    episode_lengths: list[int]
    episode_successes: list[float]

    time_rollout_s: float
    time_rollout_policy_s: float
    time_rollout_env_step_s: float
    time_rollout_intrinsic_step_s: float
    time_rollout_other_s: float

    final_env_obs: Any
    steps_collected: int


def collect_rollout(
    *,
    env: Any,
    policy: Any,
    actor_policy: Any | None,
    obs: Any,
    obs_space: Any,
    act_space: Any,
    is_image: bool,
    obs_norm: Any | None,
    intrinsic_module: Any | None,
    use_intrinsic: bool,
    method_l: str,
    T: int,
    B: int,
    device: torch.device,
    logger,
) -> RolloutBatch:
    t_rollout_policy = 0.0
    t_rollout_env_step = 0.0
    t_rollout_intrinsic_step = 0.0

    obs_var = obs

    if not is_image:
        obs_dim = int(obs_space.shape[0])
        obs_seq = np.zeros((T, B, obs_dim), dtype=np.float32)
        next_obs_seq = np.zeros((T, B, obs_dim), dtype=np.float32)
        obs_shape = (int(obs_space.shape[0]),)
    else:
        obs_seq_list: list[np.ndarray] = []
        next_obs_seq_list: list[np.ndarray] = []
        obs_shape = tuple(int(s) for s in obs_space.shape)

    is_discrete = hasattr(act_space, "n")
    actions_seq = (
        np.zeros((T, B), dtype=np.int64)
        if is_discrete
        else np.zeros((T, B, int(act_space.shape[0])), dtype=np.float32)
    )
    rew_ext_seq = np.zeros((T, B), dtype=np.float32)
    dones_seq = np.zeros((T, B), dtype=np.float32)
    terminals_seq = np.zeros((T, B), dtype=np.float32)
    truncations_seq = np.zeros((T, B), dtype=np.float32)

    r_int_raw_seq = (
        np.zeros((T, B), dtype=np.float32)
        if (intrinsic_module is not None and str(method_l) == "ride" and bool(use_intrinsic))
        else None
    )

    ep_returns: list[float] = []
    ep_lengths: list[int] = []
    ep_successes: list[float] = []

    t_rollout_start = time.perf_counter()

    prev_done_flags: np.ndarray | None = None
    if r_int_raw_seq is not None and intrinsic_module is not None:
        prev = getattr(intrinsic_module, "_ride_prev_done_flags", None)
        if prev is not None:
            try:
                arr = np.asarray(prev, dtype=bool).reshape(-1)
                if int(arr.size) == int(B):
                    prev_done_flags = arr.astype(bool, copy=False)
            except Exception:
                prev_done_flags = None
        if prev_done_flags is None:
            prev_done_flags = np.zeros((B,), dtype=bool)

    for t in range(int(T)):
        obs_b = obs_var if B > 1 else obs_var[None, ...]
        if not is_image:
            obs_norm.update(obs_b)
            obs_b_norm = obs_norm.normalize(obs_b)
        else:
            obs_b_norm = obs_b

        if not is_image:
            obs_seq[t] = obs_b_norm.astype(np.float32)
        else:
            obs_seq_list.append(np.array(obs_b_norm, copy=True))

        pi_t0 = time.perf_counter()
        with torch.no_grad():
            if actor_policy is not None:
                a_tensor, _ = actor_policy.act(obs_b_norm)
                a_np = a_tensor.detach().numpy()
            else:
                if is_image:
                    a_tensor, _ = policy.act(obs_b_norm)
                else:
                    obs_tensor = torch.as_tensor(obs_b_norm, device=device, dtype=torch.float32)
                    a_tensor, _ = policy.act(obs_tensor)
                a_np = a_tensor.detach().cpu().numpy()
        t_rollout_policy += time.perf_counter() - pi_t0

        if is_discrete:
            a_np = a_np.astype(np.int64).reshape(B)
        else:
            a_np = a_np.reshape(B, -1).astype(np.float32)

        env_t0 = time.perf_counter()
        next_obs_env, rewards, terms, truncs, infos = env.step(a_np if B > 1 else a_np[0])
        t_rollout_env_step += time.perf_counter() - env_t0

        terms_b = np.asarray(terms, dtype=bool).reshape(B)
        truncs_b = np.asarray(truncs, dtype=bool).reshape(B)
        done_flags = terms_b | truncs_b

        er, el, es = _extract_completed_episodes(infos, done_flags)
        ep_returns.extend(er)
        ep_lengths.extend(el)
        ep_successes.extend(es)

        next_obs_rollout = _apply_final_observation(next_obs_env, done_flags, infos)

        next_obs_b = next_obs_rollout if B > 1 else next_obs_rollout[None, ...]
        if not is_image:
            next_obs_b_norm = obs_norm.normalize(next_obs_b)
        else:
            next_obs_b_norm = next_obs_b

        actions_seq[t] = a_np if B > 1 else (a_np if is_discrete else a_np[0:1, :])
        rew_ext_seq[t] = np.asarray(rewards, dtype=np.float32).reshape(B)
        dones_seq[t] = np.asarray(done_flags, dtype=np.float32).reshape(B)
        terminals_seq[t] = terms_b.astype(np.float32, copy=False)
        truncations_seq[t] = truncs_b.astype(np.float32, copy=False)

        if not is_image:
            next_obs_seq[t] = next_obs_b_norm.astype(np.float32)
        else:
            next_obs_seq_list.append(np.array(next_obs_b_norm, copy=True))

        if r_int_raw_seq is not None:
            int_step_t0 = time.perf_counter()
            r_step = intrinsic_module.compute_impact_binned(
                obs_b_norm,
                next_obs_b_norm,
                dones=prev_done_flags,
                reduction="none",
            )
            r_step_np = r_step.detach().cpu().numpy().reshape(B).astype(np.float32)
            r_step_np[terms_b] = 0.0
            r_int_raw_seq[t] = r_step_np
            t_rollout_intrinsic_step += time.perf_counter() - int_step_t0

            if prev_done_flags is not None:
                prev_done_flags = done_flags.astype(bool, copy=False)

        obs_var = next_obs_env

    if r_int_raw_seq is not None and intrinsic_module is not None and prev_done_flags is not None:
        try:
            setattr(
                intrinsic_module,
                "_ride_prev_done_flags",
                prev_done_flags.astype(bool, copy=True),
            )
        except Exception:
            pass

    if not is_image:
        obs_seq_final = obs_seq
        next_obs_seq_final = next_obs_seq
    else:
        obs_seq_final = np.stack(obs_seq_list, axis=0)
        next_obs_seq_final = np.stack(next_obs_seq_list, axis=0)

        try:
            if T >= 2:
                sample_pairs = int(min(16, T - 1))
                idxs = np.linspace(0, T - 2, sample_pairs, dtype=np.int64)
                b0 = 0
                same = 0
                for ti in idxs:
                    if np.array_equal(obs_seq_final[ti, b0], obs_seq_final[ti + 1, b0]):
                        same += 1
                frac_same = float(same) / float(sample_pairs)
                if frac_same > 0.9:
                    logger.warning(
                        "Image rollout check: %.0f%% sampled consecutive frames identical.",
                        100.0 * frac_same,
                    )
        except Exception:
            pass

    obs_seq_final = _ensure_time_major_np(obs_seq_final, T, B, "obs_seq")
    next_obs_seq_final = _ensure_time_major_np(next_obs_seq_final, T, B, "next_obs_seq")
    rew_ext_seq = _ensure_time_major_np(rew_ext_seq, T, B, "rewards")
    dones_seq = _ensure_time_major_np(dones_seq, T, B, "dones")
    terminals_seq = _ensure_time_major_np(terminals_seq, T, B, "terminals")
    truncations_seq = _ensure_time_major_np(truncations_seq, T, B, "truncations")
    if r_int_raw_seq is not None:
        r_int_raw_seq = _ensure_time_major_np(r_int_raw_seq, T, B, "r_int_raw")

    t_rollout_total = time.perf_counter() - t_rollout_start
    t_rollout_other = max(
        0.0, t_rollout_total - (t_rollout_policy + t_rollout_env_step + t_rollout_intrinsic_step)
    )

    return RolloutBatch(
        T=int(T),
        B=int(B),
        obs_seq=obs_seq_final,
        next_obs_seq=next_obs_seq_final,
        actions_seq=actions_seq,
        rewards_ext_seq=rew_ext_seq,
        dones_seq=dones_seq,
        terminals_seq=terminals_seq,
        truncations_seq=truncations_seq,
        obs_shape=tuple(int(x) for x in obs_shape),
        is_discrete=bool(is_discrete),
        r_int_raw_seq=r_int_raw_seq,
        episode_returns=ep_returns,
        episode_lengths=ep_lengths,
        episode_successes=ep_successes,
        time_rollout_s=float(t_rollout_total),
        time_rollout_policy_s=float(t_rollout_policy),
        time_rollout_env_step_s=float(t_rollout_env_step),
        time_rollout_intrinsic_step_s=float(t_rollout_intrinsic_step),
        time_rollout_other_s=float(t_rollout_other),
        final_env_obs=obs_var,
        steps_collected=int(T) * int(B),
    )
