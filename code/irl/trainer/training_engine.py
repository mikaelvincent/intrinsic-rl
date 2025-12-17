from __future__ import annotations

import time
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from irl.algo.advantage import compute_gae
from irl.algo.ppo import ppo_update
from irl.cfg import Config
from irl.intrinsic import compute_intrinsic_batch, update_module
from irl.models import PolicyNetwork
from irl.utils.checkpoint_schema import build_checkpoint_payload

from .runtime_utils import _apply_final_observation, _ensure_time_major_np
from .training_setup import TrainingSession


# CUDA kernels launch asynchronously; synchronize only for timing/profiling.
def _maybe_cuda_sync(device: torch.device, enabled: bool) -> None:
    if not enabled:
        return
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass


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


def run_training_loop(
    cfg: Config,
    session: TrainingSession,
    *,
    total_steps: int,
    logger,
    log_every_updates: int = 10,
) -> None:
    run_dir = session.run_dir
    env = session.env
    obs_space = session.obs_space
    act_space = session.act_space
    is_image = session.is_image
    B = int(session.num_envs)

    policy = session.policy
    value = session.value
    pol_opt = session.optimizers.policy
    val_opt = session.optimizers.value

    intrinsic_module = session.intrinsic.module
    method_l = str(session.intrinsic.method)
    eta = float(session.intrinsic.eta)
    use_intrinsic = bool(session.intrinsic.use_intrinsic)
    intrinsic_norm_mode = str(session.intrinsic.norm_mode)
    intrinsic_outputs_normalized_flag = session.intrinsic.outputs_normalized
    int_rms = session.intrinsic.rms

    obs_norm = session.obs_norm
    ml = session.metric_logger
    device = session.device
    ckpt = session.ckpt

    profile_cuda_sync = False
    try:
        profile_cuda_sync = bool(getattr(getattr(cfg, "exp", None), "profile_cuda_sync", False))
    except Exception:
        profile_cuda_sync = False

    obs = session.obs
    global_step = int(session.global_step)
    update_idx = int(session.update_idx)

    actor_policy: Optional[PolicyNetwork] = None
    if device.type != "cpu":
        actor_policy = PolicyNetwork(obs_space, act_space).to(torch.device("cpu"))
        actor_policy.eval()
        for p in actor_policy.parameters():
            p.requires_grad = False
        logger.info("Using CPU actor policy for env stepping; syncing each PPO update.")

    def _sync_actor_from_learner() -> None:
        if actor_policy is None:
            return
        with torch.no_grad():
            sd = policy.state_dict()
            cpu_sd = {k: v.detach().to("cpu") for k, v in sd.items()}
            actor_policy.load_state_dict(cpu_sd, strict=True)

    if int(global_step) == 0 and int(update_idx) == 0 and not ckpt.latest_path.exists():
        payload0 = build_checkpoint_payload(
            cfg,
            global_step=0,
            update_idx=0,
            policy=policy,
            value=value,
            is_image=is_image,
            obs_norm=obs_norm,
            int_rms=int_rms,
            pol_opt=pol_opt,
            val_opt=val_opt,
            intrinsic_module=intrinsic_module,
            method_l=method_l,
        )
        ckpt_path0 = ckpt.save(step=0, payload=payload0)
        logger.info("Saved baseline checkpoint at step=0 to %s", ckpt_path0)

    start_wall = time.time()
    last_log_time = start_wall
    last_log_step = global_step

    while global_step < int(total_steps):
        t_update_start = time.perf_counter()

        t_rollout_total = 0.0
        t_rollout_policy = 0.0
        t_rollout_env_step = 0.0
        t_rollout_intrinsic_step = 0.0
        t_rollout_other = 0.0

        t_intrinsic_compute = 0.0
        t_intrinsic_update = 0.0
        t_gae = 0.0
        t_ppo = 0.0

        t_logging_compute = 0.0
        t_ml_log = 0.0

        _sync_actor_from_learner()

        per_env_steps = min(
            int(cfg.ppo.steps_per_update),
            max(1, ceil((int(total_steps) - int(global_step)) / B)),
        )
        T = per_env_steps

        if not is_image:
            obs_dim = int(obs_space.shape[0])
            obs_seq = np.zeros((T, B, obs_dim), dtype=np.float32)
            next_obs_seq = np.zeros((T, B, obs_dim), dtype=np.float32)
        else:
            obs_seq_list = []
            next_obs_seq_list = []

        is_discrete = hasattr(act_space, "n")
        acts_seq = (
            np.zeros((T, B), dtype=np.int64)
            if is_discrete
            else np.zeros((T, B, int(act_space.shape[0])), dtype=np.float32)
        )
        rew_ext_seq = np.zeros((T, B), dtype=np.float32)
        done_seq = np.zeros((T, B), dtype=np.float32)
        terms_seq = np.zeros((T, B), dtype=np.float32)
        truncs_seq = np.zeros((T, B), dtype=np.float32)
        r_int_raw_seq = (
            np.zeros((T, B), dtype=np.float32)
            if (intrinsic_module is not None and method_l == "ride" and use_intrinsic)
            else None
        )

        ep_returns: list[float] = []
        ep_lengths: list[int] = []
        ep_successes: list[float] = []

        t_rollout_start = time.perf_counter()

        for t in range(T):
            obs_b = obs if B > 1 else obs[None, ...]
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

            acts_seq[t] = a_np if B > 1 else (a_np if is_discrete else a_np[0:1, :])
            rew_ext_seq[t] = np.asarray(rewards, dtype=np.float32).reshape(B)
            done_seq[t] = np.asarray(done_flags, dtype=np.float32).reshape(B)
            terms_seq[t] = terms_b.astype(np.float32, copy=False)
            truncs_seq[t] = truncs_b.astype(np.float32, copy=False)

            if not is_image:
                next_obs_seq[t] = next_obs_b_norm.astype(np.float32)
            else:
                next_obs_seq_list.append(np.array(next_obs_b_norm, copy=True))

            if r_int_raw_seq is not None:
                int_step_t0 = time.perf_counter()
                r_step = intrinsic_module.compute_impact_binned(
                    obs_b_norm,
                    next_obs_b_norm,
                    dones=done_flags,
                    reduction="none",
                )
                r_step_np = r_step.detach().cpu().numpy().reshape(B).astype(np.float32)
                r_step_np[terms_b] = 0.0
                r_int_raw_seq[t] = r_step_np
                t_rollout_intrinsic_step += time.perf_counter() - int_step_t0

            obs = next_obs_env
            global_step += B

        if not is_image:
            obs_seq_final = obs_seq
            next_obs_seq_final = next_obs_seq
            obs_shape = (int(obs_space.shape[0]),)
        else:
            obs_seq_final = np.stack(obs_seq_list, axis=0)
            next_obs_seq_final = np.stack(next_obs_seq_list, axis=0)
            obs_shape = tuple(int(s) for s in obs_space.shape)

        if is_image:
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
        done_seq = _ensure_time_major_np(done_seq, T, B, "dones")
        terms_seq = _ensure_time_major_np(terms_seq, T, B, "terminals")
        truncs_seq = _ensure_time_major_np(truncs_seq, T, B, "truncations")
        if r_int_raw_seq is not None:
            r_int_raw_seq = _ensure_time_major_np(r_int_raw_seq, T, B, "r_int_raw")

        t_rollout_total = time.perf_counter() - t_rollout_start
        t_rollout_other = max(
            0.0,
            t_rollout_total - (t_rollout_policy + t_rollout_env_step + t_rollout_intrinsic_step),
        )

        ep_count = int(len(ep_returns))
        ep_return_mean = float(np.mean(ep_returns)) if ep_returns else float("nan")
        ep_return_std = float(np.std(ep_returns, ddof=0)) if ep_returns else float("nan")
        ep_length_mean = float(np.mean(ep_lengths)) if ep_lengths else float("nan")
        ep_length_std = float(np.std(ep_lengths, ddof=0)) if ep_lengths else float("nan")
        ep_success_rate = float(np.mean(ep_successes)) if ep_successes else float("nan")

        r_int_raw_flat = None
        r_int_scaled_flat = None
        mod_metrics = {}

        if intrinsic_module is not None and use_intrinsic:
            intrinsic_compute_t0 = time.perf_counter()

            if method_l == "ride":
                r_int_raw_flat = r_int_raw_seq.reshape(T * B).astype(np.float32)
            else:
                _maybe_cuda_sync(device, profile_cuda_sync)

                obs_flat = obs_seq_final.reshape((T * B,) + obs_shape)
                next_obs_flat = next_obs_seq_final.reshape((T * B,) + obs_shape)
                acts_flat = acts_seq.reshape(T * B) if is_discrete else acts_seq.reshape(T * B, -1)

                r_int_raw_t = compute_intrinsic_batch(
                    intrinsic_module,
                    method_l,
                    obs_flat,
                    next_obs_flat,
                    acts_flat,
                )
                r_int_raw_flat = r_int_raw_t.detach().cpu().numpy().astype(np.float32)

            term_mask_flat = (terms_seq.reshape(T * B) > 0.0)
            r_int_raw_flat = np.asarray(r_int_raw_flat, dtype=np.float32)
            r_int_raw_flat[term_mask_flat] = 0.0

            r_clip = float(cfg.intrinsic.r_clip)
            outputs_norm = bool(getattr(intrinsic_module, "outputs_normalized", False))
            if outputs_norm:
                r_int_scaled_flat = eta * np.clip(r_int_raw_flat, -r_clip, r_clip)
            else:
                int_rms.update(r_int_raw_flat)
                r_int_norm_flat = int_rms.normalize(r_int_raw_flat)
                r_int_scaled_flat = eta * np.clip(r_int_norm_flat, -r_clip, r_clip)

            _maybe_cuda_sync(device, profile_cuda_sync)
            t_intrinsic_compute = time.perf_counter() - intrinsic_compute_t0

            intrinsic_update_t0 = time.perf_counter()
            try:
                if method_l == "ride":
                    obs_flat = obs_seq_final.reshape((T * B,) + obs_shape)
                    next_obs_flat = next_obs_seq_final.reshape((T * B,) + obs_shape)
                    acts_flat = acts_seq.reshape(T * B) if is_discrete else acts_seq.reshape(T * B, -1)

                _maybe_cuda_sync(device, profile_cuda_sync)

                mod_metrics = update_module(
                    intrinsic_module,
                    method_l,
                    obs_flat,
                    next_obs_flat,
                    acts_flat,
                )

                _maybe_cuda_sync(device, profile_cuda_sync)
            except Exception:
                mod_metrics = {}
            t_intrinsic_update = time.perf_counter() - intrinsic_update_t0

        if r_int_scaled_flat is not None:
            rew_total_seq = rew_ext_seq + r_int_scaled_flat.reshape(T, B)
        else:
            rew_total_seq = rew_ext_seq

        gae_batch = {
            "obs": obs_seq_final,
            "next_observations": next_obs_seq_final,
            "rewards": rew_total_seq,
            "terminals": terms_seq,
            "truncations": truncs_seq,
        }
        gae_t0 = time.perf_counter()
        _maybe_cuda_sync(device, profile_cuda_sync)
        adv, v_targets = compute_gae(
            gae_batch,
            value,
            gamma=float(cfg.ppo.gamma),
            lam=float(cfg.ppo.gae_lambda),
            bootstrap_on_timeouts=True,
        )
        _maybe_cuda_sync(device, profile_cuda_sync)
        t_gae = time.perf_counter() - gae_t0

        obs_flat_for_ppo = (
            obs_seq_final.reshape(T * B, -1)
            if not is_image
            else obs_seq_final.reshape((T * B,) + obs_shape)
        )
        acts_flat_for_ppo = acts_seq.reshape(T * B) if is_discrete else acts_seq.reshape(T * B, -1)
        batch = {
            "obs": obs_flat_for_ppo,
            "actions": acts_flat_for_ppo,
            "rewards": rew_total_seq.reshape(T * B),
            "dones": done_seq.reshape(T * B),
        }

        ppo_t0 = time.perf_counter()
        _maybe_cuda_sync(device, profile_cuda_sync)
        ppo_stats = ppo_update(
            policy,
            value,
            batch,
            adv,
            v_targets,
            cfg.ppo,
            optimizers=(pol_opt, val_opt),
            return_stats=True,
        )
        _maybe_cuda_sync(device, profile_cuda_sync)
        t_ppo = time.perf_counter() - ppo_t0

        update_idx += 1

        log_compute_t0 = time.perf_counter()
        with torch.no_grad():
            last_obs = obs_seq_final[-1]
            if is_image:
                ent_last = float(policy.entropy(last_obs).mean().item())
            else:
                last_obs_t = torch.as_tensor(last_obs, device=device, dtype=torch.float32)
                ent_last = float(policy.entropy(last_obs_t).mean().item())

            ENTROPY_MAX_SAMPLES = 1024
            total_samples = int(T * B)

            ent_mean_update = float("nan")
            if total_samples <= ENTROPY_MAX_SAMPLES:
                if is_image:
                    ent_mean_update = float(policy.entropy(obs_flat_for_ppo).mean().item())
                else:
                    obs_flat_t = torch.as_tensor(
                        obs_flat_for_ppo, device=device, dtype=torch.float32
                    )
                    ent_mean_update = float(policy.entropy(obs_flat_t).mean().item())

        log_payload = {
            "entropy_last": float(ent_last),
            "entropy_update_mean": float(ent_mean_update),
            "reward_mean": float(rew_ext_seq.mean()),
            "reward_total_mean": float(rew_total_seq.mean()),
            "episode_count": int(ep_count),
            "episode_return_mean": float(ep_return_mean),
            "episode_return_std": float(ep_return_std),
            "episode_length_mean": float(ep_length_mean),
            "episode_length_std": float(ep_length_std),
            "success_rate": float(ep_success_rate),
            "intrinsic_norm_mode": intrinsic_norm_mode,
        }
        if intrinsic_outputs_normalized_flag is not None:
            log_payload["intrinsic_outputs_normalized"] = bool(intrinsic_outputs_normalized_flag)

        if isinstance(ppo_stats, dict):
            try:
                clip_frac = float(ppo_stats.get("clip_frac", float("nan")))
                log_payload.update(
                    {
                        "approx_kl": float(ppo_stats.get("approx_kl", float("nan"))),
                        "clip_frac": clip_frac,
                        "clip_frac_pct": (100.0 * clip_frac)
                        if np.isfinite(clip_frac)
                        else float("nan"),
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
                if intrinsic_module
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
                        log_payload["impact_rms"] = imp_rms_val
                        log_payload["lp_rms"] = lp_rms_val
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
                    "r_int_rms": r_int_rms_val,
                }
            )
            for k, v in (mod_metrics or {}).items():
                try:
                    log_payload[f"{method_l}_{k}"] = float(v)
                except Exception:
                    pass

        if (
            intrinsic_module is not None
            and method_l.startswith("glpe")
            and hasattr(intrinsic_module, "gate_rate")
        ):
            try:
                gr = float(getattr(intrinsic_module, "gate_rate"))
                log_payload["gate_rate"] = gr
                log_payload["gate_rate_pct"] = 100.0 * gr
            except Exception:
                pass

        log_payload.update(
            {
                "time_rollout_s": float(t_rollout_total),
                "time_rollout_policy_s": float(t_rollout_policy),
                "time_rollout_env_step_s": float(t_rollout_env_step),
                "time_rollout_intrinsic_step_s": float(t_rollout_intrinsic_step),
                "time_rollout_other_s": float(t_rollout_other),
                "time_intrinsic_compute_s": float(t_intrinsic_compute),
                "time_intrinsic_update_s": float(t_intrinsic_update),
                "time_gae_s": float(t_gae),
                "time_ppo_s": float(t_ppo),
            }
        )

        t_logging_compute = time.perf_counter() - log_compute_t0

        ml_t0 = time.perf_counter()
        ml.log(step=int(global_step), **log_payload)
        t_ml_log = time.perf_counter() - ml_t0

        t_update_total = time.perf_counter() - t_update_start

        if update_idx % int(log_every_updates) == 0 or global_step >= int(total_steps):
            now = time.time()
            elapsed = max(now - start_wall, 1e-6)
            avg_sps = float(global_step) / elapsed if global_step > 0 else 0.0
            delta_steps = int(global_step - last_log_step)
            delta_t = max(now - last_log_time, 1e-6)
            recent_sps = float(delta_steps) / delta_t if delta_steps > 0 else 0.0

            approx_kl = float(log_payload.get("approx_kl", float("nan")))
            clip_frac = float(log_payload.get("clip_frac", float("nan")))
            r_total = float(log_payload.get("reward_total_mean", float("nan")))
            r_int_mean = (
                float(log_payload.get("r_int_mean", 0.0)) if "r_int_mean" in log_payload else 0.0
            )

            logger.info(
                "Train progress: step=%d update=%d avg_sps=%.1f recent_sps=%.1f "
                "reward_total_mean=%.3f r_int_mean=%.3f approx_kl=%.4f clip_frac=%.3f",
                int(global_step),
                int(update_idx),
                avg_sps,
                recent_sps,
                r_total,
                r_int_mean,
                approx_kl,
                clip_frac,
            )

            intrinsic_total = float(
                t_rollout_intrinsic_step + t_intrinsic_compute + t_intrinsic_update
            )
            logging_total = float(t_logging_compute + t_ml_log)
            logger.info(
                "Timings (s) update=%d: total=%.3f | rollout=%.3f (policy=%.3f, env_step=%.3f, "
                "intr_step=%.3f, other=%.3f) | intrinsic_total=%.3f (batch_compute=%.3f, update=%.3f) | "
                "gae=%.3f | ppo=%.3f | logging=%.3f (compute=%.3f, io=%.3f)",
                int(update_idx),
                float(t_update_total),
                float(t_rollout_total),
                float(t_rollout_policy),
                float(t_rollout_env_step),
                float(t_rollout_intrinsic_step),
                float(t_rollout_other),
                intrinsic_total,
                float(t_intrinsic_compute),
                float(t_intrinsic_update),
                float(t_gae),
                float(t_ppo),
                logging_total,
                float(t_logging_compute),
                float(t_ml_log),
            )

            last_log_time = now
            last_log_step = global_step

        try:
            if (
                intrinsic_module is not None
                and use_intrinsic
                and method_l == "riac"
                and int(global_step) % int(cfg.logging.csv_interval) == 0
                and hasattr(intrinsic_module, "export_diagnostics")
            ):
                diag_dir = run_dir / "diagnostics"
                intrinsic_module.export_diagnostics(diag_dir, step=int(global_step))
        except Exception:
            pass

        if ckpt.should_save(int(global_step)):
            payload = build_checkpoint_payload(
                cfg,
                global_step=int(global_step),
                update_idx=int(update_idx),
                policy=policy,
                value=value,
                is_image=is_image,
                obs_norm=obs_norm,
                int_rms=int_rms,
                pol_opt=pol_opt,
                val_opt=val_opt,
                intrinsic_module=intrinsic_module,
                method_l=method_l,
            )
            ckpt_path = ckpt.save(step=int(global_step), payload=payload)
            logger.info("Saved checkpoint at step=%d to %s", int(global_step), ckpt_path)

    payload = build_checkpoint_payload(
        cfg,
        global_step=int(global_step),
        update_idx=int(update_idx),
        policy=policy,
        value=value,
        is_image=is_image,
        obs_norm=obs_norm,
        int_rms=int_rms,
        pol_opt=pol_opt,
        val_opt=val_opt,
        intrinsic_module=intrinsic_module,
        method_l=method_l,
    )
    final_ckpt_path = ckpt.save(step=int(global_step), payload=payload)
    logger.info("Saved final checkpoint at step=%d to %s", int(global_step), final_ckpt_path)

    session.obs = obs
    session.global_step = int(global_step)
    session.update_idx = int(update_idx)
