from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from irl.envs.builder import make_env
from irl.intrinsic.factory import create_intrinsic_module
from irl.models import PolicyNetwork
from irl.pipelines.runtime import build_obs_normalizer, extract_env_runtime
from irl.trainer.build import single_spaces
from irl.utils.checkpoint import load_checkpoint
from irl.utils.determinism import seed_everything
from irl.utils.spaces import is_image_space


def _seed_torch(seed: int) -> None:
    s = int(seed)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(s)
        except Exception:
            pass


def evaluate(
    *,
    env: str,
    ckpt: Path,
    episodes: int = 20,
    device: str = "cpu",
    save_traj: bool = False,
    traj_out_dir: Path | None = None,
    policy_mode: str = "mode",
    episode_seeds: Sequence[int] | None = None,
    seed_offset: int = 0,
) -> dict:
    payload = load_checkpoint(ckpt, map_location=device)
    cfg = payload.get("cfg", {}) or {}
    seed_cfg = int(cfg.get("seed", 1))
    seed_eval_base = int(seed_cfg) + int(seed_offset)
    step = int(payload.get("step", -1))

    seed_everything(seed_cfg, deterministic=True)

    runtime = extract_env_runtime(cfg)
    frame_skip = int(runtime["frame_skip"])
    discrete_actions = bool(runtime["discrete_actions"])
    car_action_set = runtime["car_action_set"]

    e = make_env(
        env_id=env,
        num_envs=1,
        seed=seed_eval_base,
        frame_skip=frame_skip,
        domain_randomization=False,
        discrete_actions=discrete_actions,
        car_action_set=car_action_set,
    )
    obs_space, act_space = single_spaces(e)

    policy = PolicyNetwork(obs_space, act_space).to(device)
    policy.load_state_dict(payload["policy"])
    policy.eval()

    intrinsic_module = None
    method = str(cfg.get("method", "vanilla"))
    if save_traj and "intrinsic" in payload:
        try:
            intr_state = payload["intrinsic"]
            int_cfg = cfg.get("intrinsic", {}) if isinstance(cfg, dict) else {}
            gate_cfg = int_cfg.get("gate", {}) if isinstance(int_cfg, dict) else {}
            intrinsic_module = create_intrinsic_module(
                method,
                obs_space,
                act_space,
                device=device,
                bin_size=float(int_cfg.get("bin_size", 0.25)),
                alpha_impact=float(int_cfg.get("alpha_impact", 1.0)),
                alpha_lp=float(int_cfg.get("alpha_lp", 0.5)),
                region_capacity=int(int_cfg.get("region_capacity", 200)),
                depth_max=int(int_cfg.get("depth_max", 12)),
                ema_beta_long=float(int_cfg.get("ema_beta_long", 0.995)),
                ema_beta_short=float(int_cfg.get("ema_beta_short", 0.9)),
                gate_tau_lp_mult=float(gate_cfg.get("tau_lp_mult", 0.01)),
                gate_tau_s=float(gate_cfg.get("tau_s", 2.0)),
                gate_hysteresis_up_mult=float(gate_cfg.get("hysteresis_up_mult", 2.0)),
                gate_min_consec_to_gate=int(gate_cfg.get("min_consec_to_gate", 5)),
                gate_min_regions_for_gating=int(gate_cfg.get("min_regions_for_gating", 3)),
                normalize_inside=bool(int_cfg.get("normalize_inside", True)),
                gating_enabled=bool(gate_cfg.get("enabled", True)),
            )
            if isinstance(intr_state, dict) and "state_dict" in intr_state:
                intrinsic_module.load_state_dict(intr_state["state_dict"])
            intrinsic_module.eval()
        except Exception:
            intrinsic_module = None

    is_image = is_image_space(obs_space)
    norm = None if is_image else build_obs_normalizer(payload)

    def _normalize(x: np.ndarray) -> np.ndarray:
        if norm is None:
            return x
        mean_arr, std_arr = norm
        return (x - mean_arr) / std_arr

    def _glpe_gate_and_intrinsic_no_update(
        mod: object, obs_1d: torch.Tensor, next_obs_1d: torch.Tensor
    ) -> tuple[int, float] | None:
        if not (hasattr(mod, "icm") and hasattr(mod, "store")):
            return None
        stats = getattr(mod, "_stats", None)
        if not isinstance(stats, dict):
            return None
        try:
            with torch.no_grad():
                b_obs = obs_1d.unsqueeze(0)
                b_next = next_obs_1d.unsqueeze(0)

                phi_t = mod.icm._phi(b_obs)
                rid = int(mod.store.locate(phi_t.detach().cpu().numpy().reshape(-1)))

                st = stats.get(rid)
                gate = 1 if st is None else int(getattr(st, "gate", 1))

                lp_raw = 0.0
                if st is not None and int(getattr(st, "count", 0)) > 0:
                    lp_raw = max(
                        0.0,
                        float(getattr(st, "ema_long", 0.0) - getattr(st, "ema_short", 0.0)),
                    )

                impact_raw_t = mod._impact_per_sample(b_obs, b_next)
                impact_raw = float(impact_raw_t.view(-1)[0].item())

                a_imp = float(getattr(mod, "alpha_impact", 1.0))
                a_lp = float(getattr(mod, "alpha_lp", 0.0))

                if bool(getattr(mod, "_normalize_inside", False)) and hasattr(mod, "_rms"):
                    imp_n, lp_n = mod._rms.normalize(
                        np.asarray([impact_raw], dtype=np.float32),
                        np.asarray([lp_raw], dtype=np.float32),
                    )
                    combined = a_imp * float(imp_n[0]) + a_lp * float(lp_n[0])
                else:
                    combined = a_imp * float(impact_raw) + a_lp * float(lp_raw)

                return int(gate), float(gate) * float(combined)
        except Exception:
            return None

    ep_n = int(episodes)
    if ep_n <= 0:
        e.close()
        raise ValueError("episodes must be >= 1")

    if episode_seeds is None:
        episode_seeds_list = [seed_eval_base + i for i in range(ep_n)]
    else:
        episode_seeds_list = [int(s) for s in episode_seeds]
        if len(episode_seeds_list) != ep_n:
            e.close()
            raise ValueError("episode_seeds length must match episodes")

    def _run(action_mode: str, *, save_traj_local: bool) -> dict:
        mode = str(action_mode).strip().lower()
        if mode not in {"mode", "sample"}:
            raise ValueError("policy_mode must be 'mode' or 'sample'")

        returns: list[float] = []
        lengths: list[int] = []

        traj_obs: list[np.ndarray] = []
        traj_gates: list[int] = []
        traj_int_vals: list[float] = []
        traj_gate_source: str | None = None

        method_l = str(method).strip().lower()
        is_glpe_family = method_l.startswith("glpe")

        for ep_seed in episode_seeds_list:
            _seed_torch(int(ep_seed))
            obs, _ = e.reset(seed=int(ep_seed))
            done = False
            ep_ret = 0.0
            ep_len = 0

            while not done:
                x_raw = obs if isinstance(obs, np.ndarray) else np.asarray(obs, dtype=np.float32)
                x = _normalize(x_raw)

                with torch.no_grad():
                    obs_t = torch.as_tensor(x, dtype=torch.float32, device=device)
                    dist = policy.distribution(obs_t)
                    act = dist.mode() if mode == "mode" else dist.sample()
                    a_np = act.detach().cpu().numpy()

                if hasattr(act_space, "n"):
                    action_for_env = int(a_np.item())
                else:
                    action_for_env = a_np.reshape(-1)

                next_obs, r, term, trunc, _ = e.step(action_for_env)
                ep_ret += float(r)
                ep_len += 1

                if save_traj_local and not is_image:
                    traj_obs.append(x_raw.copy())

                    gate_val = 1
                    int_val = 0.0
                    gate_source = "n/a" if not is_glpe_family else "recomputed"

                    if intrinsic_module is not None:
                        try:
                            next_raw = (
                                next_obs
                                if isinstance(next_obs, np.ndarray)
                                else np.asarray(next_obs, dtype=np.float32)
                            )
                            next_t = torch.as_tensor(
                                _normalize(next_raw), dtype=torch.float32, device=device
                            )

                            if is_glpe_family:
                                res = _glpe_gate_and_intrinsic_no_update(
                                    intrinsic_module, obs_t, next_t
                                )
                                if res is not None:
                                    gate_val, int_val = res
                                    gate_source = "checkpoint"
                            else:
                                try:
                                    r_out = intrinsic_module.compute_batch(
                                        obs_t.unsqueeze(0),
                                        next_t.unsqueeze(0),
                                        act.unsqueeze(0),
                                        reduction="none",
                                    )
                                    int_val = float(r_out.mean().item())
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    traj_gates.append(int(gate_val))
                    traj_int_vals.append(float(int_val))

                    if traj_gate_source is None:
                        traj_gate_source = gate_source
                    elif traj_gate_source != gate_source:
                        traj_gate_source = "mixed"

                obs = next_obs
                done = bool(term) or bool(trunc)

            returns.append(ep_ret)
            lengths.append(ep_len)

        summary = {
            "env_id": str(env),
            "episodes": int(ep_n),
            "seed": int(seed_cfg),
            "seed_offset": int(seed_offset),
            "episode_seeds": [int(s) for s in episode_seeds_list],
            "policy_mode": mode,
            "checkpoint_step": step,
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "std_return": float(np.std(returns, ddof=0)) if len(returns) > 1 else 0.0,
            "min_return": float(min(returns)) if returns else 0.0,
            "max_return": float(max(returns)) if returns else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "std_length": float(np.std(lengths, ddof=0)) if len(lengths) > 1 else 0.0,
            "returns": [float(x) for x in returns],
            "lengths": [int(x) for x in lengths],
        }

        if save_traj_local and traj_obs:
            out_dir = traj_out_dir or ckpt.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            env_tag = env.replace("/", "-")
            traj_file = out_dir / f"{env_tag}_trajectory.npz"

            gate_src_out = traj_gate_source
            if gate_src_out is None:
                gate_src_out = "recomputed" if is_glpe_family else "n/a"

            np.savez_compressed(
                traj_file,
                obs=np.stack(traj_obs),
                gates=np.array(traj_gates, dtype=np.int8),
                intrinsic=np.array(traj_int_vals, dtype=np.float32),
                env_id=np.array([str(env)], dtype=np.str_),
                method=np.array([str(method)], dtype=np.str_),
                gate_source=np.array([str(gate_src_out)], dtype=np.str_),
            )

        return summary

    pm = str(policy_mode).strip().lower()
    if pm == "both":
        det = _run("mode", save_traj_local=bool(save_traj))
        stoch = _run("sample", save_traj_local=False)
        e.close()
        return {
            "env_id": str(env),
            "episodes": int(ep_n),
            "seed": int(seed_cfg),
            "seed_offset": int(seed_offset),
            "episode_seeds": [int(s) for s in episode_seeds_list],
            "policy_mode": "both",
            "checkpoint_step": step,
            "deterministic": det,
            "stochastic": stoch,
        }

    out = _run(pm, save_traj_local=bool(save_traj))
    e.close()
    return out
