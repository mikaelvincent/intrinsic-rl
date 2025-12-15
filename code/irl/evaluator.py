from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from irl.envs import EnvManager
from irl.intrinsic.factory import create_intrinsic_module
from irl.models import PolicyNetwork
from irl.trainer.build import single_spaces
from irl.utils.checkpoint import load_checkpoint
from irl.utils.determinism import seed_everything


def _is_image_space(space) -> bool:
    return hasattr(space, "shape") and len(space.shape) >= 2


def _build_normalizer(payload) -> tuple[np.ndarray, np.ndarray] | None:
    on = payload.get("obs_norm")
    if on is None:
        return None
    mean_arr = np.asarray(on.get("mean"), dtype=np.float64)
    var_arr = np.asarray(on.get("var"), dtype=np.float64)
    std_arr = np.sqrt(var_arr + 1e-8)
    return mean_arr, std_arr


def evaluate(
    *,
    env: str,
    ckpt: Path,
    episodes: int = 10,
    device: str = "cpu",
    save_traj: bool = False,
    traj_out_dir: Path | None = None,
) -> dict:
    payload = load_checkpoint(ckpt, map_location=device)
    cfg = payload.get("cfg", {}) or {}
    seed = int(cfg.get("seed", 1))
    step = int(payload.get("step", -1))

    seed_everything(seed, deterministic=True)

    env_cfg = (cfg.get("env") or {}) if isinstance(cfg, dict) else {}
    frame_skip = int(env_cfg.get("frame_skip", 1))
    discrete_actions = bool(env_cfg.get("discrete_actions", True))
    car_action_set = env_cfg.get("car_discrete_action_set", None)

    manager = EnvManager(
        env_id=env,
        num_envs=1,
        seed=seed,
        frame_skip=frame_skip,
        domain_randomization=False,
        discrete_actions=discrete_actions,
        car_action_set=car_action_set,
    )
    e = manager.make()
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

    is_image = _is_image_space(obs_space)
    norm = None if is_image else _build_normalizer(payload)

    def _normalize(x: np.ndarray) -> np.ndarray:
        if norm is None:
            return x
        mean_arr, std_arr = norm
        return (x - mean_arr) / std_arr

    returns: list[float] = []
    lengths: list[int] = []

    traj_obs: list[np.ndarray] = []
    traj_gates: list[int] = []
    traj_int_vals: list[float] = []

    for _ in range(int(episodes)):
        obs, _ = e.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0

        while not done:
            x_raw = obs if isinstance(obs, np.ndarray) else np.asarray(obs, dtype=np.float32)
            x = _normalize(x_raw)

            with torch.no_grad():
                obs_t = torch.as_tensor(x, dtype=torch.float32, device=device)
                dist = policy.distribution(obs_t)
                act = dist.mode()
                a_np = act.detach().cpu().numpy()

            if hasattr(act_space, "n"):
                action_for_env = int(a_np.item())
            else:
                action_for_env = a_np.reshape(-1)

            next_obs, r, term, trunc, _ = e.step(action_for_env)
            ep_ret += float(r)
            ep_len += 1

            if save_traj and not is_image:
                traj_obs.append(x_raw.copy())
                gate_val = 1
                int_val = 0.0
                if intrinsic_module is not None:
                    batch_obs = obs_t.unsqueeze(0)
                    batch_next = torch.as_tensor(
                        _normalize(next_obs), dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    batch_act = act.unsqueeze(0)
                    try:
                        r_out = intrinsic_module.compute_batch(
                            batch_obs, batch_next, batch_act, reduction="none"
                        )
                        int_val = float(r_out.mean().item())
                        if method == "proposed" and hasattr(intrinsic_module, "store"):
                            with torch.no_grad():
                                phi = intrinsic_module.icm._phi(batch_obs)
                                phi_np = phi.cpu().numpy().reshape(-1)
                            rid = intrinsic_module.store.locate(phi_np)
                            st = intrinsic_module._stats.get(rid)
                            if st is not None:
                                gate_val = int(st.gate)
                    except Exception:
                        pass
                traj_gates.append(gate_val)
                traj_int_vals.append(int_val)

            obs = next_obs
            done = bool(term) or bool(trunc)

        returns.append(ep_ret)
        lengths.append(ep_len)

    summary = {
        "env_id": str(env),
        "episodes": int(episodes),
        "seed": seed,
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

    if save_traj and traj_obs:
        out_dir = traj_out_dir or ckpt.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        env_tag = env.replace("/", "-")
        traj_file = out_dir / f"{env_tag}_trajectory.npz"
        np.savez_compressed(
            traj_file,
            obs=np.stack(traj_obs),
            gates=np.array(traj_gates),
            intrinsic=np.array(traj_int_vals),
        )

    e.close()
    return summary
