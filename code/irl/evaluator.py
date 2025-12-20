from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from irl.cli.validators import normalize_policy_mode
from irl.envs.builder import make_env
from irl.evaluation.rollout import RolloutResult, run_eval_episodes
from irl.evaluation.session import build_eval_session
from irl.evaluation.trajectory import save_trajectory_npz
from irl.models import PolicyNetwork
from irl.utils.checkpoint import load_checkpoint
from irl.utils.determinism import seed_everything


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
    ckpt_path = Path(ckpt)
    payload = load_checkpoint(ckpt_path, map_location=device)
    cfg = payload.get("cfg", {}) or {}
    seed_cfg = int(cfg.get("seed", 1))
    seed_eval_base = int(seed_cfg) + int(seed_offset)
    step = int(payload.get("step", -1))

    seed_everything(seed_cfg, deterministic=True)

    session = build_eval_session(
        env_id=str(env),
        cfg=cfg,
        payload=payload,
        device=str(device),
        seed_eval_base=int(seed_eval_base),
        save_traj=bool(save_traj),
        make_env_fn=make_env,
        policy_cls=PolicyNetwork,
    )
    e = session.env

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

    method = str(cfg.get("method", "vanilla"))
    dev = torch.device(device)

    def _write_traj(rr: RolloutResult) -> None:
        if not bool(save_traj) or rr.trajectory is None or not rr.trajectory.obs:
            return
        out_dir = traj_out_dir or ckpt_path.parent
        save_trajectory_npz(
            out_dir=Path(out_dir),
            env_id=str(env),
            method=str(method),
            obs=rr.trajectory.obs,
            rewards_ext=rr.trajectory.rewards_ext,
            gates=rr.trajectory.gates,
            intrinsic=rr.trajectory.intrinsic,
            gate_source=rr.trajectory.gate_source,
        )

    def _summary(rr: RolloutResult, mode: str) -> dict:
        returns = rr.returns
        lengths = rr.lengths
        return {
            "env_id": str(env),
            "episodes": int(ep_n),
            "seed": int(seed_cfg),
            "seed_offset": int(seed_offset),
            "episode_seeds": [int(s) for s in episode_seeds_list],
            "policy_mode": str(mode),
            "checkpoint_step": int(step),
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "std_return": float(np.std(returns, ddof=0)) if len(returns) > 1 else 0.0,
            "min_return": float(min(returns)) if returns else 0.0,
            "max_return": float(max(returns)) if returns else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "std_length": float(np.std(lengths, ddof=0)) if len(lengths) > 1 else 0.0,
            "returns": [float(x) for x in returns],
            "lengths": [int(x) for x in lengths],
        }

    pm = normalize_policy_mode(policy_mode, allowed=("mode", "sample", "both"), name="policy_mode")

    if pm == "both":
        det_rr = run_eval_episodes(
            env=e,
            policy=session.policy,
            act_space=session.act_space,
            device=dev,
            policy_mode="mode",
            episode_seeds=episode_seeds_list,
            normalize_obs=session.normalize_obs,
            save_traj=bool(save_traj),
            is_image=bool(session.is_image),
            intrinsic_module=session.intrinsic_module,
            method=str(method),
        )
        _write_traj(det_rr)
        det = _summary(det_rr, "mode")

        stoch_rr = run_eval_episodes(
            env=e,
            policy=session.policy,
            act_space=session.act_space,
            device=dev,
            policy_mode="sample",
            episode_seeds=episode_seeds_list,
            normalize_obs=session.normalize_obs,
            save_traj=False,
            is_image=bool(session.is_image),
            intrinsic_module=session.intrinsic_module,
            method=str(method),
        )
        stoch = _summary(stoch_rr, "sample")

        e.close()
        return {
            "env_id": str(env),
            "episodes": int(ep_n),
            "seed": int(seed_cfg),
            "seed_offset": int(seed_offset),
            "episode_seeds": [int(s) for s in episode_seeds_list],
            "policy_mode": "both",
            "checkpoint_step": int(step),
            "deterministic": det,
            "stochastic": stoch,
        }

    rr = run_eval_episodes(
        env=e,
        policy=session.policy,
        act_space=session.act_space,
        device=dev,
        policy_mode=str(pm),
        episode_seeds=episode_seeds_list,
        normalize_obs=session.normalize_obs,
        save_traj=bool(save_traj),
        is_image=bool(session.is_image),
        intrinsic_module=session.intrinsic_module,
        method=str(method),
    )
    _write_traj(rr)

    out = _summary(rr, str(pm))
    e.close()
    return out
