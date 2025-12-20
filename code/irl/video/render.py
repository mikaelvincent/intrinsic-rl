from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch

from irl.checkpoints.runtime import build_obs_normalizer, extract_env_settings
from irl.cli.validators import normalize_policy_mode
from irl.envs.builder import make_env
from irl.models import PolicyNetwork
from irl.pipelines.policy_rollout import iter_policy_rollout
from irl.trainer.build import ensure_mujoco_gl, single_spaces
from irl.utils.checkpoint import load_checkpoint
from irl.utils.determinism import seed_everything
from irl.utils.spaces import is_image_space

from .frames import _add_label, _is_blank_frame, _pad_frames_for_ffmpeg, _render_frame


def render_rollout_video(
    *,
    ckpt_path: Path,
    out_path: Path,
    seed: int = 42,
    max_steps: int = 1000,
    device: str = "cpu",
    policy_mode: str = "mode",
    fps: int = 30,
) -> None:
    pm = normalize_policy_mode(policy_mode, allowed=("mode", "sample"), name="policy_mode")

    payload = load_checkpoint(Path(ckpt_path), map_location=device)
    cfg = payload.get("cfg", {}) or {}

    env_cfg = (cfg.get("env") or {}) if isinstance(cfg, dict) else {}
    env_id = str(env_cfg.get("id") or "MountainCar-v0")
    method = str(cfg.get("method", "vanilla"))
    ckpt_step = int(payload.get("step", -1))

    ensure_mujoco_gl(env_id)
    seed_everything(int(seed), deterministic=True)

    runtime = extract_env_settings(cfg)
    frame_skip = int(runtime["frame_skip"])
    discrete_actions = bool(runtime["discrete_actions"])
    car_action_set = runtime["car_action_set"]

    env = make_env(
        env_id=env_id,
        num_envs=1,
        seed=int(seed),
        frame_skip=frame_skip,
        domain_randomization=False,
        discrete_actions=discrete_actions,
        car_action_set=car_action_set,
        render_mode="rgb_array",
    )

    try:
        obs_space, act_space = single_spaces(env)
        is_image = is_image_space(obs_space)
        norm = None if is_image else build_obs_normalizer(payload)

        def _norm_obs(x: np.ndarray) -> np.ndarray:
            if norm is None:
                return x
            mean_arr, std_arr = norm
            return (x - mean_arr) / std_arr

        policy_torch = PolicyNetwork(obs_space, act_space).to(device)
        policy_torch.load_state_dict(payload["policy"])
        policy_torch.eval()

        obs0, _ = env.reset(seed=int(seed))

        frames: list[np.ndarray] = []
        blank_frames = 0
        ret = 0.0

        f0 = _render_frame(env)
        blank_frames += 1 if _is_blank_frame(f0) else 0
        label0 = f"{env_id} | {method} | step={ckpt_step} | {pm} | eval_seed={seed}"
        frames.append(_add_label(f0, label0, score=ret))

        dev = torch.device(device)

        for step_rec in iter_policy_rollout(
            env=env,
            policy=policy_torch,
            obs0=obs0,
            act_space=act_space,
            device=dev,
            policy_mode=pm,
            normalize_obs=_norm_obs,
            max_steps=int(max_steps),
        ):
            ret += float(step_rec.reward)

            fr = _render_frame(env)
            blank_frames += 1 if _is_blank_frame(fr) else 0
            frames.append(_add_label(fr, label0, score=ret))

        if not frames:
            raise RuntimeError("No frames captured.")

        blank_ratio = float(blank_frames) / float(len(frames))
        if len(frames) >= 10 and blank_ratio > 0.9:
            raise RuntimeError(f"Render appears blank (blank_ratio={blank_ratio:.2f}).")

        frames = _pad_frames_for_ffmpeg(frames, macro_block_size=16)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(out_path), frames, fps=int(fps))
    finally:
        env.close()
