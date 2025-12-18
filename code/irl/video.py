from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from irl.cli.validators import normalize_policy_mode
from irl.envs.builder import make_env
from irl.models import PolicyNetwork
from irl.pipelines.policy_rollout import iter_policy_rollout
from irl.pipelines.runtime import build_obs_normalizer, extract_env_runtime
from irl.trainer.build import ensure_mujoco_gl, single_spaces
from irl.utils.checkpoint import load_checkpoint
from irl.utils.determinism import seed_everything
from irl.utils.spaces import is_image_space


def _render_frame(env: gym.Env) -> np.ndarray:
    try:
        rgb = env.render()
    except Exception:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    if rgb is None:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    if isinstance(rgb, list):
        rgb = np.array(rgb)

    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
    return arr


def _is_blank_frame(frame: np.ndarray) -> bool:
    try:
        f = np.asarray(frame)
        if f.size == 0:
            return True
        if f.ndim != 3:
            return True
        mn = int(f.min())
        mx = int(f.max())
        return mn == mx
    except Exception:
        return True


def _add_label(frame: np.ndarray, label: str, score: float | None = None) -> np.ndarray:
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    text = label if score is None else f"{label}\nReturn: {score:.1f}"
    x, y = 10, 10

    for ox, oy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw.text((x + ox, y + oy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    return np.array(pil_img)


def _ceil_to_multiple(x: int, m: int) -> int:
    mm = int(m)
    if mm <= 1:
        return int(x)
    xi = int(x)
    return int(((xi + mm - 1) // mm) * mm)


def _pad_frame_to(frame: np.ndarray, *, target_h: int, target_w: int) -> np.ndarray:
    f = np.asarray(frame)
    th = int(max(1, target_h))
    tw = int(max(1, target_w))

    if f.ndim != 3 or f.shape[-1] != 3:
        return np.zeros((th, tw, 3), dtype=np.uint8)

    h, w = int(f.shape[0]), int(f.shape[1])
    if h == th and w == tw:
        return f.astype(np.uint8, copy=False)

    f2 = f[: min(h, th), : min(w, tw), :3]
    h2, w2 = int(f2.shape[0]), int(f2.shape[1])

    pad_h = max(0, th - h2)
    pad_w = max(0, tw - w2)
    if pad_h == 0 and pad_w == 0:
        return f2.astype(np.uint8, copy=False)

    mode = "edge" if (h2 > 0 and w2 > 0) else "constant"
    out = np.pad(f2, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    return out.astype(np.uint8, copy=False)


def _pad_frames_for_ffmpeg(frames: list[np.ndarray], *, macro_block_size: int = 16) -> list[np.ndarray]:
    # imageio-ffmpeg may resize frames when dimensions are not divisible by macro_block_size (default 16).
    # Padding avoids unintended rescaling while keeping codec-friendly dimensions.
    if not frames:
        return frames

    mbs = int(max(1, macro_block_size))

    hs: list[int] = []
    ws: list[int] = []
    same = True
    first_hw: tuple[int, int] | None = None

    for fr in frames:
        a = np.asarray(fr)
        if a.ndim != 3:
            same = False
            continue
        h, w = int(a.shape[0]), int(a.shape[1])
        hs.append(h)
        ws.append(w)
        if first_hw is None:
            first_hw = (h, w)
        elif first_hw != (h, w):
            same = False

    if not hs or not ws:
        return frames

    target_h = _ceil_to_multiple(max(hs), mbs)
    target_w = _ceil_to_multiple(max(ws), mbs)

    if same and first_hw == (target_h, target_w):
        return frames

    return [_pad_frame_to(fr, target_h=target_h, target_w=target_w) for fr in frames]


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

    runtime = extract_env_runtime(cfg)
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
