from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from irl.envs.builder import make_env
from irl.models import PolicyNetwork
from irl.trainer.build import ensure_mujoco_gl, single_spaces
from irl.utils.checkpoint import load_checkpoint
from irl.utils.determinism import seed_everything
from irl.utils.spaces import is_image_space


def _build_normalizer(payload) -> tuple[np.ndarray, np.ndarray] | None:
    on = payload.get("obs_norm")
    if on is None:
        return None
    mean_arr = np.asarray(on.get("mean"), dtype=np.float64)
    var_arr = np.asarray(on.get("var"), dtype=np.float64)
    std_arr = np.sqrt(var_arr + 1e-8)
    return mean_arr, std_arr


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
    pm = str(policy_mode).strip().lower()
    if pm not in {"mode", "sample"}:
        raise ValueError("policy_mode must be 'mode' or 'sample'")

    payload = load_checkpoint(Path(ckpt_path), map_location=device)
    cfg = payload.get("cfg", {}) or {}

    env_cfg = (cfg.get("env") or {}) if isinstance(cfg, dict) else {}
    env_id = str(env_cfg.get("id") or "MountainCar-v0")
    method = str(cfg.get("method", "vanilla"))
    ckpt_step = int(payload.get("step", -1))

    ensure_mujoco_gl(env_id)
    seed_everything(int(seed), deterministic=True)

    frame_skip = int(env_cfg.get("frame_skip", 1))
    discrete_actions = bool(env_cfg.get("discrete_actions", True))
    car_action_set = env_cfg.get("car_discrete_action_set", None)

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
        norm = None if is_image else _build_normalizer(payload)

        def _norm_obs(x: np.ndarray) -> np.ndarray:
            if norm is None:
                return x
            mean_arr, std_arr = norm
            return (x - mean_arr) / std_arr

        policy = PolicyNetwork(obs_space, act_space).to(device)
        policy.load_state_dict(payload["policy"])
        policy.eval()

        obs, _ = env.reset(seed=int(seed))

        frames: list[np.ndarray] = []
        blank_frames = 0
        ret = 0.0

        f0 = _render_frame(env)
        blank_frames += 1 if _is_blank_frame(f0) else 0
        label0 = f"{env_id} | {method} | step={ckpt_step} | {pm} | eval_seed={seed}"
        frames.append(_add_label(f0, label0, score=ret))

        done = False
        for _ in range(int(max_steps)):
            if done:
                break

            x_raw = obs if isinstance(obs, np.ndarray) else np.asarray(obs)
            x_in = _norm_obs(x_raw) if not is_image else x_raw

            with torch.no_grad():
                obs_t = torch.as_tensor(x_in, dtype=torch.float32, device=device)
                dist = policy.distribution(obs_t)
                act = dist.mode() if pm == "mode" else dist.sample()
                a_np = act.detach().cpu().numpy()

            if hasattr(act_space, "n"):
                action_for_env = int(a_np.item())
            else:
                action_for_env = a_np.reshape(-1)

            obs, r, term, trunc, _ = env.step(action_for_env)
            ret += float(r)
            done = bool(term) or bool(trunc)

            fr = _render_frame(env)
            blank_frames += 1 if _is_blank_frame(fr) else 0
            frames.append(_add_label(fr, label0, score=ret))

        if not frames:
            raise RuntimeError("No frames captured.")

        blank_ratio = float(blank_frames) / float(len(frames))
        if len(frames) >= 10 and blank_ratio > 0.9:
            raise RuntimeError(f"Render appears blank (blank_ratio={blank_ratio:.2f}).")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(out_path), frames, fps=int(fps))
    finally:
        env.close()
