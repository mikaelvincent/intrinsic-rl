from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from irl.envs import EnvManager
from irl.models import PolicyNetwork
from irl.trainer.build import single_spaces
from irl.utils.checkpoint import load_checkpoint
from irl.utils.determinism import seed_everything


def _load_policy_for_eval(ckpt_path: Path, device: str) -> tuple[PolicyNetwork, dict]:
    payload = load_checkpoint(ckpt_path, map_location=device)
    cfg = payload.get("cfg", {}) or {}
    env_id = str((cfg.get("env") or {}).get("id", "MountainCar-v0"))

    manager = EnvManager(env_id=env_id, num_envs=1)
    env = manager.make()
    try:
        obs_space, act_space = single_spaces(env)
    finally:
        env.close()

    policy = PolicyNetwork(obs_space, act_space).to(device)
    policy.load_state_dict(payload["policy"])
    policy.eval()
    return policy, cfg


def _render_frame(env: gym.Env) -> np.ndarray:
    try:
        rgb = env.render()
    except Exception:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    if rgb is None:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    if isinstance(rgb, list):
        rgb = np.array(rgb)
    return rgb.astype(np.uint8)


def _add_label(
    frame: np.ndarray,
    label: str,
    score: float | None = None,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    text = label if score is None else f"{label}\nReturn: {score:.1f}"
    x, y = 10, 10

    for ox, oy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw.text((x + ox, y + oy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=color)

    return np.array(pil_img)


def render_side_by_side(
    env_id: str,
    ckpt_left: Path,
    ckpt_right: Path,
    out_path: Path,
    max_steps: int = 1000,
    seed: int = 42,
    device: str = "cpu",
    label_left: str = "Baseline",
    label_right: str = "Gated Learning-Progress Exploration (GLPE)",
) -> None:
    seed_everything(seed, deterministic=True)

    pol_l, cfg_l = _load_policy_for_eval(ckpt_left, device)
    pol_r, _ = _load_policy_for_eval(ckpt_right, device)
    env_cfg = cfg_l.get("env", {})

    def make_env(rank: int):
        return EnvManager(
            env_id=env_id,
            num_envs=1,
            seed=seed + rank,
            frame_skip=int(env_cfg.get("frame_skip", 1)),
            discrete_actions=bool(env_cfg.get("discrete_actions", True)),
            car_action_set=env_cfg.get("car_discrete_action_set", None),
            render_mode="rgb_array",
        ).make()

    env_l = make_env(0)
    env_l.reset(seed=seed)

    env_r = make_env(0)
    env_r.reset(seed=seed)

    obs_l, _ = env_l.reset(seed=seed)
    obs_r, _ = env_r.reset(seed=seed)

    done_l = False
    done_r = False
    ret_l = 0.0
    ret_r = 0.0
    frames: list[np.ndarray] = []

    for _ in range(int(max_steps)):
        if done_l and done_r:
            break

        if not done_l:
            with torch.no_grad():
                a_t, _ = pol_l.act(obs_l)
                act_l = a_t.cpu().numpy()

            act_l_env = int(act_l.item()) if env_l.action_space.shape == () else act_l.reshape(-1)
            obs_l, r, term, trunc, _ = env_l.step(act_l_env)
            ret_l += float(r)
            done_l = bool(term) or bool(trunc)
            frame_l = _render_frame(env_l)
        else:
            if "frame_l" not in locals():
                frame_l = _render_frame(env_l)

        if not done_r:
            with torch.no_grad():
                a_t, _ = pol_r.act(obs_r)
                act_r = a_t.cpu().numpy()

            act_r_env = int(act_r.item()) if env_r.action_space.shape == () else act_r.reshape(-1)
            obs_r, r, term, trunc, _ = env_r.step(act_r_env)
            ret_r += float(r)
            done_r = bool(term) or bool(trunc)
            frame_r = _render_frame(env_r)
        else:
            if "frame_r" not in locals():
                frame_r = _render_frame(env_r)

        labeled_l = _add_label(frame_l, label_left, ret_l, color=(200, 200, 255))
        labeled_r = _add_label(frame_r, label_right, ret_r, color=(255, 200, 200))

        h_l = int(labeled_l.shape[0])
        h_r = int(labeled_r.shape[0])
        w_r = int(labeled_r.shape[1])
        if h_l != h_r:
            pil_r = Image.fromarray(labeled_r).resize((int(w_r * h_l / h_r), h_l))
            labeled_r = np.array(pil_r)

        frames.append(np.concatenate([labeled_l, labeled_r], axis=1))

    env_l.close()
    env_r.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        imageio.mimsave(str(out_path), frames, fps=30)
