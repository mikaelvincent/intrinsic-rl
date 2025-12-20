from __future__ import annotations

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
