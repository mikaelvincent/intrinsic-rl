"""Video generation utilities for side-by-side agent comparisons.

Uses ``imageio`` to write MP4/GIFs and ``PIL`` for text overlays.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import imageio
import numpy as np
import torch
import gymnasium as gym
from PIL import Image, ImageDraw, ImageFont

from irl.envs import EnvManager
from irl.models import PolicyNetwork
from irl.utils.checkpoint import load_checkpoint
from irl.utils.determinism import seed_everything
from irl.trainer.build import single_spaces


def _load_policy_for_eval(ckpt_path: Path, device: str) -> tuple[PolicyNetwork, dict]:
    """Load a policy network and its config from a checkpoint."""
    payload = load_checkpoint(ckpt_path, map_location=device)
    cfg = payload.get("cfg", {}) or {}
    
    # Infer spaces from a dummy env to ensure architecture matches
    # We assume the env_id in the config is correct.
    env_id = str((cfg.get("env") or {}).get("id", "MountainCar-v0"))
    
    # Create dummy env just to get spaces
    manager = EnvManager(env_id=env_id, num_envs=1)
    env = manager.make()
    obs_space, act_space = single_spaces(env)
    env.close()

    policy = PolicyNetwork(obs_space, act_space).to(device)
    policy.load_state_dict(payload["policy"])
    policy.eval()
    
    return policy, cfg


def _render_frame(env: gym.Env) -> np.ndarray:
    """Capture a frame from the environment."""
    # Try different render modes if the env is finicky
    try:
        rgb = env.render()
    except Exception:
        return np.zeros((240, 320, 3), dtype=np.uint8)
        
    if rgb is None:
        # Fallback for headless systems that might return None despite render_mode
        return np.zeros((240, 320, 3), dtype=np.uint8)
        
    if isinstance(rgb, list):
        rgb = np.array(rgb)
        
    return rgb.astype(np.uint8)


def _add_label(
    frame: np.ndarray, 
    label: str, 
    score: float | None = None, 
    color: tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """Overlay text label and optional score on a frame."""
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    
    # Basic font handling - default bitmap font is always available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    # Draw Text with simple outline for readability
    text = label
    if score is not None:
        text += f"\nReturn: {score:.1f}"
        
    x, y = 10, 10
    
    # Outline (black)
    for off in [(-1,-1), (-1,1), (1,-1), (1,1)]:
        draw.text((x+off[0], y+off[1]), text, font=font, fill=(0,0,0))
    
    # Main text
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
    label_right: str = "Proposed",
) -> None:
    """Generate a split-screen video comparing two agents.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.
    ckpt_left : Path
        Checkpoint for the left side (e.g., Baseline/Vanilla).
    ckpt_right : Path
        Checkpoint for the right side (e.g., Proposed).
    out_path : Path
        Output video path (e.g. .mp4 or .gif).
    max_steps : int
        Maximum frames to record.
    seed : int
        Seed for environment reproducibility.
    device : str
        Torch device.
    label_left : str
        Label for the left agent.
    label_right : str
        Label for the right agent.
    """
    seed_everything(seed, deterministic=True)
    
    # Load policies
    pol_l, cfg_l = _load_policy_for_eval(ckpt_left, device)
    pol_r, cfg_r = _load_policy_for_eval(ckpt_right, device)
    
    # Config details needed for Env instantiation
    # Use left config as primary source for env settings (frame skip etc), assume compatible
    env_cfg = cfg_l.get("env", {})
    
    # Create two separate env instances
    # Force render_mode="rgb_array"
    def make_env(rank: int):
        return EnvManager(
            env_id=env_id,
            num_envs=1,
            seed=seed + rank, # slight offset or same? Prompt implies direct comparison, same seed is best.
            # Actually, using exactly the same seed allows checking divergence on identical start states.
            # However, if envs are not 100% deterministic across instances, they might drift.
            # Let's use SAME seed for fair "start state" comparison.
            frame_skip=int(env_cfg.get("frame_skip", 1)),
            discrete_actions=bool(env_cfg.get("discrete_actions", True)),
            car_action_set=env_cfg.get("car_discrete_action_set", None),
            render_mode="rgb_array",
        ).make()

    env_l = make_env(0)
    # Re-seed explicitly to ensure identical start
    env_l.reset(seed=seed) 
    
    env_r = make_env(0) 
    env_r.reset(seed=seed)

    # Reset both
    obs_l, _ = env_l.reset(seed=seed)
    obs_r, _ = env_r.reset(seed=seed)
    
    done_l = False
    done_r = False
    ret_l = 0.0
    ret_r = 0.0
    
    frames = []
    
    for _ in range(max_steps):
        if done_l and done_r:
            break
            
        # --- Left Step ---
        if not done_l:
            with torch.no_grad():
                # Preprocess obs
                # Policy expects standard input. If image env, ensure uint8 0-255 is passed if that's what training used.
                # EnvManager returns standard Gym obs.
                if isinstance(obs_l, np.ndarray):
                    # Ensure float conversion if network expects it, but networks.py handles it.
                    # Networks usually handle numpy -> tensor -> preprocess.
                    o_t = torch.as_tensor(obs_l, device=device) # type might be inferred
                    # But PolicyNetwork.act usually expects preprocessing inside or handled by caller?
                    # PolicyNetwork handles image preprocessing internally if rank >= 2 (ConvEncoder).
                    # Vector obs needs no special prep here besides to_tensor.
                    pass
                
                # Use act() to sample
                a_t, _ = pol_l.act(obs_l)
                act_l = a_t.cpu().numpy()
                
            # Handle discrete scalar vs box
            if env_l.action_space.shape == ():
                # Discrete
                act_l_env = int(act_l.item())
            else:
                act_l_env = act_l.reshape(-1)
                
            obs_l, r, term, trunc, _ = env_l.step(act_l_env)
            ret_l += float(r)
            done_l = term or trunc
            frame_l = _render_frame(env_l)
        else:
            # Hold last frame
            # We assume we have at least one frame captured previously
            # If done on step 0, capture one now
            if not 'frame_l' in locals():
                frame_l = _render_frame(env_l)
            pass # frame_l remains last frame

        # --- Right Step ---
        if not done_r:
            with torch.no_grad():
                a_t, _ = pol_r.act(obs_r)
                act_r = a_t.cpu().numpy()
            
            if env_r.action_space.shape == ():
                act_r_env = int(act_r.item())
            else:
                act_r_env = act_r.reshape(-1)
                
            obs_r, r, term, trunc, _ = env_r.step(act_r_env)
            ret_r += float(r)
            done_r = term or trunc
            frame_r = _render_frame(env_r)
        else:
            if not 'frame_r' in locals():
                frame_r = _render_frame(env_r)
            pass

        # --- Stitching ---
        # Resize to match height if needed? Usually same env = same size.
        # Add labels
        labeled_l = _add_label(frame_l, label_left, ret_l, color=(200, 200, 255))
        labeled_r = _add_label(frame_r, label_right, ret_r, color=(255, 200, 200))
        
        # Concatenate horizontally
        # Handle height mismatch defensively
        h_l, w_l, _ = labeled_l.shape
        h_r, w_r, _ = labeled_r.shape
        
        if h_l != h_r:
            # Resize right to match left height
            # Simple PIL resize
            pil_r = Image.fromarray(labeled_r).resize((int(w_r * h_l / h_r), h_l))
            labeled_r = np.array(pil_r)
            
        combined = np.concatenate([labeled_l, labeled_r], axis=1)
        frames.append(combined)

    env_l.close()
    env_r.close()
    
    # Save video
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        # 30 fps default
        imageio.mimsave(str(out_path), frames, fps=30)
