"""
Evaluate a saved policy deterministically (no intrinsic).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Any, List

import numpy as np
import typer
import torch

from irl.envs import EnvManager
from irl.models import PolicyNetwork
from irl.trainer.build import single_spaces  # use shared helper
from irl.utils.checkpoint import load_checkpoint
from irl.utils.determinism import seed_everything  # unified seeding helper
from irl.intrinsic.factory import create_intrinsic_module # factory for loading intrinsic modules

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")


def _is_image_space(space) -> bool:
    """Heuristic: Box with rank >= 2 is treated as image-like."""
    return hasattr(space, "shape") and len(space.shape) >= 2


def _build_normalizer(payload) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return (mean, std) if vector obs_norm was persisted; images return None."""
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
    traj_out_dir: Optional[Path] = None,
) -> dict:
    """Evaluate a saved policy deterministically (no intrinsic), returning stats.

    Parameters
    ----------
    env:
        Gymnasium environment id (for example "MountainCar-v0").
    ckpt:
        Path to a training checkpoint produced by the trainer.
    episodes:
        Number of episodes to run.
    device:
        Torch device string, such as "cpu" or "cuda:0".
    save_traj:
        If True, record observations and (if available) intrinsic stats/gates
        for visualization.
    traj_out_dir:
        Directory to save the trajectory .npz file if save_traj is True.
        Defaults to the parent directory of the checkpoint.

    Returns
    -------
    dict
        Dictionary with keys:

        * env_id
        * episodes
        * seed
        * checkpoint_step
        * aggregate statistics for returns and episode lengths
        * per-episode returns and lengths lists
    """
    payload = load_checkpoint(ckpt, map_location=device)
    cfg = payload.get("cfg", {}) or {}
    seed = int(cfg.get("seed", 1))
    step = int(payload.get("step", -1))

    # Apply consistent seeding so evaluation matches training determinism.
    seed_everything(seed, deterministic=True)

    # Derive env settings from the checkpoint to ensure parity with training.
    env_cfg = (cfg.get("env") or {}) if isinstance(cfg, dict) else {}
    frame_skip = int(env_cfg.get("frame_skip", 1))
    # For CarRacing, discrete vs Box(3,) matters — use the original training setting.
    discrete_actions = bool(env_cfg.get("discrete_actions", True))
    # Optional CarRacing discrete action set, if present in the original config.
    car_action_set = env_cfg.get("car_discrete_action_set", None)
    # Deterministic evaluation: disable domain randomization regardless of training value.
    domain_randomization = False

    # Build a single environment (no vectorisation) for evaluation
    manager = EnvManager(
        env_id=env,
        num_envs=1,
        seed=seed,
        frame_skip=frame_skip,
        domain_randomization=domain_randomization,
        discrete_actions=discrete_actions,
        car_action_set=car_action_set,
    )
    e = manager.make()
    obs_space, act_space = single_spaces(e)

    # Policy network (same architecture as training) and weights
    policy = PolicyNetwork(obs_space, act_space).to(device)
    policy.load_state_dict(payload["policy"])
    policy.eval()

    # Optional: Load intrinsic module to recover gating/value stats for visualization
    intrinsic_module = None
    method = str(cfg.get("method", "vanilla"))
    if save_traj and "intrinsic" in payload:
        try:
            intr_state = payload["intrinsic"]
            # Re-create using config params
            int_cfg = cfg.get("intrinsic", {})
            gate_cfg = int_cfg.get("gate", {})
            # We need to reconstruct with the same args as training
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
            if "state_dict" in intr_state:
                intrinsic_module.load_state_dict(intr_state["state_dict"])
            intrinsic_module.eval()
        except Exception:
            # Intrinsic loading is best-effort for visualization
            intrinsic_module = None

    # Observation normalization for vector observations if stored
    is_image = _is_image_space(obs_space)
    norm = None if is_image else _build_normalizer(payload)

    def _normalize(x: np.ndarray) -> np.ndarray:
        if norm is None:
            return x
        mean_arr, std_arr = norm
        return (x - mean_arr) / std_arr

    returns: list[float] = []
    lengths: list[int] = []

    # Trajectory storage
    traj_obs: list[np.ndarray] = []
    traj_gates: list[int] = []
    traj_int_vals: list[float] = []

    for _ in range(int(episodes)):
        obs, _ = e.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0

        # Trajectory recording: flush existing buffer for new episode if we separated them,
        # but here we'll just concat everything for a heatmap.
       
        while not done:
            x_raw = obs if isinstance(obs, np.ndarray) else np.asarray(obs, dtype=np.float32)
            x = _normalize(x_raw)

            with torch.no_grad():
                obs_t = torch.as_tensor(x, dtype=torch.float32, device=device)
                dist = policy.distribution(obs_t)
                act = dist.mode()
                a_np = act.detach().cpu().numpy()

            # Discrete or continuous action routing
            if hasattr(act_space, "n"):
                action_for_env = int(a_np.item())
            else:
                action_for_env = a_np.reshape(-1)

            next_obs, r, term, trunc, _ = e.step(action_for_env)
            ep_ret += float(r)
            ep_len += 1

            # --- Trajectory Capture ---
            if save_traj and not is_image:
                # Store raw observation for plotting (normalized might be weird to interpret)
                traj_obs.append(x_raw.copy())
                gate_val = 1
                int_val = 0.0
                if intrinsic_module is not None:
                    # We need to run a "dummy" compute to get the state/gate
                    # Prepare batch-dim inputs
                    batch_obs = obs_t.unsqueeze(0)
                    batch_next = torch.as_tensor(_normalize(next_obs), dtype=torch.float32, device=device).unsqueeze(0)
                    batch_act = act.unsqueeze(0)
                    try:
                        # Some modules (Proposed) update internal stats on compute.
                        # In eval we strictly shouldn't update, but for visualization we might need
                        # to query the *current* gate status for this point.
                        # Proposed.compute_batch does updates. Proposed.compute does updates.
                        # We'll accept slight state drift or just use the current value.
                        # Better approach: Proposed.compute returns IntrinsicOutput with r_int.
                        # But we want the gate.
                        # HACK: Just run compute_batch (reduction=none) and inspect internal stats if possible,
                        # or rely on the returned reward being 0 if gated.
                        # Ideally we'd have an inspect() method.
                        r_out = intrinsic_module.compute_batch(batch_obs, batch_next, batch_act, reduction="none")
                        int_val = float(r_out.mean().item())
                        # Infer gate from internal state if Proposed
                        if method == "proposed" and hasattr(intrinsic_module, "store"):
                            # We need phi to find the region
                            with torch.no_grad():
                                phi = intrinsic_module.icm._phi(batch_obs)
                                phi_np = phi.cpu().numpy().reshape(-1)
                                rid = intrinsic_module.store.locate(phi_np)
                                st = intrinsic_module._stats.get(rid)
                                if st:
                                    gate_val = st.gate
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

    # Save trajectories if requested and data exists
    if save_traj and traj_obs:
        traj_out_dir = traj_out_dir or ckpt.parent
        traj_out_dir.mkdir(parents=True, exist_ok=True)
        env_tag = env.replace("/", "-")
        traj_file = traj_out_dir / f"{env_tag}_trajectory.npz"
        np.savez_compressed(
            traj_file,
            obs=np.stack(traj_obs),
            gates=np.array(traj_gates),
            intrinsic=np.array(traj_int_vals),
        )

    e.close()
    return summary


@app.command("eval")
def cli_eval(
    env: str = typer.Option(..., "--env", "-e", help="Gymnasium env id (e.g., MountainCar-v0)."),
    ckpt: Path = typer.Option(
        ..., "--ckpt", "-k", help="Path to a training checkpoint file.", exists=True
    ),
    episodes: int = typer.Option(10, "--episodes", "-n", help="Number of episodes to evaluate."),
    device: str = typer.Option(
        "cpu", "--device", "-d", help='Torch device, e.g., "cpu" or "cuda:0".'
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Optional path to write aggregated results as JSON.",
        dir_okay=False,
    ),
    save_traj: bool = typer.Option(
        False, "--save-traj", help="Save obs/gates to .npz for visualization."
    ),
) -> None:
    """Run evaluation episodes using mode actions and report aggregate stats."""
    summary = evaluate(
        env=env, ckpt=ckpt, episodes=episodes, device=device, save_traj=save_traj
    )

    # Per-episode lines
    for i, (ret, length) in enumerate(zip(summary["returns"], summary["lengths"]), start=1):
        typer.echo(f"Episode {i}/{summary['episodes']}: return = {ret:.2f}, length = {length}")

    # Aggregate line
    typer.echo(
        f"\n[green]Eval complete[/green] — mean return {summary['mean_return']:.2f} "
        f"± {summary['std_return']:.2f} over {summary['episodes']} episodes"
    )

    # Optional JSON dump (atomic)
    if out is not None:
        text = json.dumps(summary, indent=2)
        from irl.utils.checkpoint import atomic_write_text  # lazy import (keeps module light)

        atomic_write_text(out, text)
        typer.echo(f"Saved summary to {out}")


def main() -> None:
    """Entry point for the ``irl-eval`` console script."""
    app()


if __name__ == "__main__":
    main()
