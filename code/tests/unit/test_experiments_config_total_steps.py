from __future__ import annotations

from pathlib import Path

from irl.experiments import run_training_suite
from irl.utils.checkpoint import load_checkpoint


def _write_cfg_with_total_steps(configs_dir: Path, filename: str, total_steps: int) -> Path:
    """Create a minimal MountainCar config that sets exp.total_steps."""
    configs_dir.mkdir(parents=True, exist_ok=True)
    cfg_text = f"""
seed: 7
device: "cpu"
method: "vanilla"
env:
  id: "MountainCar-v0"
  vec_envs: 1
  frame_skip: 1
  domain_randomization: false
  discrete_actions: true
ppo:
  steps_per_update: 8
  minibatches: 2
  epochs: 1
  learning_rate: 3.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  entropy_coef: 0.0
intrinsic:
  eta: 0.0
adaptation:
  enabled: false
evaluation:
  interval_steps: 100000
  episodes: 1
logging:
  tb: false
  csv_interval: 1
  checkpoint_interval: 8
exp:
  deterministic: true
  total_steps: {int(total_steps)}
"""
    path = configs_dir / filename
    path.write_text(cfg_text.lstrip(), encoding="utf-8")
    return path


def test_suite_respects_config_total_steps_over_cli_default(tmp_path: Path) -> None:
    """run_training_suite should prefer cfg.exp.total_steps when present."""
    configs_dir = tmp_path / "configs"
    _write_cfg_with_total_steps(configs_dir, "mc_steps.yaml", total_steps=24)

    runs_root = tmp_path / "runs_suite"

    # Pass a larger CLI default; suite must pick 24 from the config instead.
    run_training_suite(
        configs_dir=configs_dir,
        include=[],
        exclude=[],
        total_steps=100,  # should be ignored in favour of cfg.exp.total_steps
        runs_root=runs_root,
        seeds=[1],
        device="cpu",
        resume=False,
    )

    # Verify checkpoint step equals config-defined total_steps (vec_envs=1 → exact).
    run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    ckpt_latest = run_dirs[0] / "checkpoints" / "ckpt_latest.pt"
    assert ckpt_latest.exists()

    payload = load_checkpoint(ckpt_latest, map_location="cpu")
    step = int(payload.get("step", -1))
    assert step == 24, f"expected step=24 from config, got {step}"
