from __future__ import annotations

import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import register

from irl.trainer.metrics_schema import SCALARS_REQUIRED_COMMON_COLS
from irl.utils.checkpoint import compute_cfg_hash, load_checkpoint


class _DummySuiteEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._t = 0
        self._rng = np.random.default_rng(0)
        self._last_action = 0

    def reset(self, *, seed=None, options=None):
        _ = options
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._t = 0
        self._last_action = 0
        obs = np.array([0.0, 0.0], dtype=np.float32)
        return obs, {}

    def step(self, action):
        self._t += 1
        self._last_action = int(action)
        obs = np.array([float(self._t), float(self._last_action)], dtype=np.float32)
        reward = float(self._last_action)
        terminated = self._t >= 3
        return obs, reward, bool(terminated), False, {}

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        v = int((self._t + 1) * 32) % 255
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = v
        frame[:, :, 1] = int(self._last_action) * 127
        frame[:, :, 2] = 64
        return frame

    def close(self) -> None:
        return


try:
    register(id="DummySuite-v0", entry_point=_DummySuiteEnv)
except Exception:
    pass


def _csv_header(path: Path) -> list[str]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        hdr = next(r)
    return [str(c).strip() for c in hdr]


def test_default_paper_pipeline_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    configs_dir = Path("configs")
    configs_dir.mkdir(parents=True, exist_ok=True)

    cfg_text = """
seed: [1]
device: cpu
method: vanilla

env:
  id: DummySuite-v0
  vec_envs: 2
  frame_skip: 1
  domain_randomization: false
  discrete_actions: true
  async_vector: false

ppo:
  steps_per_update: 2
  minibatches: 1
  epochs: 1
  learning_rate: 1e-3
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  entropy_coef: 0.0
  value_coef: 0.5
  value_clip_range: 0.0
  kl_penalty_coef: 0.0
  kl_stop: 0.0

intrinsic:
  eta: 0.0
  r_clip: 5.0

logging:
  csv_interval: 1
  checkpoint_interval: 100000
  checkpoint_max_to_keep: null

evaluation:
  interval_steps: 0
  episodes: 1

adaptation:
  enabled: false

exp:
  deterministic: true
  total_steps: 8
  profile_cuda_sync: false
""".lstrip()

    cfg_path = configs_dir / "vanilla_dummy.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    import irl.experiments.videos as videos_mod

    def _fake_render_rollout_video(
        *,
        ckpt_path: Path,
        out_path: Path,
        seed: int = 42,
        max_steps: int = 1000,
        device: str = "cpu",
        policy_mode: str = "mode",
        fps: int = 30,
    ) -> None:
        _ = ckpt_path, seed, max_steps, device, policy_mode, fps
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"mp4")

    monkeypatch.setattr(videos_mod, "render_rollout_video", _fake_render_rollout_video)

    import irl.benchmarks.suite as bench_suite

    def _fake_run_all_benchmarks(
        *,
        device: str = "cpu",
        threads: int = 1,
        seed: int = 0,
        out_dir: Path = Path("results/benchmarks"),
        quick: bool = True,
    ) -> dict[str, object]:
        _ = threads, seed, quick
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        outputs = {
            "latest_json": str((out_dir / "bench_latest.json").resolve()),
            "run_json": str((out_dir / "bench_00000000-000000.json").resolve()),
            "latest_csv": str((out_dir / "bench_latest.csv").resolve()),
            "run_csv": str((out_dir / "bench_00000000-000000.csv").resolve()),
        }

        results = [
            {
                "name": "glpe.gate_median_cache.baseline",
                "metric": "transitions_per_s",
                "unit": "transitions/s",
                "params": {"cache_interval": 1},
                "values": [100.0, 110.0, 105.0],
                "durations_s": [0.1, 0.1, 0.1],
            },
            {
                "name": "glpe.gate_median_cache.cached",
                "metric": "transitions_per_s",
                "unit": "transitions/s",
                "params": {"cache_interval": 64},
                "values": [200.0, 210.0, 205.0],
                "durations_s": [0.1, 0.1, 0.1],
            },
        ]

        payload = {
            "schema_version": 1,
            "run": {"device": str(device)},
            "total_time_s": 0.0,
            "results": results,
            "outputs": outputs,
        }

        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        Path(outputs["run_json"]).write_text(text, encoding="utf-8")
        Path(outputs["latest_json"]).write_text(text, encoding="utf-8")

        csv_text = "name,metric,unit,trials,params_json,error\n"
        csv_text += (
            "glpe.gate_median_cache.baseline,transitions_per_s,transitions/s,3,"
            + '"{\\"cache_interval\\": 1}",\n'
        )
        Path(outputs["run_csv"]).write_text(csv_text, encoding="utf-8")
        Path(outputs["latest_csv"]).write_text(csv_text, encoding="utf-8")

        return payload

    monkeypatch.setattr(bench_suite, "run_all_benchmarks", _fake_run_all_benchmarks)

    from irl.benchmarks.cli import cli_bench
    from irl.experiments import cli_eval, cli_plots, cli_train, cli_videos

    cli_train()
    cli_eval()
    cli_plots()
    cli_videos()
    cli_bench()

    runs_root = Path("runs_suite")
    run_dir = runs_root / "vanilla__DummySuite-v0__seed1__vanilla_dummy"
    assert run_dir.exists()

    scalars = run_dir / "logs" / "scalars.csv"
    assert scalars.exists()
    cols = set(_csv_header(scalars))
    assert set(SCALARS_REQUIRED_COMMON_COLS) <= cols

    ckpt_latest = run_dir / "checkpoints" / "ckpt_latest.pt"
    assert ckpt_latest.exists()
    payload = load_checkpoint(ckpt_latest, map_location="cpu")
    assert "cfg" in payload and "cfg_hash" in payload
    assert str(payload["cfg_hash"]) == compute_cfg_hash(payload["cfg"])

    cfg_hash_txt = run_dir / "config_hash.txt"
    assert cfg_hash_txt.exists()
    assert cfg_hash_txt.read_text(encoding="utf-8").strip() == str(payload["cfg_hash"])

    results_root = Path("results_suite")
    assert (results_root / "summary.csv").exists()

    plots_root = results_root / "plots"
    assert (plots_root / "DummySuite-v0__paper_baselines.png").exists()

    mp4s = list((results_root / "videos").rglob("*.mp4"))
    assert mp4s

    bench_dir = Path("results") / "benchmarks"
    assert (bench_dir / "bench_latest.json").exists()
    assert (bench_dir / "bench_latest.csv").exists()
    assert (bench_dir / "bench_latest_throughput.png").exists()
    assert (bench_dir / "bench_latest_speedup.png").exists()
