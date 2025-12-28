from __future__ import annotations

import csv
import hashlib
import json
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.envs.registration import register

from irl.evaluation.rollout import run_eval_episodes
from irl.experiments.evaluation import run_eval_suite
from irl.intrinsic.icm import ICMConfig
from irl.intrinsic.riac import RIAC
from irl.models.networks import PolicyNetwork


class _DummyEvalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        _ = options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(4,)).astype(np.float32)
        return obs, {}

    def step(self, action):
        self._t += 1
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(4,)).astype(np.float32)
        terminated = self._t >= 3
        return obs, float(action), bool(terminated), False, {}

    def close(self) -> None:
        return


try:
    register(id="DummyEval-v0", entry_point=_DummyEvalEnv)
except Exception:
    pass


def _write_latest_ckpt(
    run_dir: Path,
    *,
    env_id: str,
    method: str,
    seed: int,
    step: int,
    eval_interval_steps: int = 50_000,
    episodes: int = 1,
) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "cfg": {
            "env": {"id": str(env_id)},
            "method": str(method),
            "seed": int(seed),
            "evaluation": {"interval_steps": int(eval_interval_steps), "episodes": int(episodes)},
        },
    }
    torch.save(payload, ckpt_dir / "ckpt_latest.pt")


def _write_step_ckpt(
    run_dir: Path,
    *,
    env_id: str,
    method: str,
    seed: int,
    step: int,
    eval_interval_steps: int = 50_000,
    episodes: int = 1,
) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "cfg": {
            "env": {"id": str(env_id)},
            "method": str(method),
            "seed": int(seed),
            "evaluation": {"interval_steps": int(eval_interval_steps), "episodes": int(episodes)},
        },
    }
    ckpt_path = ckpt_dir / f"ckpt_step_{int(step)}.pt"
    torch.save(payload, ckpt_path)
    return ckpt_path


def _fake_evaluate(*, env: str, ckpt: Path, episodes: int, device: str, **kwargs) -> dict:
    _ = ckpt, device, kwargs
    return {
        "env_id": str(env),
        "episodes": int(episodes),
        "seed": 0,
        "mean_return": 1.0,
        "std_return": 0.0,
        "min_return": 1.0,
        "max_return": 1.0,
        "mean_length": 5.0,
        "std_length": 0.0,
        "returns": [1.0],
        "lengths": [5],
    }


def test_eval_suite_enforces_coverage_and_selects_ckpts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import irl.experiments.evaluation as eval_module

    monkeypatch.setattr(eval_module, "evaluate", _fake_evaluate)

    runs_root = tmp_path / "runs_suite_seed"
    results_dir = tmp_path / "results_suite_seed"
    group = runs_root / "groupA"
    group.mkdir(parents=True, exist_ok=True)

    r_v1 = group / "vanilla__DummyEval-v0__seed1__cfgA"
    r_v2 = group / "vanilla__DummyEval-v0__seed2__cfgA"
    r_g1 = group / "glpe__DummyEval-v0__seed1__cfgA"
    for rd in (r_v1, r_v2, r_g1):
        rd.mkdir(parents=True, exist_ok=True)

    _write_latest_ckpt(r_v1, env_id="DummyEval-v0", method="vanilla", seed=1, step=100)
    _write_latest_ckpt(r_v2, env_id="DummyEval-v0", method="vanilla", seed=2, step=100)
    _write_latest_ckpt(r_g1, env_id="DummyEval-v0", method="glpe", seed=1, step=50)

    with pytest.raises(RuntimeError, match="Seed coverage mismatch"):
        run_eval_suite(runs_root=runs_root, results_dir=results_dir)

    cov_path = results_dir / "coverage.csv"
    with cov_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    glpe_rows = [r for r in rows if r["env_id"] == "DummyEval-v0" and r["method"] == "glpe"]
    assert len(glpe_rows) == 1
    assert glpe_rows[0]["missing_seeds"] == "2"

    runs_root2 = tmp_path / "runs_suite_steps"
    results_dir2 = tmp_path / "results_suite_steps"
    runs_root2.mkdir(parents=True, exist_ok=True)

    r_v = runs_root2 / "vanilla__DummyEval-v0__seed1__cfgA"
    r_g = runs_root2 / "glpe__DummyEval-v0__seed1__cfgA"
    for rd in (r_v, r_g):
        rd.mkdir(parents=True, exist_ok=True)

    _write_latest_ckpt(r_v, env_id="DummyEval-v0", method="vanilla", seed=1, step=100_000)
    _write_latest_ckpt(r_g, env_id="DummyEval-v0", method="glpe", seed=1, step=1_000)

    with pytest.raises(RuntimeError, match="Step parity mismatch"):
        run_eval_suite(runs_root=runs_root2, results_dir=results_dir2)

    runs_root3 = tmp_path / "runs_suite_every_k"
    results_dir3 = tmp_path / "results_suite_every_k"
    runs_root3.mkdir(parents=True, exist_ok=True)

    run_dir = runs_root3 / "vanilla__DummyEval-v0__seed1__cfgA"
    run_dir.mkdir(parents=True, exist_ok=True)

    for step in (0, 10, 20, 30):
        _write_step_ckpt(
            run_dir,
            env_id="DummyEval-v0",
            method="vanilla",
            seed=1,
            step=step,
            eval_interval_steps=25,
            episodes=1,
        )

    run_eval_suite(runs_root=runs_root3, results_dir=results_dir3)

    raw_path = results_dir3 / "summary_raw.csv"
    with raw_path.open("r", newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))
    steps = sorted({int(r["ckpt_step"]) for r in raw_rows})
    assert steps == [0, 20, 30]

    meta_path = results_dir3 / "eval_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta.get("selected_ckpt_steps_union") == [0, 20, 30]


class _TinyEvalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self._t = 0
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed=None, options=None):
        _ = options
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._t = 0
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(4,)).astype(np.float32)
        return obs, {}

    def step(self, action):
        _ = action
        self._t += 1
        obs = self._rng.uniform(low=-1.0, high=1.0, size=(4,)).astype(np.float32)
        terminated = self._t >= 3
        return obs, 0.0, bool(terminated), False, {}

    def close(self) -> None:
        return


def _module_digest(mod: torch.nn.Module) -> str:
    sd = mod.state_dict()
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        h.update(str(k).encode("utf-8"))
        v = sd[k]
        if torch.is_tensor(v):
            t = v.detach().cpu()
            h.update(str(tuple(t.shape)).encode("utf-8"))
            h.update(str(t.dtype).encode("utf-8"))
            h.update(t.numpy().tobytes())
        else:
            h.update(pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL))
    return h.hexdigest()


def test_riac_eval_trajectory_does_not_mutate_state() -> None:
    torch.manual_seed(0)

    env = _TinyEvalEnv()
    try:
        obs_space = env.observation_space
        act_space = env.action_space
        device = torch.device("cpu")

        policy = PolicyNetwork(obs_space, act_space).to(device)
        policy.eval()

        riac = RIAC(
            obs_space,
            act_space,
            device="cpu",
            icm_cfg=ICMConfig(phi_dim=8, hidden=(16, 16)),
            region_capacity=64,
            depth_max=4,
            alpha_lp=0.5,
            checkpoint_include_points=True,
        )
        riac.eval()

        rng = np.random.default_rng(0)
        obs = rng.standard_normal((32, 4)).astype(np.float32)
        next_obs = rng.standard_normal((32, 4)).astype(np.float32)
        actions = rng.integers(0, int(act_space.n), size=(32,), endpoint=False, dtype=np.int64)
        _ = riac.compute_batch(obs, next_obs, actions, reduction="none")

        before = _module_digest(riac)

        def _norm(x: np.ndarray) -> np.ndarray:
            return np.asarray(x, dtype=np.float32)

        rr = run_eval_episodes(
            env=env,
            policy=policy,
            act_space=act_space,
            device=device,
            policy_mode="mode",
            episode_seeds=[123],
            normalize_obs=_norm,
            save_traj=True,
            is_image=False,
            intrinsic_module=riac,
            method="riac",
        )

        after = _module_digest(riac)
        assert before == after
        assert rr.trajectory is not None
        assert rr.trajectory.intrinsic_semantics == "frozen_checkpoint"
    finally:
        env.close()
