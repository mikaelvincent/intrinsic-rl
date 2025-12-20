from __future__ import annotations

import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.envs.registration import register

from irl.evaluator import evaluate
from irl.experiments.evaluation import run_eval_suite
from irl.intrinsic.config import build_intrinsic_kwargs
from irl.intrinsic.factory import create_intrinsic_module
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
        reward = float(action)
        terminated = self._t >= 5
        return obs, reward, bool(terminated), False, {}

    def close(self) -> None:
        return


class _DummyTrajEnv(gym.Env):
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
        reward = float(action)
        terminated = self._t >= 3
        return obs, reward, bool(terminated), False, {}

    def close(self) -> None:
        return


for _id, _cls in (("DummyEval-v0", _DummyEvalEnv), ("DummyTraj-v0", _DummyTrajEnv)):
    try:
        register(id=_id, entry_point=_cls)
    except Exception:
        pass


def test_evaluator_is_repeatable_with_fixed_seed(tmp_path: Path) -> None:
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    act_space = gym.spaces.Discrete(2)

    torch.manual_seed(0)
    policy = PolicyNetwork(obs_space, act_space)

    ckpt_path = tmp_path / "ckpt_eval_repeatable.pt"
    torch.save(
        {
            "step": 0,
            "policy": policy.state_dict(),
            "cfg": {"env": {"id": "DummyEval-v0"}, "seed": 321},
            "obs_norm": None,
        },
        ckpt_path,
    )

    s1 = evaluate(env="DummyEval-v0", ckpt=ckpt_path, episodes=2, device="cpu")
    s2 = evaluate(env="DummyEval-v0", ckpt=ckpt_path, episodes=2, device="cpu")
    assert s1["returns"] == s2["returns"]
    assert s1["lengths"] == s2["lengths"]


def test_evaluator_writes_trajectory_npz_and_gate_source(tmp_path: Path) -> None:
    def _write_ckpt(*, method: str, seed: int, include_intrinsic: bool) -> Path:
        obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        act_space = gym.spaces.Discrete(2)

        torch.manual_seed(0)
        policy = PolicyNetwork(obs_space, act_space)

        cfg = {
            "seed": int(seed),
            "method": str(method),
            "env": {
                "id": "DummyTraj-v0",
                "frame_skip": 1,
                "discrete_actions": True,
                "car_discrete_action_set": None,
            },
        }

        payload: dict[str, object] = {
            "step": 0,
            "policy": policy.state_dict(),
            "cfg": cfg,
            "obs_norm": None,
        }

        if include_intrinsic:
            mod = create_intrinsic_module(
                str(method),
                obs_space,
                act_space,
                device="cpu",
                **build_intrinsic_kwargs(cfg),
            )
            payload["intrinsic"] = {"method": str(method), "state_dict": mod.state_dict()}

        ckpt_path = tmp_path / f"ckpt_{method}{'_intr' if include_intrinsic else ''}.pt"
        torch.save(payload, ckpt_path)
        return ckpt_path

    out_v = tmp_path / "vanilla_out"
    ckpt_v = _write_ckpt(method="vanilla", seed=123, include_intrinsic=False)
    _ = evaluate(
        env="DummyTraj-v0",
        ckpt=ckpt_v,
        episodes=1,
        device="cpu",
        save_traj=True,
        traj_out_dir=out_v,
        policy_mode="mode",
    )
    traj_v = out_v / "DummyTraj-v0_trajectory.npz"
    assert traj_v.exists()

    d_v = np.load(traj_v, allow_pickle=False)
    assert set(d_v.files) == {
        "obs",
        "rewards_ext",
        "gates",
        "intrinsic",
        "env_id",
        "method",
        "gate_source",
    }
    assert str(d_v["gate_source"].reshape(-1)[0]) == "n/a"

    out_g = tmp_path / "glpe_out"
    ckpt_g = _write_ckpt(method="glpe", seed=7, include_intrinsic=True)
    _ = evaluate(
        env="DummyTraj-v0",
        ckpt=ckpt_g,
        episodes=1,
        device="cpu",
        save_traj=True,
        traj_out_dir=out_g,
        policy_mode="mode",
    )
    traj_g = out_g / "DummyTraj-v0_trajectory.npz"
    assert traj_g.exists()

    d_g = np.load(traj_g, allow_pickle=False)
    assert str(d_g["gate_source"].reshape(-1)[0]) == "checkpoint"

    out_g_missing = tmp_path / "glpe_missing_intrinsic_out"
    ckpt_g_missing = _write_ckpt(method="glpe", seed=9, include_intrinsic=False)
    _ = evaluate(
        env="DummyTraj-v0",
        ckpt=ckpt_g_missing,
        episodes=1,
        device="cpu",
        save_traj=True,
        traj_out_dir=out_g_missing,
        policy_mode="mode",
    )
    traj_g_missing = out_g_missing / "DummyTraj-v0_trajectory.npz"
    assert traj_g_missing.exists()

    d_g_missing = np.load(traj_g_missing, allow_pickle=False)
    assert str(d_g_missing["gate_source"].reshape(-1)[0]) == "missing_intrinsic"


def _write_latest_ckpt(run_dir: Path, *, env_id: str, method: str, seed: int, step: int) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "cfg": {"env": {"id": str(env_id)}, "method": str(method), "seed": int(seed)},
    }
    torch.save(payload, ckpt_dir / "ckpt_latest.pt")


def _write_step_ckpt(run_dir: Path, *, env_id: str, method: str, seed: int, step: int) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "cfg": {"env": {"id": str(env_id)}, "method": str(method), "seed": int(seed)},
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


def test_eval_suite_reports_coverage_and_step_parity(
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
        run_eval_suite(
            runs_root=runs_root,
            results_dir=results_dir,
            episodes=1,
            device="cpu",
            policy_mode="mode",
        )

    cov_path = results_dir / "coverage.csv"
    assert cov_path.exists()
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
        run_eval_suite(
            runs_root=runs_root2,
            results_dir=results_dir2,
            episodes=1,
            device="cpu",
            policy_mode="mode",
        )


def test_eval_suite_fixed_step_selects_expected_ckpt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs_suite"
    results_dir = tmp_path / "results_suite"
    runs_root.mkdir(parents=True, exist_ok=True)

    run_dir = runs_root / "vanilla__DummyEval-v0__seed1__cfgA"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_step_ckpt(run_dir, env_id="DummyEval-v0", method="vanilla", seed=1, step=10)
    _write_step_ckpt(run_dir, env_id="DummyEval-v0", method="vanilla", seed=1, step=20)
    _write_step_ckpt(run_dir, env_id="DummyEval-v0", method="vanilla", seed=1, step=30)

    import irl.experiments.evaluation as eval_module

    monkeypatch.setattr(eval_module, "evaluate", _fake_evaluate)

    run_eval_suite(
        runs_root=runs_root,
        results_dir=results_dir,
        episodes=1,
        device="cpu",
        policy_mode="mode",
        strict_coverage=True,
        strict_step_parity=True,
        ckpt_policy="fixed_step",
        target_step=25,
    )

    raw_path = results_dir / "summary_raw.csv"
    assert raw_path.exists()

    with raw_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert int(rows[0]["ckpt_step"]) == 20
    assert rows[0]["ckpt_path"].endswith("ckpt_step_20.pt")
