import numpy as np
import gymnasium as gym
from gymnasium import spaces

from irl.envs.wrappers import DomainRandomizationWrapper


class _DummyMujocoLikeEnv(gym.Env):
    """Minimal MuJoCo-like env exposing model.opt.gravity and geom_friction.

    Used to validate that DomainRandomizationWrapper perturbs physics
    around a fixed baseline instead of drifting multiplicatively across resets.
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

        class _Model:
            pass

        class _Opt:
            pass

        self.model = _Model()
        self.model.opt = _Opt()
        # Non-zero baseline gravity so per-component ratios are well-defined.
        self.model.opt.gravity = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        # Simple friction table with at least one geom.
        self.model.geom_friction = np.ones((4, 3), dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.observation_space.sample()
        return obs, {}

    def step(self, action):
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


def test_domain_randomization_mujoco_stays_near_baseline():
    env = _DummyMujocoLikeEnv()
    try:
        wrapped = DomainRandomizationWrapper(env, seed=123)
        baseline = wrapped.unwrapped.model.opt.gravity.copy()

        # After many resets, gravity should always stay within the intended
        # ±5% band around the original baseline (no multiplicative drift).
        scales = []
        for _ in range(100):
            _, info = wrapped.reset()
            g = wrapped.unwrapped.model.opt.gravity
            ratio = g / baseline
            assert np.all(np.isfinite(ratio))
            # Each component should stay inside the configured [0.95, 1.05] range.
            assert np.all(ratio >= 0.95 - 1e-6)
            assert np.all(ratio <= 1.05 + 1e-6)
            scales.append(ratio.copy())

            # Diagnostics should be present and well-formed.
            assert isinstance(info, dict)
            assert "dr_applied" in info
            diag = info["dr_applied"]
            assert isinstance(diag, dict)
            # MuJoCo perturbations may be zero in degenerate cases but must be non-negative.
            assert diag.get("mujoco", 0) >= 0

        # As an additional sanity check, confirm we actually see some variation.
        unique_scales = {tuple(np.round(s, decimals=3).tolist()) for s in scales}
        assert len(unique_scales) > 1, "Domain randomization should vary gravity across resets"
    finally:
        env.close()
