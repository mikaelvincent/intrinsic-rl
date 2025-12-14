import numpy as np

from irl.trainer.runtime_utils import _apply_final_observation


def test_final_observation_substitution_for_truncation_like_done():
    """When a done env is auto-reset, use infos['final_observation'] for rollouts.

    This simulates a VectorEnv behavior where:
      - next_obs returned by env.step() is the *reset* observation (0.0)
      - the true terminal observation is provided in infos['final_observation']
    """
    # next_obs returned from env.step(): env0 auto-reset to 0.0, env1 continues to 1.0
    next_obs = np.array([[0.0], [1.0]], dtype=np.float32)

    # done mask (could be truncation OR termination); only env0 is done
    done = np.array([True, False], dtype=bool)

    # VectorEnv-style infos: final_observation provided only for done envs
    infos = {
        "final_observation": np.array(
            [np.array([42.0], dtype=np.float32), None],
            dtype=object,
        )
    }

    fixed = _apply_final_observation(next_obs, done, infos)

    assert fixed.shape == next_obs.shape
    # done env replaced with terminal observation
    assert np.isclose(float(fixed[0, 0]), 42.0)
    # non-done env unchanged
    assert np.isclose(float(fixed[1, 0]), 1.0)

    # Ensure we didn't mutate the input observation buffer
    assert np.isclose(float(next_obs[0, 0]), 0.0)


def test_final_observation_single_env_payload():
    """Single-env convenience: accept scalar/flat final_observation payloads."""
    next_obs = np.array([0.0], dtype=np.float32)
    done = np.array([True], dtype=bool)

    infos = {"final_observation": np.array([123.0], dtype=np.float32)}
    fixed = _apply_final_observation(next_obs, done, infos)

    assert fixed.shape == next_obs.shape
    assert np.isclose(float(fixed[0]), 123.0)
