import numpy as np

from irl.trainer.runtime_utils import _apply_final_observation


def test_final_observation_substitution_for_truncation_like_done():
    next_obs = np.array([[0.0], [1.0]], dtype=np.float32)
    done = np.array([True, False], dtype=bool)
    infos = {
        "final_observation": np.array([np.array([42.0], dtype=np.float32), None], dtype=object)
    }

    fixed = _apply_final_observation(next_obs, done, infos)

    assert fixed.shape == next_obs.shape
    assert np.isclose(float(fixed[0, 0]), 42.0)
    assert np.isclose(float(fixed[1, 0]), 1.0)
    assert np.isclose(float(next_obs[0, 0]), 0.0)


def test_final_observation_single_env_payload():
    next_obs = np.array([0.0], dtype=np.float32)
    done = np.array([True], dtype=bool)
    infos = {"final_observation": np.array([123.0], dtype=np.float32)}

    fixed = _apply_final_observation(next_obs, done, infos)

    assert fixed.shape == next_obs.shape
    assert np.isclose(float(fixed[0]), 123.0)
