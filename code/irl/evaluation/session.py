from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from irl.intrinsic.config import build_intrinsic_kwargs
from irl.intrinsic.factory import create_intrinsic_module
from irl.models import PolicyNetwork
from irl.pipelines.runtime import build_obs_normalizer, extract_env_runtime
from irl.trainer.build import single_spaces
from irl.utils.spaces import is_image_space

NormalizeFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class EvalSession:
    env: Any
    obs_space: Any
    act_space: Any
    policy: Any
    intrinsic_module: Any | None
    is_image: bool
    normalize_obs: NormalizeFn


def _normalize_factory(norm: tuple[np.ndarray, np.ndarray] | None) -> NormalizeFn:
    if norm is None:

        def _id(x: np.ndarray) -> np.ndarray:
            return x

        return _id

    mean_arr, std_arr = norm

    def _normalize(x: np.ndarray) -> np.ndarray:
        return (x - mean_arr) / std_arr

    return _normalize


def build_eval_session(
    *,
    env_id: str,
    cfg: object,
    payload: Mapping[str, Any],
    device: str,
    seed_eval_base: int,
    save_traj: bool,
    make_env_fn: Any,
    policy_cls: Any = PolicyNetwork,
) -> EvalSession:
    runtime = extract_env_runtime(cfg)
    frame_skip = int(runtime["frame_skip"])
    discrete_actions = bool(runtime["discrete_actions"])
    car_action_set = runtime["car_action_set"]

    env = make_env_fn(
        env_id=env_id,
        num_envs=1,
        seed=int(seed_eval_base),
        frame_skip=int(frame_skip),
        domain_randomization=False,
        discrete_actions=bool(discrete_actions),
        car_action_set=car_action_set,
    )
    obs_space, act_space = single_spaces(env)

    policy = policy_cls(obs_space, act_space).to(device)
    policy.load_state_dict(payload["policy"])
    policy.eval()

    intrinsic_module = None
    method = str(cfg.get("method", "vanilla"))
    if save_traj and "intrinsic" in payload:
        try:
            intr_state = payload["intrinsic"]
            intrinsic_module = create_intrinsic_module(
                method,
                obs_space,
                act_space,
                device=device,
                **build_intrinsic_kwargs(cfg),
            )
            if isinstance(intr_state, dict) and "state_dict" in intr_state:
                intrinsic_module.load_state_dict(intr_state["state_dict"])
            intrinsic_module.eval()
        except Exception:
            intrinsic_module = None

    is_img = bool(is_image_space(obs_space))
    norm = None if is_img else build_obs_normalizer(payload)
    normalize_obs = _normalize_factory(norm)

    return EvalSession(
        env=env,
        obs_space=obs_space,
        act_space=act_space,
        policy=policy,
        intrinsic_module=intrinsic_module,
        is_image=is_img,
        normalize_obs=normalize_obs,
    )
