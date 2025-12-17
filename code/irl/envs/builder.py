from __future__ import annotations

from typing import Any

from .manager import EnvManager


def make_env(
    env_id: str,
    *,
    num_envs: int = 1,
    seed: int | None = 1,
    frame_skip: int = 1,
    domain_randomization: bool = False,
    discrete_actions: bool = True,
    car_action_set: object | None = None,
    render_mode: str | None = None,
    async_vector: bool = False,
    make_kwargs: dict[str, Any] | None = None,
) -> Any:
    manager = EnvManager(
        env_id=str(env_id),
        num_envs=int(num_envs),
        seed=None if seed is None else int(seed),
        frame_skip=int(frame_skip),
        domain_randomization=bool(domain_randomization),
        discrete_actions=bool(discrete_actions),
        car_action_set=car_action_set,
        render_mode=render_mode,
        async_vector=bool(async_vector),
        make_kwargs=None if make_kwargs is None else dict(make_kwargs),
    )
    return manager.make()
