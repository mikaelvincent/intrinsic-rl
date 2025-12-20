from __future__ import annotations

import torch


def glpe_gate_and_intrinsic_no_update(
    mod: object, obs_1d: torch.Tensor, next_obs_1d: torch.Tensor
) -> tuple[int, float] | None:
    from .rollout import glpe_gate_and_intrinsic_no_update as _impl

    return _impl(mod, obs_1d, next_obs_1d)
