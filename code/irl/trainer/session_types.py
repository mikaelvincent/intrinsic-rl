from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import Adam

from irl.intrinsic import RunningRMS
from irl.models import PolicyNetwork, ValueNetwork
from irl.utils.checkpoint import CheckpointManager
from irl.utils.loggers import MetricLogger

from .obs_norm import RunningObsNorm


@dataclass
class PPOOptimizers:
    policy: Adam
    value: Adam


@dataclass
class IntrinsicContext:
    module: Optional[Any]
    method: str
    eta: float
    use_intrinsic: bool
    norm_mode: str
    outputs_normalized: Optional[bool]
    rms: RunningRMS


@dataclass
class TrainingSession:
    run_dir: Path
    device: torch.device

    env: Any
    obs_space: Any
    act_space: Any
    is_image: bool
    num_envs: int

    policy: PolicyNetwork
    value: ValueNetwork
    optimizers: PPOOptimizers

    intrinsic: IntrinsicContext
    obs_norm: Optional[RunningObsNorm]

    ckpt: CheckpointManager
    metric_logger: MetricLogger

    obs: Any
    global_step: int
    update_idx: int
