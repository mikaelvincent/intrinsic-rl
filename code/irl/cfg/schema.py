from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

GLPE_FULL_NAME = "Gated Learning-Progress Exploration (GLPE)"


@dataclass(frozen=True)
class GateConfig:
    enabled: bool = True
    tau_lp_mult: float = 0.01
    tau_s: float = 2.0
    hysteresis_up_mult: float = 2.0
    min_consec_to_gate: int = 5
    min_regions_for_gating: int = 3


@dataclass(frozen=True)
class IntrinsicConfig:
    eta: float = 0.1
    alpha_impact: float = 1.0
    alpha_lp: float = 0.5
    r_clip: float = 5.0
    bin_size: float = 0.25
    region_capacity: int = 200
    depth_max: int = 12
    ema_beta_long: float = 0.995
    ema_beta_short: float = 0.90
    normalize_inside: bool = True
    fail_on_error: bool = True
    checkpoint_include_points: bool = True
    gate: GateConfig = field(default_factory=GateConfig)


@dataclass(frozen=True)
class EnvConfig:
    id: str = "MountainCar-v0"
    vec_envs: int = 16
    frame_skip: int = 1
    domain_randomization: bool = False
    discrete_actions: bool = True
    car_discrete_action_set: tuple[tuple[float, float, float], ...] | None = None
    async_vector: bool = False


@dataclass(frozen=True)
class PPOConfig:
    steps_per_update: int = 2048
    minibatches: int = 32
    epochs: int = 10
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    value_clip_range: float = 0.0
    kl_penalty_coef: float = 0.0
    kl_stop: float = 0.0


@dataclass(frozen=True)
class AdaptationConfig:
    enabled: bool = True
    interval_steps: int = 50_000
    entropy_low_frac: float = 0.3


@dataclass(frozen=True)
class EvaluationConfig:
    interval_steps: int = 50_000
    episodes: int = 10


@dataclass(frozen=True)
class LoggingConfig:
    csv_interval: int = 10_000
    checkpoint_interval: int = 100_000
    checkpoint_max_to_keep: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    deterministic: bool = True
    total_steps: int | None = None


MethodLiteral = Literal[
    "vanilla",
    "icm",
    "rnd",
    "ride",
    "riac",
    "glpe",
    "glpe_lp_only",
    "glpe_impact_only",
    "glpe_nogate",
]


@dataclass(frozen=True)
class Config:
    seed: int = 1
    device: str = "cpu"
    method: MethodLiteral = "glpe"

    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    intrinsic: IntrinsicConfig = field(default_factory=IntrinsicConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    exp: ExperimentConfig = field(default_factory=ExperimentConfig)
