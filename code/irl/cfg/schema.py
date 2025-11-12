"""Configuration dataclasses.

Defines the project's configuration surface; defaults mirror the provided
example YAML files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ----- Section Schemas -------------------------------------------------------


@dataclass(frozen=True)
class GateConfig:
    """Region gating thresholds (Proposed).

    Note
    ----
    The global medians used by the gating rule (LP and error) only become
    active once *enough* regions have seen samples. This threshold is
    controlled by `min_regions_for_gating` (default: 3).
    """

    tau_lp_mult: float = 0.01  # multiply median LP
    tau_s: float = 2.0
    hysteresis_up_mult: float = 2.0
    min_consec_to_gate: int = 5
    min_regions_for_gating: int = 3  # NEW: regions needed before using medians


@dataclass(frozen=True)
class IntrinsicConfig:
    """Intrinsic reward knobs (RIDE/R-IAC/Proposed, etc.)."""

    eta: float = 0.1
    alpha_impact: float = 1.0
    alpha_lp: float = 0.5
    r_clip: float = 5.0
    bin_size: float = 0.25
    region_capacity: int = 200
    depth_max: int = 12
    ema_beta_long: float = 0.995
    ema_beta_short: float = 0.90
    gate: GateConfig = field(default_factory=GateConfig)


@dataclass(frozen=True)
class EnvConfig:
    """Environment settings."""

    id: str = "MountainCar-v0"
    vec_envs: int = 16
    frame_skip: int = 1
    domain_randomization: bool = False
    # For CarRacing, discrete by default; ignored for continuous-control envs.
    discrete_actions: bool = True


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters."""

    steps_per_update: int = 2048
    minibatches: int = 32
    epochs: int = 10
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.0
    # NEW: value loss weighting and optional value function clipping
    value_coef: float = 0.5
    value_clip_range: float = 0.0  # <= 0 disables value clipping
    # NEW: KL control (penalty and/or early stop); both disabled by default
    kl_penalty_coef: float = 0.0  # add kl_penalty_coef * |approx_kl| to policy loss
    kl_stop: float = 0.0  # if |approx_kl| > kl_stop: early stop PPO epochs


@dataclass(frozen=True)
class AdaptationConfig:
    """Policy-aware α schedule (optional)."""

    enabled: bool = True
    interval_steps: int = 50_000
    entropy_low_frac: float = 0.3


@dataclass(frozen=True)
class EvaluationConfig:
    """Periodic evaluation (no intrinsic)."""

    interval_steps: int = 50_000
    episodes: int = 10


@dataclass(frozen=True)
class LoggingConfig:
    """Logging & checkpoint cadence."""

    tb: bool = True
    csv_interval: int = 10_000
    checkpoint_interval: int = 100_000


# ----- Root Schema -----------------------------------------------------------

MethodLiteral = Literal["vanilla", "icm", "rnd", "ride", "riac", "proposed"]


@dataclass(frozen=True)
class Config:
    """Top-level configuration spanning trainer, env, intrinsic, and logging."""

    seed: int = 1
    device: str = "cpu"  # or "cuda:0"
    method: MethodLiteral = "proposed"

    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    intrinsic: IntrinsicConfig = field(default_factory=IntrinsicConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
