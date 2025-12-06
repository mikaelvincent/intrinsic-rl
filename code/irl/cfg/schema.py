"""Configuration dataclasses defining the public config surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ----- Section Schemas -------------------------------------------------------


@dataclass(frozen=True)
class GateConfig:
    """Region gating thresholds for the Proposed intrinsic.

    The global medians used by the gating rule (learning progress and error)
    only become active once enough regions have seen samples. This threshold is
    controlled by ``min_regions_for_gating`` (default: 3).

    Attributes
    ----------
    enabled:
        Master switch for gating. When ``False``, gating is disabled regardless
        of thresholds (equivalent to a "nogate" ablation).
    """

    enabled: bool = True  # Master on/off switch for gating
    tau_lp_mult: float = 0.01  # multiply median LP
    tau_s: float = 2.0
    hysteresis_up_mult: float = 2.0
    min_consec_to_gate: int = 5
    min_regions_for_gating: int = 3  # regions needed before using medians


@dataclass(frozen=True)
class IntrinsicConfig:
    """Intrinsic reward configuration shared by intrinsic methods.

    For the Proposed method, ``normalize_inside`` controls where intrinsic
    normalization happens. When ``False``, Proposed emits raw (unnormalized)
    component signals and relies on the trainer's global ``RunningRMS`` for
    scaling. When ``True`` (the default), Proposed performs per-component
    normalization internally and sets ``outputs_normalized=True`` so the
    trainer skips its own intrinsic normalization.
    """

    eta: float = 0.1
    alpha_impact: float = 1.0
    alpha_lp: float = 0.5
    r_clip: float = 5.0
    bin_size: float = 0.25
    region_capacity: int = 200
    depth_max: int = 12
    ema_beta_long: float = 0.995
    ema_beta_short: float = 0.90
    normalize_inside: bool = True  # Proposed-only internal normalization toggle
    gate: GateConfig = field(default_factory=GateConfig)


@dataclass(frozen=True)
class EnvConfig:
    """Environment settings.

    Notes
    -----
    For CarRacing environments (``id`` starting with ``"CarRacing"``) the
    ``discrete_actions`` flag enables the :class:`CarRacingDiscreteActionWrapper`.
    When ``car_discrete_action_set`` is not ``None``, it overrides the wrapper's
    default 5-action set with a custom list of ``[steer, gas, brake]`` triples.
    """

    id: str = "MountainCar-v0"
    vec_envs: int = 16
    frame_skip: int = 1
    domain_randomization: bool = False
    # For CarRacing, discrete by default; ignored for continuous-control envs.
    discrete_actions: bool = True
    # Optional explicit discrete action set for CarRacing. When provided, it
    # must be a sequence of [steer, gas, brake] triples and will be converted
    # to a NumPy array of shape (N, 3) by the environment manager.
    car_discrete_action_set: tuple[tuple[float, float, float], ...] | None = None


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
    # Value loss weighting and optional value-function clipping
    value_coef: float = 0.5
    value_clip_range: float = 0.0  # <= 0 disables value clipping
    # KL control (penalty and/or early stop); both disabled by default
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


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment-wide toggles.

    This section is for global behaviours that are not tied to a specific
    environment or PPO setting. For example, ``deterministic`` controls
    whether training should request deterministic PyTorch behaviour where
    supported. By default this is set to ``True`` so that runs are
    reproducible given a fixed seed, unless a config explicitly opts out.

    ``total_steps`` (optional) allows a configuration file to declare the
    target number of environment steps for training runs. When provided,
    higher-level orchestration (CLI/suite) should prefer this value over
    a global default passed via CLI flags.
    """

    deterministic: bool = True
    # Optional per-config target step budget (None => defer to CLI/defaults)
    total_steps: int | None = None


# ----- Root Schema -----------------------------------------------------------

MethodLiteral = Literal["vanilla", "icm", "rnd", "ride", "riac", "proposed"]


@dataclass(frozen=True)
class Config:
    """Top-level configuration shared by all CLI entry points."""

    seed: int = 1
    device: str = "cpu"  # or "cuda:0"
    method: MethodLiteral = "proposed"

    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    intrinsic: IntrinsicConfig = field(default_factory=IntrinsicConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    exp: ExperimentConfig = field(default_factory=ExperimentConfig)
