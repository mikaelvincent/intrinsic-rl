"""PPO backbone models (MLP policy/value).

See devspec §5.1.
"""

from __future__ import annotations

from .networks import PolicyNetwork, ValueNetwork
from .layers import mlp, FlattenObs  # convenience re-exports
from .cnn import ConvEncoder, ConvEncoderConfig

__all__ = ["PolicyNetwork", "ValueNetwork", "mlp", "FlattenObs", "ConvEncoder", "ConvEncoderConfig"]
