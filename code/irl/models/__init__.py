"""PPO backbone models (MLP policy/value).

Exports policy/value networks plus shared layers and the CNN encoder.
"""

from __future__ import annotations

from .networks import PolicyNetwork, ValueNetwork
from .layers import mlp, FlattenObs  # convenience re-exports
from .cnn import ConvEncoder, ConvEncoderConfig

__all__ = ["PolicyNetwork", "ValueNetwork", "mlp", "FlattenObs", "ConvEncoder", "ConvEncoderConfig"]
