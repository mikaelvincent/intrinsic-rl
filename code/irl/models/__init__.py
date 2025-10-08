"""Model definitions for PPO backbone.

Provides ready-to-use MLP-based Policy and Value networks. CNN/image encoders for CarRacing arrive in a later sprint.
"""

from __future__ import annotations

from .networks import PolicyNetwork, ValueNetwork

__all__ = ["PolicyNetwork", "ValueNetwork"]
