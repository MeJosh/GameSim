"""The Encoder interface: game observation <-> tensor, plus mask passthrough.

One encoder per game. This is the only place that knows how a game's state maps to the
fixed-shape arrays a neural network wants, keeping that concern out of both the engine
and the learner.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from gamesim.core.types import ActionMask

# The observation is consumed by the encoder (contravariant).
_ObsContra = TypeVar("_ObsContra", contravariant=True)


class Encoder(Protocol[_ObsContra]):
    """Converts between a game's observation and a network-friendly tensor."""

    def encode(self, observation: _ObsContra) -> npt.NDArray[np.float32]:
        """Observation -> fixed-shape float tensor for the policy network."""
        ...

    def action_mask(self, observation: _ObsContra) -> ActionMask:
        """Legal-action mask aligned with the network's action head."""
        ...
