"""The Agent interface and a baseline RandomAgent.

An agent chooses an action from an observation and a legal-action mask. It is fully
decoupled from the engine and never touches game state directly, which makes agents
hot-swappable (policy / scripted / human) in the Runner.
"""

from __future__ import annotations

from typing import Generic, Protocol

import numpy as np

from .types import ActionMask, ActionT, Observation


class Agent(Protocol[Observation, ActionT]):
    """Anything that selects an action given what it can see."""

    def act(self, observation: Observation, mask: ActionMask) -> ActionT:
        """Return a legal action. Implementations must respect ``mask``."""
        ...


class RandomAgent(Generic[Observation]):
    """Picks uniformly among legal actions. Seeded for reproducibility.

    Returns the integer index of the chosen action, which matches the mask/encoder
    contract (e.g. Connect Four column index). A useful baseline opponent and a way
    to exercise the engine before any policy exists.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: Observation, mask: ActionMask) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise ValueError("RandomAgent.act called with no legal actions")
        return int(self._rng.choice(legal))
