"""Foundational type aliases used across the framework.

Kept deliberately small. Games specialize the generic ``Observation`` /action types
to their own concrete representations; the core only fixes the shapes it must.
"""

from __future__ import annotations

from typing import NewType, TypeVar

import numpy as np
import numpy.typing as npt

# A stable identifier for a player/seat. The core never hardcodes player counts;
# it addresses everyone by AgentId (see docs/adr/0002-n-agent-interface.md).
AgentId = NewType("AgentId", int)

# A boolean vector marking which actions are legal for the agent on turn.
# Index i is True iff action i is currently legal. Length == action-space size.
ActionMask = npt.NDArray[np.bool_]

# The subset of state an agent is allowed to see. Concrete shape is game-defined;
# generic here so the observation boundary is explicit everywhere it matters.
Observation = TypeVar("Observation")

# An action is any identifiable, serializable choice. Games pin the concrete type
# (e.g. an int column index for Connect Four).
ActionT = TypeVar("ActionT")
