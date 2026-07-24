"""Engine events — the unit of logging and the signal renderers subscribe to.

Events are immutable records of what happened in a game. A recorded stream of them
(plus the seed) is sufficient to replay a game exactly
(see docs/adr/0006-deterministic-event-logging.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import AgentId

# Bump when the on-disk event shape changes, so old logs remain interpretable.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Event:
    """Base class for all engine events."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict. Subclasses extend the payload."""
        return {"type": type(self).__name__, "schema": SCHEMA_VERSION}


@dataclass(frozen=True)
class GameStarted(Event):
    seed: int
    agents: tuple[AgentId, ...]

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "seed": self.seed, "agents": list(self.agents)}


@dataclass(frozen=True)
class ActionTaken(Event):
    agent: AgentId
    action: Any  # game-specific, must be JSON-serializable

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "agent": int(self.agent), "action": self.action}


@dataclass(frozen=True)
class GameEnded(Event):
    rewards: dict[AgentId, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "rewards": {int(k): v for k, v in self.rewards.items()}}
