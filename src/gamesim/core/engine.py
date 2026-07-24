"""The Engine interface — the authoritative, in-memory simulator.

An engine owns game state, enforces the rules, validates every action, and is the
single source of truth. It depends on nothing else in the project. Games implement
this Protocol; agents, logging, viz, and DRL all consume it.

This is a Phase-0 stub: the interface is fixed here so Phase 1 can implement Connect
Four against it red -> green (see plans/phase-01-engine-core.md).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar, runtime_checkable

from .events import Event
from .types import ActionMask, AgentId


@dataclass(frozen=True)
class StepResult:
    """What the engine returns after applying one action."""

    terminal: bool
    rewards: Mapping[AgentId, float]
    events: Sequence[Event] = field(default_factory=tuple)


# Protocol variance: observations are produced by the engine (covariant); actions
# are consumed by ``step`` (contravariant).
_ObsCo = TypeVar("_ObsCo", covariant=True)
_ActContra = TypeVar("_ActContra", contravariant=True)


@runtime_checkable
class Engine(Protocol[_ObsCo, _ActContra]):
    """Authoritative simulator for a single game.

    Implementations MUST:
      * route all randomness through a single seeded RNG (determinism);
      * reject illegal actions loudly in ``step`` (validation);
      * expose only permitted state via ``observation`` (per-agent boundary);
      * report legality via ``legal_actions`` (action masking).
    """

    def reset(self, *, seed: int | None = None) -> None:
        """Start a new game. A given seed must fully determine all randomness."""
        ...

    def agents(self) -> Sequence[AgentId]:
        """All players in the game, in a stable order."""
        ...

    def current_agent(self) -> AgentId:
        """The agent whose turn it is to act."""
        ...

    def legal_actions(self, agent: AgentId) -> ActionMask:
        """Boolean mask of currently-legal actions for ``agent``."""
        ...

    def step(self, agent: AgentId, action: _ActContra) -> StepResult:
        """Validate and apply ``action`` for ``agent``; advance the game.

        Raises if the action is illegal or it is not ``agent``'s turn.
        """
        ...

    def observation(self, agent: AgentId) -> _ObsCo:
        """The subset of state ``agent`` is allowed to see."""
        ...

    def rewards(self) -> Mapping[AgentId, float]:
        """Current reward for each agent (0 until terminal, by convention)."""
        ...

    def is_terminal(self) -> bool:
        """Whether the game has ended."""
        ...
