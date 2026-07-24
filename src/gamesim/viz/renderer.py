"""The Renderer interface. Game-specific, optional, side-effect-only."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gamesim.core.types import Observation


@runtime_checkable
class Renderer(Protocol[Observation]):
    """Draws a game observation. Live or replay — same interface."""

    def render(self, observation: Observation) -> None: ...
