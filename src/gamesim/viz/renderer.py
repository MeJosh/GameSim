"""The Renderer interface. Game-specific, optional, side-effect-only."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

# The observation is consumed by the renderer (contravariant).
_ObsContra = TypeVar("_ObsContra", contravariant=True)


@runtime_checkable
class Renderer(Protocol[_ObsContra]):
    """Draws a game observation. Live or replay — same interface."""

    def render(self, observation: _ObsContra) -> None: ...
