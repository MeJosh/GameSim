"""Visualization subsystem: optional, game-specific renderers.

Renderers are pure event consumers — they can never affect the game. Two modes over
one interface: attach to a live engine, or step through a recorded log
(see docs/architecture.md §3). Concrete renderers arrive in Phase 3.
"""

from .renderer import Renderer

__all__ = ["Renderer"]
