"""Reusable scripted/policy agents that implement the ``Agent`` protocol.

``core.agent`` already provides the game-agnostic ``Agent`` protocol and the
baseline ``RandomAgent``. This package holds additional agents -- currently a
Connect Four ``MinimaxAgent`` -- built on top of that same interface so they drop
into ``run_game`` unchanged. Game-specific search/move-generation logic (e.g.
Connect Four's board evaluation) lives alongside the agent that uses it rather than
in the game's engine package, keeping the engine itself free of agent concerns.
"""

from __future__ import annotations

from gamesim.core.agent import RandomAgent

from .scripted import MinimaxAgent

__all__ = ["MinimaxAgent", "RandomAgent"]
