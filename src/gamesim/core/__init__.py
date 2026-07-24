"""Core interfaces shared by every game, agent, and subsystem.

Nothing in ``core`` may import agents, visualization, logging back-ends, or DRL
libraries. Dependencies point *inward* toward ``core`` (see docs/architecture.md).
"""

from .agent import Agent, RandomAgent
from .engine import Engine, StepResult
from .events import ActionTaken, Event, GameEnded, GameStarted
from .replay import GameLog, replay_game
from .runner import run_game
from .types import ActionMask, AgentId, Observation

__all__ = [
    "Agent",
    "RandomAgent",
    "Engine",
    "StepResult",
    "Event",
    "GameStarted",
    "ActionTaken",
    "GameEnded",
    "AgentId",
    "ActionMask",
    "Observation",
    "GameLog",
    "replay_game",
    "run_game",
]
