"""The Runner — the game loop that wires an engine, agents, and a recorder together.

Phase-0 stub. Phase 1 implements and tests this against the Connect Four engine.
The loop: ask the engine whose turn it is, get that agent's observation + mask, ask
the agent for an action, step the engine, forward events to the recorder, repeat until
terminal. Training loops are a specialized runner built later.
"""

from __future__ import annotations

from collections.abc import Mapping

from .agent import Agent
from .engine import Engine
from .types import AgentId

# Imported lazily/structurally to avoid a hard dependency; a recorder just needs a
# ``record`` method. The concrete Recorder protocol lives in gamesim.recording.


def run_game(
    engine: Engine,
    agents: Mapping[AgentId, Agent],
    *,
    seed: int | None = None,
    recorder: object | None = None,
) -> Mapping[AgentId, float]:
    """Drive one full game to termination and return final rewards.

    NOTE: Phase-0 skeleton — the real implementation and its tests land in Phase 1
    (plans/phase-01-engine-core.md, test group F). Kept minimal and dependency-light
    on purpose.
    """
    raise NotImplementedError("Runner is implemented in Phase 1")
