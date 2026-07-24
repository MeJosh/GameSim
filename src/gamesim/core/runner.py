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
from .types import ActionT, AgentId, Observation

# The recorder is typed structurally as ``object`` on purpose: ``core`` must not
# depend on the recording back-end (see docs/architecture.md dependency rule).
# The Runner only needs something with a ``record`` method, wired in Phase 1.


def run_game(
    engine: Engine[Observation, ActionT],
    agents: Mapping[AgentId, Agent[Observation, ActionT]],
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
