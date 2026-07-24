"""The Runner — the game loop that wires an engine, agents, and a recorder together.

Phase-0 stub. Phase 1 implements and tests this against the Connect Four engine.
The loop: ask the engine whose turn it is, get that agent's observation + mask, ask
the agent for an action, step the engine, forward events to the recorder, repeat until
terminal. Training loops are a specialized runner built later.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .agent import Agent
from .engine import Engine
from .events import Event, GameStarted
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

    The loop: ask the engine whose turn it is, get that agent's observation and
    legal-action mask, ask the agent for an action, ``step`` the engine, forward
    every emitted event to ``recorder`` (if given), repeat until terminal.

    If ``seed`` is ``None`` a seed is generated so the game is still fully
    determined and reproducible from the ``GameStarted`` event alone (the engine
    owns a single seeded RNG -- see docs/adr/0006-deterministic-event-logging.md).
    """
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))

    def _record(event: Event) -> None:
        if recorder is not None:
            recorder.record(event)  # type: ignore[attr-defined]

    engine.reset(seed=seed)
    _record(GameStarted(seed=seed, agents=tuple(engine.agents())))

    while not engine.is_terminal():
        current = engine.current_agent()
        agent = agents[current]
        observation = engine.observation(current)
        mask = engine.legal_actions(current)
        action = agent.act(observation, mask)
        result = engine.step(current, action)
        for event in result.events:
            _record(event)

    return engine.rewards()
