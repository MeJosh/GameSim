"""Replay -- reconstruct a game from a recorded ``{seed, actions}`` log.

The engine is deterministic (a single seeded RNG owns all randomness -- see
docs/adr/0006-deterministic-event-logging.md), so feeding the same seed and action
sequence back through it reproduces the exact same state. ``core`` must not depend
on the recording back-end, so this module works against plain ``Event`` objects or
plain dicts (the same shape ``Event.to_dict()`` produces), never against
``JsonlRecorder`` directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .engine import Engine
from .events import ActionTaken, Event, GameStarted
from .types import ActionT, AgentId, Observation


@dataclass(frozen=True)
class GameLog:
    """A minimal, replayable record of one game: the seed plus the ordered
    sequence of (agent, action) pairs that were played."""

    seed: int | None
    actions: tuple[tuple[AgentId, Any], ...]

    @classmethod
    def from_events(cls, events: Iterable[Event]) -> GameLog:
        """Build a log from a stream of engine ``Event`` objects (e.g. everything a
        recorder saw during ``run_game``)."""
        seed: int | None = None
        actions: list[tuple[AgentId, Any]] = []
        for event in events:
            if isinstance(event, GameStarted):
                seed = event.seed
            elif isinstance(event, ActionTaken):
                actions.append((event.agent, event.action))
        return cls(seed=seed, actions=tuple(actions))

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> GameLog:
        """Build a log from plain dicts in the ``Event.to_dict()`` shape -- e.g.
        JSON-decoded lines read back from a ``JsonlRecorder`` ``.jsonl`` file."""
        seed: int | None = None
        actions: list[tuple[AgentId, Any]] = []
        for record in records:
            if record.get("type") == "GameStarted":
                seed = record["seed"]
            elif record.get("type") == "ActionTaken":
                actions.append((AgentId(record["agent"]), record["action"]))
        return cls(seed=seed, actions=tuple(actions))


def replay_game(
    engine: Engine[Observation, ActionT],
    log: GameLog,
    *,
    up_to: int | None = None,
) -> Engine[Observation, ActionT]:
    """Replay ``log`` through ``engine`` in place and return it.

    ``up_to`` truncates replay to the first ``up_to`` recorded actions, reproducing
    a valid mid-game state (for step-through debugging / visualization). ``None``
    (the default) replays the full log to its terminal state.
    """
    engine.reset(seed=log.seed)
    actions: Sequence[tuple[AgentId, Any]] = log.actions if up_to is None else log.actions[:up_to]
    for agent, action in actions:
        engine.step(agent, action)
    return engine
