"""Smoke tests: prove the package imports and the baseline primitives work.

This is the harness check that makes the red -> green loop ready. Real engine/game
behavior is specified and tested in Phase 1 (plans/phase-01-engine-core.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gamesim import __version__
from gamesim.core import RandomAgent
from gamesim.core.events import ActionTaken, GameStarted
from gamesim.core.types import AgentId
from gamesim.recording import JsonlRecorder, NullRecorder, Recorder


def test_package_imports() -> None:
    assert __version__


def test_random_agent_only_picks_legal_actions() -> None:
    agent: RandomAgent[object] = RandomAgent(seed=0)
    mask = np.array([True, False, True, False], dtype=np.bool_)
    for _ in range(50):
        choice = agent.act(observation=None, mask=mask)
        assert mask[choice], "RandomAgent chose a masked-illegal action"


def test_random_agent_is_seed_reproducible() -> None:
    mask = np.array([True, True, True, True], dtype=np.bool_)
    a: RandomAgent[object] = RandomAgent(seed=42)
    b: RandomAgent[object] = RandomAgent(seed=42)
    seq_a = [a.act(None, mask) for _ in range(20)]
    seq_b = [b.act(None, mask) for _ in range(20)]
    assert seq_a == seq_b


def test_random_agent_raises_with_no_legal_actions() -> None:
    agent: RandomAgent[object] = RandomAgent(seed=0)
    mask = np.zeros(4, dtype=np.bool_)
    try:
        agent.act(None, mask)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on empty mask")


def test_null_recorder_is_a_noop() -> None:
    rec = NullRecorder()
    assert isinstance(rec, Recorder)
    rec.record(GameStarted(seed=1, agents=(AgentId(0), AgentId(1))))
    rec.close()  # no error, no output


def test_jsonl_recorder_writes_events_in_order(tmp_path: Path) -> None:
    log = tmp_path / "game.jsonl"
    with JsonlRecorder(log) as rec:
        assert isinstance(rec, Recorder)
        rec.record(GameStarted(seed=7, agents=(AgentId(0), AgentId(1))))
        rec.record(ActionTaken(agent=AgentId(0), action=3))

    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["type"] == "GameStarted"
    assert first["seed"] == 7
    second = json.loads(lines[1])
    assert second["type"] == "ActionTaken"
    assert second["action"] == 3
