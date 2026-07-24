"""Runner tests -- plan group F (tests 17-19) and the recorder-wiring parts of
group G (tests 20-21), since those are naturally exercised through a real game.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gamesim.core.agent import RandomAgent
from gamesim.core.events import ActionTaken, Event, GameEnded, GameStarted
from gamesim.core.runner import run_game
from gamesim.core.types import AgentId
from gamesim.games.connect_four import ConnectFourEngine
from gamesim.games.connect_four.state import NUM_COLUMNS, NUM_ROWS
from gamesim.recording import JsonlRecorder, NullRecorder


class _CollectingRecorder:
    """Test double: keeps every recorded event in memory, in order."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def record(self, event: Event) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None


def _agents(seed: int) -> dict[AgentId, RandomAgent[object]]:
    return {
        AgentId(0): RandomAgent(seed=seed),
        AgentId(1): RandomAgent(seed=seed + 1),
    }


# --- F. Determinism & the Runner ----------------------------------------------------


def test_runner_drives_two_random_agents_to_terminal_legally() -> None:
    engine = ConnectFourEngine()
    rewards = run_game(engine, _agents(seed=1), seed=123)
    assert engine.is_terminal()
    assert set(rewards) == {AgentId(0), AgentId(1)}
    # Legal-outcome sanity: either a decisive result or a draw, never a rules violation
    # (which would have raised inside step()).
    total = rewards[AgentId(0)] + rewards[AgentId(1)]
    assert total in (0, 0.0)


def test_runner_same_seed_yields_identical_games() -> None:
    engine_a = ConnectFourEngine()
    recorder_a = _CollectingRecorder()
    rewards_a = run_game(engine_a, _agents(seed=7), seed=999, recorder=recorder_a)

    engine_b = ConnectFourEngine()
    recorder_b = _CollectingRecorder()
    rewards_b = run_game(engine_b, _agents(seed=7), seed=999, recorder=recorder_b)

    actions_a = [e.action for e in recorder_a.events if isinstance(e, ActionTaken)]
    actions_b = [e.action for e in recorder_b.events if isinstance(e, ActionTaken)]
    assert actions_a == actions_b
    assert rewards_a == rewards_b
    obs_a = engine_a.observation(AgentId(0))
    obs_b = engine_b.observation(AgentId(0))
    assert np.array_equal(obs_a.board, obs_b.board)


def test_random_agent_never_selects_masked_illegal_action_over_many_games() -> None:
    for seed in range(25):
        engine = ConnectFourEngine()
        engine.reset(seed=seed)
        agents = _agents(seed=seed)
        while not engine.is_terminal():
            current = engine.current_agent()
            mask = engine.legal_actions(current)
            action = agents[current].act(engine.observation(current), mask)
            assert mask[action], "RandomAgent chose a masked-illegal action"
            engine.step(current, action)


# --- G. Logging & replay (recorder-through-Runner parts) ----------------------------


def test_null_recorder_produces_no_output_and_does_not_alter_play() -> None:
    engine_null = ConnectFourEngine()
    rewards_null = run_game(engine_null, _agents(seed=3), seed=55, recorder=NullRecorder())

    engine_plain = ConnectFourEngine()
    rewards_plain = run_game(engine_plain, _agents(seed=3), seed=55, recorder=None)

    assert rewards_null == rewards_plain
    assert np.array_equal(
        engine_null.observation(AgentId(0)).board,
        engine_plain.observation(AgentId(0)).board,
    )


def test_jsonl_recorder_writes_one_event_per_engine_event_in_order(tmp_path: Path) -> None:
    engine = ConnectFourEngine()
    log_path = tmp_path / "game.jsonl"
    collector = _CollectingRecorder()

    class _TeeRecorder:
        def __init__(self, jsonl: JsonlRecorder, collector: _CollectingRecorder) -> None:
            self._jsonl = jsonl
            self._collector = collector

        def record(self, event: Event) -> None:
            self._jsonl.record(event)
            self._collector.record(event)

        def close(self) -> None:
            self._jsonl.close()

    with JsonlRecorder(log_path) as jsonl_recorder:
        tee = _TeeRecorder(jsonl_recorder, collector)
        run_game(engine, _agents(seed=11), seed=321, recorder=tee)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(collector.events)

    # First event is always GameStarted, last is always GameEnded, one line each.
    assert isinstance(collector.events[0], GameStarted)
    assert isinstance(collector.events[-1], GameEnded)

    import json

    first = json.loads(lines[0])
    assert first["type"] == "GameStarted"
    assert first["seed"] == 321
    last = json.loads(lines[-1])
    assert last["type"] == "GameEnded"


def test_engine_action_space_matches_column_count() -> None:
    # Sanity check tying the Runner tests to the plan's mask-alignment contract.
    engine = ConnectFourEngine()
    engine.reset(seed=0)
    assert len(engine.legal_actions(AgentId(0))) == NUM_COLUMNS
    assert NUM_ROWS == 6 and NUM_COLUMNS == 7
