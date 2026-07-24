"""Replay tests -- plan group G (tests 22-23)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gamesim.core.agent import RandomAgent
from gamesim.core.events import Event
from gamesim.core.replay import GameLog, replay_game
from gamesim.core.runner import run_game
from gamesim.core.types import AgentId
from gamesim.games.connect_four import ConnectFourEngine


class _CollectingRecorder:
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


def test_replay_from_events_reconstructs_identical_final_state() -> None:
    original = ConnectFourEngine()
    recorder = _CollectingRecorder()
    rewards = run_game(original, _agents(seed=17), seed=555, recorder=recorder)

    log = GameLog.from_events(recorder.events)
    assert log.seed == 555

    replayed = ConnectFourEngine()
    replay_game(replayed, log)

    assert replayed.is_terminal() == original.is_terminal()
    assert replayed.rewards() == rewards
    assert np.array_equal(
        replayed.observation(AgentId(0)).board, original.observation(AgentId(0)).board
    )


def test_replay_from_jsonl_file_reconstructs_identical_final_state(tmp_path: Path) -> None:
    from gamesim.recording import JsonlRecorder

    original = ConnectFourEngine()
    log_path = tmp_path / "game.jsonl"
    with JsonlRecorder(log_path) as recorder:
        rewards = run_game(original, _agents(seed=4), seed=808, recorder=recorder)

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    log = GameLog.from_records(records)

    replayed = ConnectFourEngine()
    replay_game(replayed, log)

    assert replayed.rewards() == rewards
    assert np.array_equal(
        replayed.observation(AgentId(0)).board, original.observation(AgentId(0)).board
    )


def test_truncated_log_replays_valid_mid_game_state() -> None:
    original = ConnectFourEngine()
    recorder = _CollectingRecorder()
    run_game(original, _agents(seed=9), seed=222, recorder=recorder)

    full_log = GameLog.from_events(recorder.events)
    assert len(full_log.actions) > 4

    truncated = ConnectFourEngine()
    replay_game(truncated, full_log, up_to=4)

    assert not truncated.is_terminal()
    assert truncated.observation(AgentId(0)).board.sum() > 0

    # Replaying the same prefix twice gives the same mid-game state (determinism).
    truncated_again = ConnectFourEngine()
    replay_game(truncated_again, full_log, up_to=4)
    assert np.array_equal(
        truncated.observation(AgentId(0)).board,
        truncated_again.observation(AgentId(0)).board,
    )

    # And replaying the rest of the log from scratch reaches the same terminal state
    # as the original, full-length game.
    rest = ConnectFourEngine()
    replay_game(rest, full_log)
    assert rest.is_terminal()
