"""Tests for browser-play orchestration independent of HTTP."""

from __future__ import annotations

from pathlib import Path

import pytest

from gamesim.analysis.summary import summarize_match
from gamesim.core.agent import Agent, RandomAgent
from gamesim.core.types import ActionMask
from gamesim.games.connect_four import ConnectFourObservation
from gamesim.recording import record_match
from gamesim.web.game_service import ConnectFourGameService, GameServiceError


class FirstLegalAgent:
    """Predictable opponent used to pin down service-to-engine wiring."""

    def act(self, observation: ConnectFourObservation, mask: ActionMask) -> int:
        return int(mask.nonzero()[0][0])


def test_random_game_applies_human_and_opponent_moves_through_engine() -> None:
    service = ConnectFourGameService()
    game = service.start_game(opponent="random", seed=4)

    result = service.play_human_move(game.game_id, 3)

    assert result.moves[0] == 3
    assert len(result.moves) == 2
    assert result.board[0][3] == 1
    assert sum(cell == 2 for row in result.board for cell in row) == 1
    assert result.current_player == "human"
    assert result.outcome == "in_progress"


def test_service_rejects_unknown_games_and_illegal_human_moves() -> None:
    service = ConnectFourGameService()
    with pytest.raises(GameServiceError, match="unknown game"):
        service.play_human_move("missing", 0)

    game = service.start_game(opponent="random", seed=0)
    with pytest.raises(ValueError, match="out of range"):
        service.play_human_move(game.game_id, 7)


def test_trained_agent_is_loaded_only_for_trained_games() -> None:
    paths: list[Path] = []

    def load_agent(path: Path) -> Agent[ConnectFourObservation, int]:
        paths.append(path)
        return FirstLegalAgent()

    service = ConnectFourGameService(trained_agent_loader=load_agent)
    service.start_game(opponent="random")
    assert paths == []

    checkpoint = Path("checkpoints/custom.zip")
    game = service.start_game(opponent="trained", checkpoint_path=checkpoint)
    result = service.play_human_move(game.game_id, 2)

    assert paths == [checkpoint]
    assert result.moves == [2, 0]
    assert result.board[0][0] == 2


def test_minimax_opponent_is_engine_adjudicated_and_deterministic() -> None:
    service = ConnectFourGameService()
    game = service.start_game(opponent="minimax", seed=9)
    assert game.opponent == "minimax"

    result = service.play_human_move(game.game_id, 3)

    assert result.moves[0] == 3
    assert len(result.moves) == 2
    assert result.board[0][3] == 1
    assert sum(cell == 2 for row in result.board for cell in row) == 1
    # MinimaxAgent is deterministic, so replaying the identical human move against
    # a fresh session produces the identical opponent reply.
    replay_service = ConnectFourGameService()
    replay_game = replay_service.start_game(opponent="minimax", seed=9)
    replay_result = replay_service.play_human_move(replay_game.game_id, 3)
    assert replay_result.moves == result.moves


def test_summary_matches_summarize_match_for_an_uploaded_log() -> None:
    match = record_match(
        RandomAgent[ConnectFourObservation](seed=1),
        RandomAgent[ConnectFourObservation](seed=2),
        agent_a_name="a",
        agent_b_name="b",
        num_games=2,
        seed=3,
    )
    service = ConnectFourGameService()
    loaded = service.load_match(match.to_dict())

    summary = service.summary(loaded.match_id)

    assert summary == summarize_match(match)


def test_summary_rejects_an_unknown_match_id() -> None:
    service = ConnectFourGameService()
    with pytest.raises(GameServiceError, match="unknown replay match"):
        service.summary("missing")
