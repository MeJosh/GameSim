"""Tests for replay_match_game -- plan Slice 3a, test 6."""

from __future__ import annotations

from gamesim.analysis import replay_match_game
from gamesim.core.agent import RandomAgent
from gamesim.core.replay import GameLog, replay_game
from gamesim.core.types import AgentId
from gamesim.games.connect_four import ConnectFourEngine, ConnectFourObservation
from gamesim.recording import record_match
from gamesim.recording.match_log import MatchGameLog


def _sample_game() -> MatchGameLog:
    match = record_match(
        RandomAgent[ConnectFourObservation](seed=1),
        RandomAgent[ConnectFourObservation](seed=2),
        agent_a_name="a",
        agent_b_name="b",
        num_games=1,
        seed=7,
    )
    return match.games[0]


def _direct_engine_replay(game: MatchGameLog) -> ConnectFourEngine:
    engine = ConnectFourEngine()
    replay_game(
        engine,
        GameLog(seed=game.seed, actions=tuple((AgentId(a), c) for a, c in game.actions)),
    )
    return engine


def test_replay_match_game_length_is_moves_plus_one() -> None:
    game = _sample_game()

    boards = replay_match_game(game)

    assert len(boards) == len(game.actions) + 1


def test_replay_match_game_first_board_is_empty() -> None:
    game = _sample_game()

    boards = replay_match_game(game)

    assert all(cell == 0 for row in boards[0] for cell in row)


def test_replay_match_game_final_board_matches_direct_engine_replay() -> None:
    game = _sample_game()

    boards = replay_match_game(game)
    engine = _direct_engine_replay(game)
    expected_final_board = engine.observation(AgentId(0)).board.astype(int).tolist()

    assert boards[-1] == expected_final_board
    assert engine.is_terminal()


def test_replay_match_game_intermediate_board_matches_partial_engine_replay() -> None:
    game = _sample_game()
    up_to = min(2, len(game.actions))

    boards = replay_match_game(game)

    partial_engine = ConnectFourEngine()
    replay_game(
        partial_engine,
        GameLog(seed=game.seed, actions=tuple((AgentId(a), c) for a, c in game.actions)),
        up_to=up_to,
    )
    expected_board = partial_engine.observation(AgentId(0)).board.astype(int).tolist()

    assert boards[up_to] == expected_board
