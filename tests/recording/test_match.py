"""Tests for batch match recording and its replayable JSON format."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from gamesim.core.agent import RandomAgent
from gamesim.core.replay import replay_game
from gamesim.core.types import AgentId
from gamesim.games.connect_four import ConnectFourEngine, ConnectFourObservation
from gamesim.recording import MatchLog, read_match_log, record_match, write_match_log
from gamesim.rl.record_matches import _parse_args
from gamesim.web.game_service import ConnectFourGameService


def _recorded_match() -> MatchLog:
    return record_match(
        RandomAgent[ConnectFourObservation](seed=4),
        RandomAgent[ConnectFourObservation](seed=9),
        agent_a_name="trained",
        agent_b_name="random",
        num_games=4,
        seed=12,
    )


def test_record_match_alternates_seats_and_preserves_replayable_actions() -> None:
    match = _recorded_match()

    assert [game.seats for game in match.games] == [
        ("trained", "random"),
        ("random", "trained"),
        ("trained", "random"),
        ("random", "trained"),
    ]
    assert all(game.actions for game in match.games)

    for game in match.games:
        engine = ConnectFourEngine()
        from gamesim.core.replay import GameLog

        replay_game(
            engine,
            GameLog(
                seed=game.seed,
                actions=tuple((AgentId(agent), action) for agent, action in game.actions),
            ),
        )
        assert engine.is_terminal()


def test_web_service_validates_match_and_reconstructs_requested_move() -> None:
    service = ConnectFourGameService()
    match = _recorded_match()

    loaded = service.load_match(match.to_dict())
    state = service.replay_at(loaded.match_id, game_index=0, move=1)

    assert len(loaded.games) == 4
    assert state.move == 1
    assert state.total_moves == len(match.games[0].actions)
    assert sum(cell != 0 for row in state.board for cell in row) == 1


def test_web_service_rejects_tampered_match_outcome() -> None:
    service = ConnectFourGameService()
    record = _recorded_match().to_dict()
    games = record["games"]
    assert isinstance(games, list)
    games[0]["outcome"] = "draw"

    try:
        service.load_match(record)
    except ValueError as error:
        assert "outcome" in str(error)
    else:
        raise AssertionError("tampered outcome should not be accepted")


def test_write_match_log_persists_manifest_and_one_file_per_game(tmp_path: Path) -> None:
    output_path = write_match_log(tmp_path / "match.zip", _recorded_match())

    with ZipFile(output_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert archive.namelist() == [
            "manifest.json",
            "games/0000.json",
            "games/0001.json",
            "games/0002.json",
            "games/0003.json",
        ]
    persisted = read_match_log(output_path)

    assert output_path.name == "match.zip"
    assert manifest["games"][2]["path"] == "games/0002.json"
    assert len(persisted.games) == 4


def test_record_matches_cli_defaults_to_a_zip_archive() -> None:
    args = _parse_args([])

    assert args.output.suffix == ".zip"
