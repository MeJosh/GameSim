"""Tests for batch Euchre match recording and its replayable JSON format."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from gamesim.core.agent import RandomAgent
from gamesim.core.types import AgentId
from gamesim.games.euchre import EuchreEngine, EuchreObservation, EuchreRules
from gamesim.games.euchre.actions import Action
from gamesim.recording import (
    EuchreMatchLog,
    read_euchre_match_log,
    record_euchre_match,
    write_euchre_match_log,
)
from gamesim.recording.euchre_match_log import EuchreMatchGameLog


def _recorded_match(num_hands: int = 6, seed: int = 12) -> EuchreMatchLog:
    return record_euchre_match(
        RandomAgent[EuchreObservation](seed=4),
        RandomAgent[EuchreObservation](seed=9),
        team_a_name="alpha",
        team_b_name="beta",
        num_hands=num_hands,
        seed=seed,
    )


def _direct_replay(game: EuchreMatchGameLog) -> EuchreEngine:
    """Replay a recorded hand through a fresh engine, independent of the analysis
    layer's own replay helper (see analysis/replay_euchre.py) -- this test file
    should still catch a broken recording even if that module also had a bug."""
    engine = EuchreEngine()
    engine.reset(seed=game.seed, dealer=game.dealer, rules=EuchreRules(game.stick_the_dealer))
    for agent, action in game.actions:
        engine.step(AgentId(agent), Action(action))
    return engine


def test_record_euchre_match_alternates_seat_parity_and_produces_replayable_actions() -> None:
    match = _recorded_match()

    assert [game.seats for game in match.games] == [
        ("alpha", "beta", "alpha", "beta"),
        ("beta", "alpha", "beta", "alpha"),
        ("alpha", "beta", "alpha", "beta"),
        ("beta", "alpha", "beta", "alpha"),
        ("alpha", "beta", "alpha", "beta"),
        ("beta", "alpha", "beta", "alpha"),
    ]
    assert all(game.actions for game in match.games)

    for game in match.games:
        engine = _direct_replay(game)
        assert engine.is_terminal()


def test_record_euchre_match_metadata_matches_final_engine_state() -> None:
    """The trickiest part of record_euchre_match is mapping seat parity -> team
    name; cross-check every logged outcome/maker_team/points/trump/alone against
    a fresh, independent replay of that same hand."""
    match = _recorded_match()

    for game in match.games:
        engine = _direct_replay(game)
        final = engine.observation(AgentId(0))
        assert final.terminal
        assert final.trump is not None
        assert final.maker is not None
        assert final.points == game.points
        assert final.alone == game.alone
        assert int(final.trump) == game.trump

        maker_seat_is_even = int(final.maker) % 2 == 0
        maker_seat_name = game.seats[0] if maker_seat_is_even else game.seats[1]
        assert (game.maker_team == "team_a") == (maker_seat_name == "alpha")

        scoring_seat_is_even = final.scoring_team == 0
        scoring_seat_name = game.seats[0] if scoring_seat_is_even else game.seats[1]
        assert (game.outcome == "team_a") == (scoring_seat_name == "alpha")


def test_record_euchre_match_rejects_fewer_than_one_hand() -> None:
    with pytest.raises(ValueError):
        record_euchre_match(
            RandomAgent(seed=1),
            RandomAgent(seed=2),
            team_a_name="a",
            team_b_name="b",
            num_hands=0,
        )


def test_write_and_read_euchre_match_log_round_trip(tmp_path: Path) -> None:
    output_path = write_euchre_match_log(tmp_path / "match.zip", _recorded_match())

    with ZipFile(output_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert archive.namelist() == [
            "manifest.json",
            "games/0000.json",
            "games/0001.json",
            "games/0002.json",
            "games/0003.json",
            "games/0004.json",
            "games/0005.json",
        ]
    persisted = read_euchre_match_log(output_path)

    assert output_path.name == "match.zip"
    assert manifest["games"][2]["path"] == "games/0002.json"
    assert len(persisted.games) == 6
    assert persisted.games[0].to_dict() == _recorded_match().games[0].to_dict()


def test_read_euchre_match_log_rejects_unsupported_format() -> None:
    payload = json.dumps({"format": "not.a.real.format/v1", "teams": {}, "games": []}).encode()
    with pytest.raises(ValueError, match="unsupported"):
        EuchreMatchLog.from_dict(json.loads(payload))


def test_read_euchre_match_log_rejects_tampered_manifest(tmp_path: Path) -> None:
    log = _recorded_match(num_hands=1)
    manifest = {
        "format": "gamesim.euchre.match-archive/v1",
        "teams": {"team_a": log.team_a, "team_b": log.team_b},
        "games": [
            {
                **log.games[0].to_manifest_entry(),
                "outcome": "team_b" if log.games[0].outcome == "team_a" else "team_a",
            }
        ],
    }
    archive_path = tmp_path / "tampered.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("games/0000.json", json.dumps(log.games[0].to_dict()))

    with pytest.raises(ValueError, match="disagrees"):
        read_euchre_match_log(archive_path)
