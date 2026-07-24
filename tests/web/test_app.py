"""HTTP boundary tests for the optional local play UI."""

# The module-level dependency check must precede FastAPI imports so the normal
# development extra can run this suite without installing the optional web stack.
# ruff: noqa: E402

from __future__ import annotations

import base64

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from gamesim.core.agent import RandomAgent
from gamesim.games.connect_four import ConnectFourObservation
from gamesim.recording import record_match
from gamesim.web.app import create_app
from gamesim.web.game_service import ConnectFourGameService


def test_api_starts_game_and_returns_engine_adjudicated_move() -> None:
    client = TestClient(create_app(ConnectFourGameService()))

    start = client.post("/api/games", json={"opponent": "random", "seed": 3})
    assert start.status_code == 200
    game = start.json()
    assert game["legal_columns"] == list(range(7))

    moved = client.post(f"/api/games/{game['game_id']}/moves", json={"column": 4})
    assert moved.status_code == 200
    state = moved.json()
    assert state["moves"][0] == 4
    assert len(state["moves"]) == 2
    assert state["board"][0][4] == 1


def test_api_returns_validation_error_for_illegal_column() -> None:
    client = TestClient(create_app())
    game = client.post("/api/games", json={"opponent": "random"}).json()

    response = client.post(f"/api/games/{game['game_id']}/moves", json={"column": 9})

    assert response.status_code == 422
    assert "out of range" in response.json()["detail"]


def test_api_loads_a_match_log_and_replays_a_turn() -> None:
    match = record_match(
        RandomAgent[ConnectFourObservation](seed=1),
        RandomAgent[ConnectFourObservation](seed=2),
        agent_a_name="trained",
        agent_b_name="random",
        num_games=1,
        seed=3,
    )
    client = TestClient(create_app())

    loaded = client.post("/api/replays", json={"log": match.to_dict()})
    assert loaded.status_code == 200
    match_summary = loaded.json()
    match_id = match_summary["match_id"]
    assert match_summary["games"][0]["outcome"] in {"trained", "random", "draw"}
    assert match_summary["games"][0]["total_moves"] > 0

    replay = client.get(f"/api/replays/{match_id}/games/0?move=1")
    assert replay.status_code == 200
    assert replay.json()["move"] == 1


def test_api_loads_a_match_archive_and_replays_a_turn() -> None:
    match = record_match(
        RandomAgent[ConnectFourObservation](seed=1),
        RandomAgent[ConnectFourObservation](seed=2),
        agent_a_name="trained",
        agent_b_name="random",
        num_games=1,
        seed=3,
    )
    client = TestClient(create_app())
    payload = base64.b64encode(match.to_archive_bytes()).decode("ascii")

    loaded = client.post("/api/replays/archive", json={"archive_base64": payload})
    assert loaded.status_code == 200
    match_id = loaded.json()["match_id"]

    replay = client.get(f"/api/replays/{match_id}/games/0?move=1")
    assert replay.status_code == 200
    assert replay.json()["move"] == 1
