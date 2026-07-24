"""HTTP boundary tests for the optional local play UI."""

# The module-level dependency check must precede FastAPI imports so the normal
# development extra can run this suite without installing the optional web stack.
# ruff: noqa: E402

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

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
