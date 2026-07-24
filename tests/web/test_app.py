"""HTTP boundary tests for the optional local play UI."""

# The module-level dependency check must precede FastAPI imports so the normal
# development extra can run this suite without installing the optional web stack.
# ruff: noqa: E402

from __future__ import annotations

import base64
import json
from dataclasses import asdict
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from gamesim.agents.scripted import MinimaxAgent
from gamesim.analysis.replay import replay_match_game
from gamesim.analysis.summary import summarize_match
from gamesim.core.agent import Agent, RandomAgent
from gamesim.core.types import ActionMask
from gamesim.games.connect_four import ConnectFourObservation
from gamesim.recording import record_match
from gamesim.web.app import create_app
from gamesim.web.game_service import ConnectFourGameService


class _FirstLegalAgent:
    """Deterministic, torch-free stand-in for a "trained" agent in tests."""

    def act(self, observation: ConnectFourObservation, mask: ActionMask) -> int:
        return int(mask.nonzero()[0][0])


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


@pytest.mark.parametrize("opponent", ["random", "minimax"])
def test_api_plays_a_full_game_to_an_engine_consistent_outcome(opponent: str) -> None:
    """Drive a whole game to a terminal state against each non-torch opponent."""
    client = TestClient(create_app(ConnectFourGameService()))

    start = client.post("/api/games", json={"opponent": opponent, "seed": 7})
    assert start.status_code == 200
    state = start.json()
    assert state["opponent"] == opponent

    while state["outcome"] == "in_progress":
        column = state["legal_columns"][0]
        response = client.post(f"/api/games/{state['game_id']}/moves", json={"column": column})
        assert response.status_code == 200
        state = response.json()

    assert state["outcome"] in {"human_won", "opponent_won", "draw"}
    assert state["legal_columns"] == []
    discs_on_board = sum(cell != 0 for row in state["board"] for cell in row)
    assert discs_on_board == len(state["moves"])

    # Illegal columns are still rejected once a session exists for this opponent.
    fresh = client.post("/api/games", json={"opponent": opponent, "seed": 7}).json()
    illegal = client.post(f"/api/games/{fresh['game_id']}/moves", json={"column": 9})
    assert illegal.status_code == 422
    assert "out of range" in illegal.json()["detail"]


def test_api_plays_against_a_monkeypatched_trained_opponent_without_torch() -> None:
    """The 'trained' opponent path is exercised via the loader seam, no torch."""
    loaded_paths: list[Path] = []

    def load_agent(path: Path) -> Agent[ConnectFourObservation, int]:
        loaded_paths.append(path)
        return _FirstLegalAgent()

    service = ConnectFourGameService(trained_agent_loader=load_agent)
    client = TestClient(create_app(service))

    start = client.post(
        "/api/games",
        json={"opponent": "trained", "checkpoint_path": "checkpoints/fake.zip"},
    )
    assert start.status_code == 200
    game = start.json()
    assert game["opponent"] == "trained"
    assert loaded_paths == [Path("checkpoints/fake.zip")]

    moved = client.post(
        f"/api/games/{game['game_id']}/moves", json={"column": game["legal_columns"][0]}
    )
    assert moved.status_code == 200
    state = moved.json()
    assert len(state["moves"]) == 2


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


def test_api_summary_and_replay_endpoints_match_the_analysis_helpers() -> None:
    """Upload a minimax-v-random log; the summary/replay endpoints match 3a's helpers."""
    match = record_match(
        MinimaxAgent(depth=2),
        RandomAgent[ConnectFourObservation](seed=5),
        agent_a_name="minimax",
        agent_b_name="random",
        num_games=3,
        seed=11,
    )
    client = TestClient(create_app())

    loaded = client.post("/api/replays", json={"log": match.to_dict()})
    assert loaded.status_code == 200
    match_id = loaded.json()["match_id"]

    summary_response = client.get(f"/api/replays/{match_id}/summary")
    assert summary_response.status_code == 200
    # Round-trip the expected summary through JSON too, so tuple fields compare
    # against the same list representation the HTTP response uses.
    expected_summary = json.loads(json.dumps(asdict(summarize_match(match))))
    assert summary_response.json() == expected_summary

    game = match.games[0]
    move = min(2, len(game.actions))
    replay_response = client.get(f"/api/replays/{match_id}/games/0?move={move}")
    assert replay_response.status_code == 200
    replay_state = replay_response.json()
    assert replay_state["board"] == replay_match_game(game)[move]
    assert replay_state["move"] == move
    assert replay_state["total_moves"] == len(game.actions)
