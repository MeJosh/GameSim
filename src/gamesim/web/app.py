"""FastAPI application for the optional local Connect Four play UI."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .game_service import ConnectFourGameService, GameServiceError

_STATIC_DIR = Path(__file__).parent / "static"


class NewGameRequest(BaseModel):
    opponent: Literal["random", "trained"] = "random"
    seed: int | None = None
    checkpoint_path: str | None = None


class MoveRequest(BaseModel):
    column: int


class ReplayLoadRequest(BaseModel):
    log: dict[str, Any]


class ReplayArchiveRequest(BaseModel):
    archive_base64: str


def create_app(service: ConnectFourGameService | None = None) -> FastAPI:
    """Build the local app with an injectable orchestration service for tests."""
    game_service = service or ConnectFourGameService()
    app = FastAPI(title="GameSim Connect Four", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.post("/api/games")
    def new_game(request: NewGameRequest) -> dict[str, object]:
        try:
            checkpoint_path = Path(request.checkpoint_path) if request.checkpoint_path else None
            if checkpoint_path is None:
                snapshot = game_service.start_game(opponent=request.opponent, seed=request.seed)
            else:
                snapshot = game_service.start_game(
                    opponent=request.opponent,
                    seed=request.seed,
                    checkpoint_path=checkpoint_path,
                )
        except GameServiceError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return snapshot.__dict__

    @app.post("/api/games/{game_id}/moves")
    def play_move(game_id: str, request: MoveRequest) -> dict[str, object]:
        try:
            snapshot = game_service.play_human_move(game_id, request.column)
        except GameServiceError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return snapshot.__dict__

    @app.post("/api/replays")
    def load_replay(request: ReplayLoadRequest) -> dict[str, object]:
        try:
            snapshot = game_service.load_match(request.log)
        except GameServiceError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return snapshot.__dict__

    @app.post("/api/replays/archive")
    def load_replay_archive(request: ReplayArchiveRequest) -> dict[str, object]:
        try:
            payload = base64.b64decode(request.archive_base64, validate=True)
            snapshot = game_service.load_match_archive(payload)
        except (GameServiceError, binascii.Error) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return snapshot.__dict__

    @app.get("/api/replays/{match_id}/games/{game_index}")
    def replay_at(match_id: str, game_index: int, move: int = 0) -> dict[str, object]:
        try:
            snapshot = game_service.replay_at(match_id, game_index, move)
        except GameServiceError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return snapshot.__dict__

    return app
