"""In-memory orchestration for a human-versus-agent Connect Four game.

This module contains no HTTP or presentation logic. It is a narrow adapter around
the public engine and agent interfaces, which keeps the local browser client easy
to remove or replace later.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4

from gamesim.core.agent import Agent, RandomAgent
from gamesim.core.types import AgentId
from gamesim.games.connect_four import ConnectFourEncoder, ConnectFourEngine, ConnectFourObservation
from gamesim.rl.train import DEFAULT_CHECKPOINT_DIR, DEFAULT_CHECKPOINT_NAME, MaskablePolicyAgent

OpponentKind = Literal["random", "trained"]
PlayerKind = Literal["human", "opponent", "none"]
OutcomeKind = Literal["in_progress", "human_won", "opponent_won", "draw"]
ConnectFourAgent = Agent[ConnectFourObservation, int]
TrainedAgentLoader = Callable[[Path], ConnectFourAgent]

DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / f"{DEFAULT_CHECKPOINT_NAME}.zip"


class GameServiceError(ValueError):
    """Raised when a browser request cannot be applied to a game session."""


@dataclass(frozen=True)
class GameSnapshot:
    """JSON-friendly state returned after a game starts or a move is made."""

    game_id: str
    board: list[list[int]]
    legal_columns: list[int]
    current_player: PlayerKind
    outcome: OutcomeKind
    moves: list[int]
    opponent: OpponentKind


@dataclass
class _GameSession:
    engine: ConnectFourEngine
    opponent: ConnectFourAgent
    opponent_kind: OpponentKind
    moves: list[int] = field(default_factory=list)


def _load_trained_agent(checkpoint_path: Path) -> ConnectFourAgent:
    """Load the optional RL policy only when a trained game is requested."""
    if not checkpoint_path.is_file():
        raise GameServiceError(f"trained checkpoint not found: {checkpoint_path}")
    return MaskablePolicyAgent.load(checkpoint_path, ConnectFourEncoder())


class ConnectFourGameService:
    """Owns ephemeral human-versus-agent sessions for the browser adapter.

    The human is always agent 0 and moves first. This policy is deliberately local
    to the web adapter; the engine itself continues to know only about ``AgentId``.
    """

    def __init__(self, trained_agent_loader: TrainedAgentLoader = _load_trained_agent) -> None:
        self._games: dict[str, _GameSession] = {}
        self._trained_agent_loader = trained_agent_loader

    def start_game(
        self,
        *,
        opponent: OpponentKind,
        seed: int | None = None,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    ) -> GameSnapshot:
        """Create an in-memory game; the human takes the first turn."""
        engine = ConnectFourEngine()
        engine.reset(seed=seed)
        if opponent == "random":
            agent: ConnectFourAgent = RandomAgent(seed=seed)
        else:
            agent = self._trained_agent_loader(checkpoint_path)

        game_id = str(uuid4())
        self._games[game_id] = _GameSession(
            engine=engine,
            opponent=agent,
            opponent_kind=opponent,
        )
        return self._snapshot(game_id, self._games[game_id])

    def play_human_move(self, game_id: str, column: int) -> GameSnapshot:
        """Apply a human move, then one engine-adjudicated opponent response."""
        session = self._games.get(game_id)
        if session is None:
            raise GameServiceError(f"unknown game: {game_id}")
        engine = session.engine
        if engine.is_terminal():
            raise GameServiceError("the game has already ended")
        if engine.current_agent() != AgentId(0):
            raise GameServiceError("it is not the human player's turn")

        self._step(session, AgentId(0), column)
        if not engine.is_terminal():
            opponent_id = AgentId(1)
            observation = engine.observation(opponent_id)
            action = session.opponent.act(observation, engine.legal_actions(opponent_id))
            self._step(session, opponent_id, action)
        return self._snapshot(game_id, session)

    @staticmethod
    def _step(session: _GameSession, agent: AgentId, column: int) -> None:
        session.engine.step(agent, column)
        session.moves.append(column)

    @staticmethod
    def _snapshot(game_id: str, session: _GameSession) -> GameSnapshot:
        engine = session.engine
        board = engine.observation(AgentId(0)).board.astype(int).tolist()
        if engine.is_terminal():
            rewards = engine.rewards()
            if rewards[AgentId(0)] > 0:
                outcome: OutcomeKind = "human_won"
            elif rewards[AgentId(1)] > 0:
                outcome = "opponent_won"
            else:
                outcome = "draw"
            current_player: PlayerKind = "none"
            legal_columns: list[int] = []
        else:
            outcome = "in_progress"
            current_player = "human" if engine.current_agent() == AgentId(0) else "opponent"
            legal_columns = [
                column
                for column, legal in enumerate(engine.legal_actions(engine.current_agent()))
                if legal
            ]
        return GameSnapshot(
            game_id=game_id,
            board=board,
            legal_columns=legal_columns,
            current_player=current_player,
            outcome=outcome,
            moves=list(session.moves),
            opponent=session.opponent_kind,
        )
