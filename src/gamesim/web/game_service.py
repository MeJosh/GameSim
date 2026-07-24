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
from gamesim.core.replay import GameLog, replay_game
from gamesim.core.types import AgentId
from gamesim.games.connect_four import ConnectFourEncoder, ConnectFourEngine, ConnectFourObservation
from gamesim.recording.match_log import MatchGameLog, MatchLog
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


@dataclass(frozen=True)
class ReplayGameSummary:
    """Small, list-friendly description of one recorded game."""

    index: int
    seats: list[str]
    outcome: str
    total_moves: int


@dataclass(frozen=True)
class ReplayMatchSnapshot:
    """Metadata returned after the browser uploads a match log."""

    match_id: str
    agent_a: str
    agent_b: str
    games: list[ReplayGameSummary]


@dataclass(frozen=True)
class ReplaySnapshot:
    """An engine-reconstructed state at one move in a recorded game."""

    board: list[list[int]]
    game_index: int
    seats: list[str]
    outcome: str
    move: int
    total_moves: int
    current_player: str | None


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
        self._replays: dict[str, MatchLog] = {}
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

    def load_match(self, record: dict[str, object]) -> ReplayMatchSnapshot:
        """Validate a match log by replaying every game through the engine."""
        try:
            match_log = MatchLog.from_dict(record)
        except ValueError as error:
            raise GameServiceError(f"invalid match log: {error}") from error
        return self._store_match(match_log)

    def load_match_archive(self, payload: bytes) -> ReplayMatchSnapshot:
        """Validate an uploaded ZIP match archive before exposing its replay states."""
        try:
            match_log = MatchLog.from_archive_bytes(payload)
        except ValueError as error:
            raise GameServiceError(f"invalid match archive: {error}") from error
        return self._store_match(match_log)

    def _store_match(self, match_log: MatchLog) -> ReplayMatchSnapshot:
        for game in match_log.games:
            try:
                self._validate_game(match_log, game)
            except ValueError as error:
                raise GameServiceError(f"invalid match log: {error}") from error

        match_id = str(uuid4())
        self._replays[match_id] = match_log
        return ReplayMatchSnapshot(
            match_id=match_id,
            agent_a=match_log.agent_a,
            agent_b=match_log.agent_b,
            games=[
                ReplayGameSummary(
                    index=game.index,
                    seats=list(game.seats),
                    outcome=self._outcome_label(match_log, game),
                    total_moves=len(game.actions),
                )
                for game in match_log.games
            ],
        )

    def replay_at(self, match_id: str, game_index: int, move: int) -> ReplaySnapshot:
        """Reconstruct a specific recorded turn through the authoritative engine."""
        match_log = self._replays.get(match_id)
        if match_log is None:
            raise GameServiceError(f"unknown replay match: {match_id}")
        if not 0 <= game_index < len(match_log.games):
            raise GameServiceError(f"unknown game index: {game_index}")
        game = match_log.games[game_index]
        if not 0 <= move <= len(game.actions):
            raise GameServiceError(f"move must be between 0 and {len(game.actions)}")

        engine = ConnectFourEngine()
        replay_game(engine, self._game_log(game), up_to=move)
        current_player = None if engine.is_terminal() else game.seats[int(engine.current_agent())]
        return ReplaySnapshot(
            board=engine.observation(AgentId(0)).board.astype(int).tolist(),
            game_index=game.index,
            seats=list(game.seats),
            outcome=self._outcome_label(match_log, game),
            move=move,
            total_moves=len(game.actions),
            current_player=current_player,
        )

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

    @staticmethod
    def _game_log(game: MatchGameLog) -> GameLog:
        return GameLog(
            seed=game.seed,
            actions=tuple((AgentId(agent), action) for agent, action in game.actions),
        )

    @classmethod
    def _validate_game(cls, match_log: MatchLog, game: MatchGameLog) -> None:
        engine = ConnectFourEngine()
        replay_game(engine, cls._game_log(game))
        if not engine.is_terminal():
            raise ValueError(f"game {game.index} does not reach a terminal state")
        expected_outcome = cls._outcome_from_rewards(match_log, game, engine)
        if game.outcome != expected_outcome:
            raise ValueError(f"game {game.index} outcome does not match its actions")

    @staticmethod
    def _outcome_from_rewards(
        match_log: MatchLog, game: MatchGameLog, engine: ConnectFourEngine
    ) -> str:
        rewards = engine.rewards()
        if rewards[AgentId(0)] == rewards[AgentId(1)]:
            return "draw"
        winning_seat = AgentId(0) if rewards[AgentId(0)] > rewards[AgentId(1)] else AgentId(1)
        winner = game.seats[int(winning_seat)]
        if winner == match_log.agent_a:
            return "agent_a"
        if winner == match_log.agent_b:
            return "agent_b"
        raise ValueError(f"game {game.index} names a seat outside the match agents")

    @classmethod
    def _outcome_label(cls, match_log: MatchLog, game: MatchGameLog) -> str:
        if game.outcome == "draw":
            return "draw"
        return match_log.agent_a if game.outcome == "agent_a" else match_log.agent_b
