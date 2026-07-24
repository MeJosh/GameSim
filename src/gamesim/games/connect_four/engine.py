"""Connect Four engine: rules, legal-action masking, validation, terminal detection.

Implements the ``Engine`` protocol from ``gamesim.core.engine``. Plain state, no ECS
(see docs/architecture.md and plans/phase-01-engine-core.md -- Connect Four is the
clarity baseline).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from gamesim.core.engine import StepResult
from gamesim.core.events import ActionTaken, Event, GameEnded
from gamesim.core.types import ActionMask, AgentId

from .actions import Action
from .state import EMPTY, NUM_COLUMNS, NUM_ROWS, PLAYER_TOKENS, ConnectFourState

# (row-delta, col-delta) for the four line orientations checked from a placed disc:
# horizontal, vertical, and the two diagonals. Each is checked in both directions
# from the placed disc.
_DIRECTIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))


@dataclass(frozen=True)
class ConnectFourObservation:
    """What an agent sees.

    Connect Four has no hidden information, so every agent's observation shows the
    identical, full board (placeholder for the hidden-info boundary MTG will
    exercise -- see plan test group E).
    """

    board: npt.NDArray[np.int8]
    current_agent: AgentId


def _lowest_empty_row(board: npt.NDArray[np.int8], col: int) -> int | None:
    """Row index a disc dropped into ``col`` would land on, or ``None`` if full."""
    for row in range(NUM_ROWS):
        if board[row, col] == EMPTY:
            return row
    return None


def _count_along(
    board: npt.NDArray[np.int8], row: int, col: int, d_row: int, d_col: int, token: int
) -> int:
    """Count consecutive ``token`` cells starting one step past (row, col)."""
    count = 0
    r, c = row + d_row, col + d_col
    while 0 <= r < NUM_ROWS and 0 <= c < NUM_COLUMNS and board[r, c] == token:
        count += 1
        r += d_row
        c += d_col
    return count


def _is_winning_move(board: npt.NDArray[np.int8], row: int, col: int, token: int) -> bool:
    """Whether placing ``token`` at (row, col) completes a 4-in-a-row."""
    for d_row, d_col in _DIRECTIONS:
        total = (
            1
            + _count_along(board, row, col, d_row, d_col, token)
            + _count_along(board, row, col, -d_row, -d_col, token)
        )
        if total >= 4:
            return True
    return False


class ConnectFourEngine:
    """Authoritative Connect Four simulator: two agents, alternating turns."""

    def __init__(self) -> None:
        self._state: ConnectFourState | None = None

    def reset(self, *, seed: int | None = None) -> None:
        self._state = ConnectFourState.new_game(seed=seed)

    def _require_state(self) -> ConnectFourState:
        if self._state is None:
            raise RuntimeError("ConnectFourEngine.reset() must be called before use")
        return self._state

    def agents(self) -> Sequence[AgentId]:
        return (AgentId(0), AgentId(1))

    def current_agent(self) -> AgentId:
        """Agent whose turn it is.

        The turn index only advances on non-terminal steps (see ``step()``), so once
        ``is_terminal()`` is ``True`` this keeps returning whichever agent last acted
        (the mover who just won, or -- on a draw -- the agent who played the final
        disc) rather than "no one". That return value is unspecified/not meaningful
        once the game has ended: callers must check ``is_terminal()`` first and must
        not treat a terminal ``current_agent()`` as "whose turn is next", since there
        is no next turn.
        """
        return AgentId(self._require_state().current_agent_index)

    def legal_actions(self, agent: AgentId) -> ActionMask:
        state = self._require_state()
        mask = np.zeros(NUM_COLUMNS, dtype=np.bool_)
        if state.terminal:
            return mask
        for col in range(NUM_COLUMNS):
            mask[col] = state.board[NUM_ROWS - 1, col] == EMPTY
        return mask

    def step(self, agent: AgentId, action: Action) -> StepResult:
        state = self._require_state()
        if state.terminal:
            raise ValueError("cannot step: the game has already ended")
        if agent != state.current_agent_index:
            raise ValueError(
                f"it is agent {state.current_agent_index}'s turn, agent {agent} cannot act"
            )
        if not (0 <= action < NUM_COLUMNS):
            raise ValueError(f"illegal action: column {action} is out of range")
        row = _lowest_empty_row(state.board, action)
        if row is None:
            raise ValueError(f"illegal action: column {action} is full")

        token = PLAYER_TOKENS[agent]
        state.board[row, action] = token
        state.move_count += 1

        events: list[Event] = [ActionTaken(agent=agent, action=int(action))]

        if _is_winning_move(state.board, row, action, token):
            state.winner = agent
            state.terminal = True
        elif state.move_count >= NUM_ROWS * NUM_COLUMNS:
            state.terminal = True  # board full, no winner: draw
        else:
            state.current_agent_index = 1 - state.current_agent_index

        if state.terminal:
            rewards = self.rewards()
            events.append(GameEnded(rewards=dict(rewards)))

        return StepResult(terminal=state.terminal, rewards=self.rewards(), events=tuple(events))

    def observation(self, agent: AgentId) -> ConnectFourObservation:
        state = self._require_state()
        return ConnectFourObservation(
            board=state.board.copy(), current_agent=AgentId(state.current_agent_index)
        )

    def rewards(self) -> Mapping[AgentId, float]:
        state = self._require_state()
        if not state.terminal or state.winner is None:
            return {AgentId(0): 0.0, AgentId(1): 0.0}
        winner = AgentId(state.winner)
        loser = AgentId(1 - state.winner)
        return {winner: 1.0, loser: -1.0}

    def is_terminal(self) -> bool:
        return self._require_state().terminal
