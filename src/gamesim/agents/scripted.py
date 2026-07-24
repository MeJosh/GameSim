"""Scripted (non-learned) agents.

``MinimaxAgent`` is a depth-limited alpha-beta search baseline for Connect Four,
used as an evaluation opponent in Phase 2 (see plans/phase-02-drl-selfplay.md,
Slice 2a). It implements the ``Agent`` protocol from ``gamesim.core.agent`` and only
depends on the Connect Four game's public board layout (``NUM_ROWS``,
``NUM_COLUMNS``, ``PLAYER_TOKENS``, ``EMPTY``) -- not on the engine's internal
search/validation code -- so the engine stays untouched.

The search works on a private, mutable copy of the board (plain numpy array
manipulation with backtracking) rather than repeatedly calling into
``ConnectFourEngine``, since minimax needs to explore many hypothetical futures
cheaply and the engine is a stateful, validating object not meant for that.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from gamesim.core.types import ActionMask
from gamesim.games.connect_four.engine import ConnectFourObservation
from gamesim.games.connect_four.state import EMPTY, NUM_COLUMNS, NUM_ROWS, PLAYER_TOKENS

# Directions checked from a placed disc: horizontal, vertical, and both diagonals.
# Mirrors ConnectFourEngine's own set (see games/connect_four/engine.py); duplicated
# here rather than imported since that helper is a private engine implementation
# detail and minimax operates on its own scratch board, not the engine's state.
_DIRECTIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))

# A score large enough to dominate any heuristic value, with a small per-ply bonus/
# penalty so faster forced wins/slower forced losses are preferred among winning
# lines (see docs on `_negamax` below).
_WIN_SCORE = 1_000_000.0

# Center-column control bonus per disc, and per-window heuristic weights. These are
# a standard, well-known Connect Four evaluation (see e.g. the widely-used
# "count open three/two windows" heuristic); tuned for "clearly beats random", not
# perfect play.
_CENTER_WEIGHT = 3.0
_THREE_IN_WINDOW = 5.0
_TWO_IN_WINDOW = 2.0
_OPPONENT_THREE_IN_WINDOW = 4.0
_OPPONENT_TWO_IN_WINDOW = 1.0


def _lowest_empty_row(board: npt.NDArray[np.int8], col: int) -> int | None:
    for row in range(NUM_ROWS):
        if board[row, col] == EMPTY:
            return row
    return None


def _legal_columns(board: npt.NDArray[np.int8]) -> list[int]:
    return [col for col in range(NUM_COLUMNS) if board[NUM_ROWS - 1, col] == EMPTY]


def _count_along(
    board: npt.NDArray[np.int8], row: int, col: int, d_row: int, d_col: int, token: int
) -> int:
    count = 0
    r, c = row + d_row, col + d_col
    while 0 <= r < NUM_ROWS and 0 <= c < NUM_COLUMNS and board[r, c] == token:
        count += 1
        r += d_row
        c += d_col
    return count


def _is_winning_move(board: npt.NDArray[np.int8], row: int, col: int, token: int) -> bool:
    for d_row, d_col in _DIRECTIONS:
        total = (
            1
            + _count_along(board, row, col, d_row, d_col, token)
            + _count_along(board, row, col, -d_row, -d_col, token)
        )
        if total >= 4:
            return True
    return False


def _score_window(window: npt.NDArray[np.int8], token: int, opponent_token: int) -> float:
    token_count = int(np.count_nonzero(window == token))
    opponent_count = int(np.count_nonzero(window == opponent_token))
    empty_count = int(np.count_nonzero(window == EMPTY))

    if token_count > 0 and opponent_count > 0:
        return 0.0  # contested window, neither side can complete it
    if token_count == 3 and empty_count == 1:
        return _THREE_IN_WINDOW
    if token_count == 2 and empty_count == 2:
        return _TWO_IN_WINDOW
    if opponent_count == 3 and empty_count == 1:
        return -_OPPONENT_THREE_IN_WINDOW
    if opponent_count == 2 and empty_count == 2:
        return -_OPPONENT_TWO_IN_WINDOW
    return 0.0


def _heuristic(board: npt.NDArray[np.int8], token: int, opponent_token: int) -> float:
    """Static evaluation from ``token``'s point of view (higher = better for it)."""
    score = 0.0
    center_col = NUM_COLUMNS // 2
    score += _CENTER_WEIGHT * int(np.count_nonzero(board[:, center_col] == token))

    for row in range(NUM_ROWS):
        for col in range(NUM_COLUMNS - 3):
            window = board[row, col : col + 4]
            score += _score_window(window, token, opponent_token)

    for col in range(NUM_COLUMNS):
        for row in range(NUM_ROWS - 3):
            window = board[row : row + 4, col]
            score += _score_window(window, token, opponent_token)

    for row in range(NUM_ROWS - 3):
        for col in range(NUM_COLUMNS - 3):
            rising = np.array([board[row + i, col + i] for i in range(4)], dtype=np.int8)
            score += _score_window(rising, token, opponent_token)
            falling = np.array([board[row + 3 - i, col + i] for i in range(4)], dtype=np.int8)
            score += _score_window(falling, token, opponent_token)

    return score


def _negamax(
    board: npt.NDArray[np.int8],
    depth: int,
    alpha: float,
    beta: float,
    token: int,
    opponent_token: int,
) -> float:
    """Negamax search: returns the score from ``token``'s point of view.

    Standard negamax formulation (equivalent to minimax with alternating sign) so one
    recursive function handles both "my turn" and "opponent's turn" nodes.
    """
    legal = _legal_columns(board)
    if not legal:
        return 0.0  # board full: draw
    if depth == 0:
        return _heuristic(board, token, opponent_token)

    best = -np.inf
    for col in legal:
        row = _lowest_empty_row(board, col)
        assert row is not None  # `col` came from `_legal_columns`
        board[row, col] = token
        if _is_winning_move(board, row, col, token):
            score = _WIN_SCORE + depth  # prefer winning sooner (higher remaining depth)
        else:
            score = -_negamax(board, depth - 1, -beta, -alpha, opponent_token, token)
        board[row, col] = EMPTY

        best = max(best, score)
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return best


class MinimaxAgent:
    """Depth-limited alpha-beta Connect Four agent.

    Deterministic: ties are broken by always preferring the lowest legal column
    index among equally-scored moves (columns are searched in ascending order and a
    strict ``>`` comparison keeps the first-seen best), so the same position always
    produces the same move.

    Depth guarantee: an immediate win for the agent to move is always taken (``act``
    special-cases it before any search, at any ``depth >= 1``). Exact detection of an
    immediate *opponent* win-next-turn (i.e. guaranteed blocking) requires searching
    at least one full ply past the agent's own move, which needs ``depth >= 2`` --
    at ``depth == 1`` the search bottoms out at the heuristic immediately after the
    agent's own move without ever looking at the opponent's reply, so blocking is
    only as good as ``_heuristic``'s three-in-a-window term, not guaranteed.
    """

    def __init__(self, depth: int = 4) -> None:
        if depth < 1:
            raise ValueError("MinimaxAgent depth must be >= 1")
        self._depth = depth

    def act(self, observation: ConnectFourObservation, mask: ActionMask) -> int:
        legal = [col for col in range(NUM_COLUMNS) if mask[col]]
        if not legal:
            raise ValueError("MinimaxAgent.act called with no legal actions")

        board = observation.board.copy()
        token = PLAYER_TOKENS[observation.perspective_agent]
        opponent_token = PLAYER_TOKENS[1 - observation.perspective_agent]

        best_col = legal[0]
        best_score = -np.inf
        for col in legal:
            row = _lowest_empty_row(board, col)
            assert row is not None  # `col` came from the legal-action mask
            board[row, col] = token
            if _is_winning_move(board, row, col, token):
                board[row, col] = EMPTY
                return col  # immediate win: take it, no need to search further
            score = -_negamax(board, self._depth - 1, -np.inf, np.inf, opponent_token, token)
            board[row, col] = EMPTY

            if score > best_score:
                best_score = score
                best_col = col

        return best_col
