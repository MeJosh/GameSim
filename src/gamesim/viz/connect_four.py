"""ASCII renderer for Connect Four: live play and replay step-through.

Implements the ``Renderer`` protocol (``gamesim.viz.renderer``), fulfilling the
game-specific renderer promised in docs/architecture.md §3. Glyphs: ``.`` for an
empty cell, ``X`` for agent 0's disc, ``O`` for agent 1's disc -- matching
``PLAYER_TOKENS`` (1, 2) in ``gamesim.games.connect_four.state``.

``ConnectFourState.board`` uses row 0 as the bottom row (see that module's
docstring), so ``format_board`` prints the top row first and the bottom row last:
the text reads like an upright, real Connect Four board, with a 0-based column
ruler appended beneath it so a column index can be read straight off the render.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from gamesim.games.connect_four.engine import ConnectFourObservation
from gamesim.games.connect_four.state import EMPTY, NUM_COLUMNS, PLAYER_TOKENS

# Either the engine's numpy board or a plain JSON-friendly nested grid (e.g. a
# replay board reconstructed by gamesim.analysis.replay.replay_match_game).
BoardGrid = Sequence[Sequence[int]]
Board = npt.NDArray[np.int8] | BoardGrid

_GLYPHS: dict[int, str] = {EMPTY: ".", PLAYER_TOKENS[0]: "X", PLAYER_TOKENS[1]: "O"}


def _to_grid(board: Board) -> list[list[int]]:
    """Normalize a numpy board or a plain nested sequence to ``list[list[int]]``."""
    if isinstance(board, np.ndarray):
        return board.astype(int).tolist()  # type: ignore[no-any-return]
    return [[int(cell) for cell in row] for row in board]


def format_board(board: Board) -> str:
    """Render a raw board grid to text. Pure -- no side effects, no printing.

    Accepts either the engine's numpy board (``ConnectFourObservation.board``) or
    a plain ``Sequence[Sequence[int]]`` grid (e.g. a replay board reconstructed by
    ``gamesim.analysis.replay.replay_match_game``). Row 0 (the bottom row) is
    printed last, immediately above a 0-based column ruler.
    """
    grid = _to_grid(board)
    num_columns = len(grid[0]) if grid else NUM_COLUMNS
    lines = [" ".join(_GLYPHS[cell] for cell in grid[row]) for row in range(len(grid) - 1, -1, -1)]
    lines.append(" ".join(str(col) for col in range(num_columns)))
    return "\n".join(lines)


def render_board(board: Board) -> None:
    """Print a raw board grid. A thin, side-effecting wrapper over ``format_board``."""
    print(format_board(board))


class ConnectFourRenderer:
    """Prints an ASCII Connect Four board. Live or replay -- same interface.

    A pure consumer of the observation (see docs/architecture.md §3): it can never
    affect the game. Usable both attached to a running engine (live) and while
    stepping through a recorded log (replay).
    """

    def render(self, observation: ConnectFourObservation) -> None:
        render_board(observation.board)
