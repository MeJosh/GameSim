"""Connect Four board state.

A plain, typed mutable state object -- no ECS. Connect Four is simple enough that a
small state class is clearer than the (optional) ECS layer described in
docs/architecture.md; ECS is reserved for games whose complexity justifies it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

NUM_ROWS = 6
NUM_COLUMNS = 7
EMPTY = 0

# Board cell values for agent 0 / agent 1's discs. Index by AgentId (0 or 1).
PLAYER_TOKENS = (1, 2)


@dataclass
class ConnectFourState:
    """Mutable state for one Connect Four game in progress.

    ``board[row, col]`` uses row 0 as the bottom row (where discs land first) and
    ``NUM_ROWS - 1`` as the top row.
    """

    board: npt.NDArray[np.int8]
    current_agent_index: int = 0
    move_count: int = 0
    winner: int | None = None
    terminal: bool = False
    seed: int | None = None
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    @classmethod
    def new_game(cls, *, seed: int | None = None) -> ConnectFourState:
        """Build the initial (empty-board) state for a new game.

        The engine owns a single seeded RNG per docs/adr/0006-deterministic-event-
        logging.md, even though Connect Four itself has no random outcomes -- the
        seed is stored so it can be carried through recording and replay.
        """
        return cls(
            board=np.full((NUM_ROWS, NUM_COLUMNS), EMPTY, dtype=np.int8),
            seed=seed,
            rng=np.random.default_rng(seed),
        )
