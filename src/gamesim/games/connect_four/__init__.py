"""Connect Four: the Phase 1 reference game.

7 columns x 6 rows; drop a disc into a non-full column; win = 4 in a row
(horizontal, vertical, either diagonal); draw = board full with no winner. Plain
state, no ECS -- see docs/architecture.md and plans/phase-01-engine-core.md.
"""

from __future__ import annotations

from .actions import Action
from .encoder import ConnectFourEncoder
from .engine import ConnectFourEngine, ConnectFourObservation
from .state import NUM_COLUMNS, NUM_ROWS, ConnectFourState

__all__ = [
    "Action",
    "ConnectFourEncoder",
    "ConnectFourEngine",
    "ConnectFourObservation",
    "ConnectFourState",
    "NUM_COLUMNS",
    "NUM_ROWS",
]
