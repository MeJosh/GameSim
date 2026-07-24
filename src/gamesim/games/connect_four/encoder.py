"""Connect Four encoder: board <-> tensor, plus the DRL-facing action mask.

Implements the ``Encoder`` protocol (``gamesim.rl.encoder``). This is the only place
that knows how a Connect Four ``ConnectFourObservation`` maps to the fixed-shape
arrays a neural network wants -- the engine itself stays free of any DRL concern.

Plane layout (shape ``(3, NUM_ROWS, NUM_COLUMNS)`` = ``(3, 6, 7)``, dtype
``float32``), always from the **queried agent's perspective** (canonical form) so
the same network can serve both seats during self-play:

  - plane 0 -- "mine": 1.0 where the queried agent (``observation.perspective_agent``,
    i.e. whichever agent ``engine.observation(agent)`` was called for) has a disc,
    0.0 elsewhere.
  - plane 1 -- "opponent's": 1.0 where the other agent has a disc, 0.0 elsewhere.
  - plane 2 -- "empty": 1.0 where the cell is unoccupied, 0.0 elsewhere.

Planes 0-2 partition the board (every cell is exactly one of mine / opponent's /
empty), so plane 2 is fully redundant with 0+1 but is included because it's a cheap,
commonly-used signal that saves the network from having to learn "empty = NOT mine
AND NOT opponent's" from scratch.

``action_mask`` simply returns ``observation.legal_actions`` -- the engine already
computed the correct mask for the queried agent when it built the observation (real
legal moves iff that agent is on turn and the game isn't terminal, all-false
otherwise), so this is equal to ``engine.legal_actions`` by construction in every
state, terminal included, with no separate board-derived computation to drift out of
sync.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from gamesim.core.types import ActionMask

from .engine import ConnectFourObservation
from .state import EMPTY, NUM_COLUMNS, NUM_ROWS, PLAYER_TOKENS


class ConnectFourEncoder:
    """Converts a ``ConnectFourObservation`` to/from network-friendly arrays."""

    def encode(self, observation: ConnectFourObservation) -> npt.NDArray[np.float32]:
        board = observation.board
        my_token = PLAYER_TOKENS[observation.perspective_agent]
        opponent_token = PLAYER_TOKENS[1 - observation.perspective_agent]

        planes = np.zeros((3, NUM_ROWS, NUM_COLUMNS), dtype=np.float32)
        planes[0] = board == my_token
        planes[1] = board == opponent_token
        planes[2] = board == EMPTY
        return planes

    def action_mask(self, observation: ConnectFourObservation) -> ActionMask:
        return observation.legal_actions
