"""Engine-authoritative board reconstruction for one recorded game.

Shared by the standalone HTML report (Slice 3b) and the browser explorer
(Slice 3c) so both step-through views reconstruct state the same way: by
replaying the game's actions through ``ConnectFourEngine``, never deriving state
themselves (see docs/adr/0009-offline-analysis-and-reporting.md).
"""

from __future__ import annotations

from gamesim.core.types import AgentId
from gamesim.games.connect_four.engine import ConnectFourEngine
from gamesim.recording.match_log import MatchGameLog

# A JSON-friendly board grid: BoardGrid[row][col], row 0 == the bottom row (see
# gamesim.games.connect_four.state.ConnectFourState).
BoardGrid = list[list[int]]


def replay_match_game(game: MatchGameLog) -> list[BoardGrid]:
    """Replay every action in ``game`` and return the board state after each ply.

    The engine is the sole rules authority: this steps a fresh
    ``ConnectFourEngine`` through ``game.seed`` and ``game.actions`` rather than
    placing discs itself. The returned sequence includes the initial, empty
    board, so its length is always ``len(game.actions) + 1`` -- index ``i`` is
    the board after ``i`` moves have been played (index 0 is the starting board).
    """
    engine = ConnectFourEngine()
    engine.reset(seed=game.seed)
    boards = [_board_grid(engine)]
    for agent, action in game.actions:
        engine.step(AgentId(agent), action)
        boards.append(_board_grid(engine))
    return boards


def _board_grid(engine: ConnectFourEngine) -> BoardGrid:
    board = engine.observation(AgentId(0)).board.astype(int).tolist()
    return board  # type: ignore[no-any-return]
