"""Tests for the Connect Four ASCII renderer -- plan Slice 3a, test 1."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pytest import CaptureFixture

from gamesim.core.types import AgentId
from gamesim.games.connect_four.engine import ConnectFourObservation
from gamesim.games.connect_four.state import NUM_COLUMNS, NUM_ROWS
from gamesim.viz.connect_four import ConnectFourRenderer, format_board
from gamesim.viz.renderer import Renderer


def _empty_board() -> npt.NDArray[np.int8]:
    return np.zeros((NUM_ROWS, NUM_COLUMNS), dtype=np.int8)


def _ruler() -> str:
    return " ".join(str(col) for col in range(NUM_COLUMNS))


def test_format_board_renders_empty_board_with_dots_and_a_column_ruler() -> None:
    text = format_board(_empty_board())

    lines = text.splitlines()
    empty_row = " ".join(["."] * NUM_COLUMNS)
    assert lines == [empty_row] * NUM_ROWS + [_ruler()]


def test_format_board_renders_known_board_with_bottom_row_printed_last() -> None:
    board = _empty_board()
    board[0, 0] = 1  # agent 0's disc, bottom-left
    board[0, 1] = 2  # agent 1's disc, bottom row
    board[1, 0] = 1  # agent 0's disc, stacked on top of its first

    text = format_board(board)
    lines = text.splitlines()

    assert len(lines) == NUM_ROWS + 1
    assert lines[-1] == _ruler()  # column ruler is the last line
    assert lines[-2] == "X O . . . . ."  # row 0 (bottom) printed just above the ruler
    assert lines[-3] == "X . . . . . ."  # row 1
    assert lines[0] == ". . . . . . ."  # row 5 (top) printed first


def test_format_board_accepts_a_plain_nested_list_grid() -> None:
    grid = [[0] * NUM_COLUMNS for _ in range(NUM_ROWS)]
    grid[0][3] = 2

    text = format_board(grid)

    assert text.splitlines()[-2].split()[3] == "O"


def test_renderer_render_prints_the_formatted_board(capsys: CaptureFixture[str]) -> None:
    board = _empty_board()
    board[0, 2] = 1
    observation = ConnectFourObservation(
        board=board,
        perspective_agent=AgentId(0),
        legal_actions=np.ones(NUM_COLUMNS, dtype=np.bool_),
    )

    ConnectFourRenderer().render(observation)

    captured = capsys.readouterr()
    assert captured.out.strip() == format_board(board)


def test_connect_four_renderer_is_a_renderer() -> None:
    assert isinstance(ConnectFourRenderer(), Renderer)
