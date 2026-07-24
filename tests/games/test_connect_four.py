"""Connect Four engine tests — plan groups A-E (tests 1-16).

See plans/phase-01-engine-core.md for the full spec these pin down.
"""

from __future__ import annotations

import numpy as np
import pytest

from gamesim.core.types import AgentId
from gamesim.games.connect_four import ConnectFourEngine
from gamesim.games.connect_four.state import EMPTY, NUM_COLUMNS, NUM_ROWS


def new_engine(seed: int | None = 0) -> ConnectFourEngine:
    engine = ConnectFourEngine()
    engine.reset(seed=seed)
    return engine


# --- A. Core types & construction -------------------------------------------------


def test_fresh_engine_reports_two_agents_and_agent_zero_starts() -> None:
    engine = new_engine()
    assert list(engine.agents()) == [AgentId(0), AgentId(1)]
    assert engine.current_agent() == AgentId(0)


def test_fresh_engine_not_terminal_and_zero_rewards() -> None:
    engine = new_engine()
    assert engine.is_terminal() is False
    rewards = engine.rewards()
    assert rewards[AgentId(0)] == 0
    assert rewards[AgentId(1)] == 0


def test_fresh_board_is_empty_in_observation() -> None:
    engine = new_engine()
    obs = engine.observation(AgentId(0))
    assert obs.board.shape == (NUM_ROWS, NUM_COLUMNS)
    assert np.all(obs.board == EMPTY)


# --- B. Legal actions / masking -----------------------------------------------------


def test_all_columns_legal_at_start() -> None:
    engine = new_engine()
    mask = engine.legal_actions(AgentId(0))
    assert mask.dtype == np.bool_
    assert mask.shape == (NUM_COLUMNS,)
    assert np.all(mask)


def test_full_column_becomes_illegal() -> None:
    engine = new_engine()
    # Fill column 0 by alternating agents dropping into it six times.
    for _ in range(NUM_ROWS):
        agent = engine.current_agent()
        engine.step(agent, 0)
    mask = engine.legal_actions(engine.current_agent())
    assert not mask[0]
    # every other column still legal
    assert np.all(mask[1:])


def test_mask_length_matches_action_space_and_aligns_with_columns() -> None:
    engine = new_engine()
    mask = engine.legal_actions(AgentId(0))
    assert len(mask) == NUM_COLUMNS
    engine.step(AgentId(0), 3)
    mask_after = engine.legal_actions(engine.current_agent())
    # column 3 still legal (only one disc dropped, 5 more slots)
    assert mask_after[3]


# --- C. Applying actions & validation ------------------------------------------------


def test_step_drops_disc_at_lowest_empty_row() -> None:
    engine = new_engine()
    engine.step(AgentId(0), 2)
    obs = engine.observation(AgentId(0))
    assert obs.board[0, 2] != EMPTY
    assert np.all(obs.board[1:, 2] == EMPTY)

    engine.step(AgentId(1), 2)
    obs = engine.observation(AgentId(0))
    assert obs.board[1, 2] != EMPTY


def test_step_advances_current_agent() -> None:
    engine = new_engine()
    assert engine.current_agent() == AgentId(0)
    engine.step(AgentId(0), 0)
    assert engine.current_agent() == AgentId(1)
    engine.step(AgentId(1), 0)
    assert engine.current_agent() == AgentId(0)


def test_step_full_column_raises() -> None:
    engine = new_engine()
    for _ in range(NUM_ROWS):
        engine.step(engine.current_agent(), 5)
    with pytest.raises(ValueError):
        engine.step(engine.current_agent(), 5)


def test_step_out_of_turn_raises() -> None:
    engine = new_engine()
    assert engine.current_agent() == AgentId(0)
    with pytest.raises(ValueError):
        engine.step(AgentId(1), 0)


# --- D. Terminal conditions & rewards -------------------------------------------------


def _play(engine: ConnectFourEngine, moves: list[tuple[int, int]]) -> None:
    """Apply a sequence of (agent, column) moves."""
    for agent, col in moves:
        engine.step(AgentId(agent), col)


def test_horizontal_win_is_terminal_with_correct_rewards() -> None:
    engine = new_engine()
    # Agent 0 drops in columns 0,1,2,3 (bottom row); agent 1 drops elsewhere (col 0-3 row above).
    moves = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
        (0, 3),  # agent 0 completes horizontal 4-in-a-row on bottom row
    ]
    _play(engine, moves)
    assert engine.is_terminal()
    rewards = engine.rewards()
    assert rewards[AgentId(0)] == 1
    assert rewards[AgentId(1)] == -1


def test_vertical_win_is_terminal_with_correct_winner() -> None:
    engine = new_engine()
    moves = [
        (0, 0),
        (1, 1),
        (0, 0),
        (1, 1),
        (0, 0),
        (1, 1),
        (0, 0),  # agent 0 stacks 4 in column 0
    ]
    _play(engine, moves)
    assert engine.is_terminal()
    rewards = engine.rewards()
    assert rewards[AgentId(0)] == 1
    assert rewards[AgentId(1)] == -1


def test_diagonal_up_right_win() -> None:
    engine = new_engine()
    # Build a rising diagonal for agent 0 at (0,0),(1,1),(2,2),(3,3).
    moves = [
        (0, 0),
        (1, 1),
        (0, 1),
        (1, 2),
        (0, 2),
        (1, 3),
        (0, 2),
        (1, 3),
        (0, 3),
        (1, 0),
        (0, 3),
    ]
    _play(engine, moves)
    assert engine.is_terminal()
    rewards = engine.rewards()
    assert rewards[AgentId(0)] == 1
    assert rewards[AgentId(1)] == -1


def test_diagonal_up_left_win() -> None:
    engine = new_engine()
    # Mirror image of the up-right diagonal test, columns reversed.
    moves = [
        (0, 6),
        (1, 5),
        (0, 5),
        (1, 4),
        (0, 4),
        (1, 3),
        (0, 4),
        (1, 3),
        (0, 3),
        (1, 6),
        (0, 3),
    ]
    _play(engine, moves)
    assert engine.is_terminal()
    rewards = engine.rewards()
    assert rewards[AgentId(0)] == 1
    assert rewards[AgentId(1)] == -1


# A verified 42-move draw: generated by playing two masked-random agents (rng
# seed 40) against the real engine (game seed 0) and confirming it runs to a full
# board with rewards 0/0. Hardcoded here so the test is fast and has no search
# logic of its own to get wrong.
_DRAW_COLUMNS = [
    3,
    5,
    0,
    4,
    3,
    6,
    0,
    0,
    3,
    4,
    0,
    6,
    4,
    3,
    0,
    2,
    4,
    4,
    5,
    5,
    5,
    0,
    6,
    1,
    3,
    4,
    1,
    1,
    3,
    1,
    6,
    2,
    5,
    1,
    1,
    6,
    2,
    6,
    2,
    2,
    5,
    2,
]


def test_full_board_no_line_is_draw() -> None:
    engine = new_engine(seed=0)
    for column in _DRAW_COLUMNS:
        assert not engine.is_terminal(), "sequence ended early -- fixture is stale"
        engine.step(engine.current_agent(), column)
    assert engine.is_terminal()
    obs = engine.observation(AgentId(0))
    assert np.all(obs.board != EMPTY)
    rewards = engine.rewards()
    assert rewards[AgentId(0)] == 0
    assert rewards[AgentId(1)] == 0


def test_no_legal_actions_and_step_raises_after_terminal() -> None:
    engine = new_engine()
    moves = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
        (0, 3),
    ]
    _play(engine, moves)
    assert engine.is_terminal()
    mask = engine.legal_actions(AgentId(0))
    assert not np.any(mask)
    with pytest.raises(ValueError):
        engine.step(AgentId(1), 4)


def test_no_legal_actions_and_step_raises_after_draw_terminal() -> None:
    # Mirrors the win-terminal case above but for the draw path, which exercises a
    # different branch in step()/legal_actions() (state.winner is None at terminal).
    engine = new_engine(seed=0)
    for column in _DRAW_COLUMNS:
        engine.step(engine.current_agent(), column)
    assert engine.is_terminal()
    mask = engine.legal_actions(AgentId(0))
    assert not np.any(mask)
    with pytest.raises(ValueError):
        engine.step(AgentId(0), 3)


def test_step_out_of_range_column_raises() -> None:
    engine = new_engine()
    with pytest.raises(ValueError):
        engine.step(AgentId(0), -1)
    with pytest.raises(ValueError):
        engine.step(AgentId(0), NUM_COLUMNS)


# --- E. Observation boundary ---------------------------------------------------------


def test_observation_board_is_agent_invariant_but_honors_the_queried_agent() -> None:
    """``observation(agent)`` must honor its ``agent`` argument (review finding).

    Connect Four has no hidden information, so the *board* is identical no matter
    which agent asks. But the observation is nonetheless genuinely per-agent: it
    must be built FOR the queried agent, not silently for whoever is on turn. This
    replaces the old (buggy-behavior-matching) assertion that observations were
    simply "identical across agents" -- that was true of the board only, and masked
    the fact that ``observation(agent)`` used to ignore ``agent`` entirely.
    """
    engine = new_engine()
    engine.step(AgentId(0), 3)
    assert engine.current_agent() == AgentId(1)

    obs0 = engine.observation(AgentId(0))
    obs1 = engine.observation(AgentId(1))

    # Board is agent-invariant (perfect information).
    assert obs0.board.shape == (NUM_ROWS, NUM_COLUMNS)
    assert np.array_equal(obs0.board, obs1.board)

    # But each observation is genuinely built for the agent it was requested for.
    assert obs0.perspective_agent == AgentId(0)
    assert obs1.perspective_agent == AgentId(1)

    # Only the on-turn agent (agent 1) has real legal actions right now; the
    # inactive agent 0's observation carries an all-false mask -- a non-acting
    # agent has no legal actions.
    assert np.array_equal(obs1.legal_actions, engine.legal_actions(AgentId(1)))
    assert np.any(obs1.legal_actions)
    assert not np.any(obs0.legal_actions)
