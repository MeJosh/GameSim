"""Connect Four encoder tests -- Phase 2 plan, Slice 2a test list items 1-4.

See plans/phase-02-drl-selfplay.md for the full spec these pin down.
"""

from __future__ import annotations

import numpy as np

from gamesim.core.types import AgentId
from gamesim.games.connect_four import ConnectFourEngine
from gamesim.games.connect_four.encoder import ConnectFourEncoder
from gamesim.games.connect_four.state import NUM_COLUMNS, NUM_ROWS


def new_engine(seed: int | None = 0) -> ConnectFourEngine:
    engine = ConnectFourEngine()
    engine.reset(seed=seed)
    return engine


# --- 1. Shape and dtype -------------------------------------------------------------


def test_encoder_output_shape_and_dtype_for_empty_board() -> None:
    engine = new_engine()
    encoder = ConnectFourEncoder()
    observation = engine.observation(AgentId(0))

    planes = encoder.encode(observation)

    assert planes.shape == (3, NUM_ROWS, NUM_COLUMNS)
    assert planes.dtype == np.float32
    # Empty board: no discs for either side, every cell reported empty.
    assert np.all(planes[0] == 0.0)
    assert np.all(planes[1] == 0.0)
    assert np.all(planes[2] == 1.0)


# --- 2. Perspective correctness (canonical form) -------------------------------------


def test_encoder_is_perspective_correct_for_acting_agent() -> None:
    engine = new_engine()
    encoder = ConnectFourEncoder()

    # Agent 0 drops in column 0 (bottom row); now it's agent 1's turn.
    engine.step(AgentId(0), 0)

    obs_for_agent1 = engine.observation(AgentId(1))
    assert obs_for_agent1.perspective_agent == AgentId(1)
    planes_agent1 = encoder.encode(obs_for_agent1)
    # From agent 1's perspective, the opponent (agent 0) placed the only disc.
    assert planes_agent1[0, 0, 0] == 0.0  # "my" plane: not agent 1's disc
    assert planes_agent1[1, 0, 0] == 1.0  # "opponent" plane: agent 0's disc

    engine.step(AgentId(1), 1)
    obs_for_agent0 = engine.observation(AgentId(0))
    assert obs_for_agent0.perspective_agent == AgentId(0)
    planes_agent0 = encoder.encode(obs_for_agent0)
    # From agent 0's perspective, agent 0's own earlier disc is now "my" plane.
    assert planes_agent0[0, 0, 0] == 1.0
    assert planes_agent0[1, 0, 0] == 0.0
    # And agent 1's disc in column 1 is the opponent's.
    assert planes_agent0[1, 0, 1] == 1.0
    assert planes_agent0[0, 0, 1] == 0.0


# --- 3. Round-trip sanity on a known board -------------------------------------------


def test_encoder_matches_known_board() -> None:
    engine = new_engine()
    encoder = ConnectFourEncoder()

    # Agent 0: col 0, col 0 (stacked). Agent 1: col 1. Agent 0 to move again ->
    # agent 1 is next to act.
    engine.step(AgentId(0), 0)
    engine.step(AgentId(1), 1)
    engine.step(AgentId(0), 0)
    assert engine.current_agent() == AgentId(1)

    obs = engine.observation(AgentId(1))  # agent 1 to move next (acting perspective)
    planes = encoder.encode(obs)

    # "Mine" plane (agent 1): disc at (row 0, col 1).
    expected_mine = np.zeros((NUM_ROWS, NUM_COLUMNS), dtype=np.float32)
    expected_mine[0, 1] = 1.0
    assert np.array_equal(planes[0], expected_mine)

    # "Opponent" plane (agent 0): discs at (row 0, col 0) and (row 1, col 0).
    expected_opp = np.zeros((NUM_ROWS, NUM_COLUMNS), dtype=np.float32)
    expected_opp[0, 0] = 1.0
    expected_opp[1, 0] = 1.0
    assert np.array_equal(planes[1], expected_opp)

    # Empty plane is the complement of the two occupied planes.
    expected_empty = np.ones((NUM_ROWS, NUM_COLUMNS), dtype=np.float32) - (
        expected_mine + expected_opp
    )
    assert np.array_equal(planes[2], expected_empty)


# --- 4. action_mask matches engine.legal_actions --------------------------------------


def test_action_mask_matches_engine_legal_actions_at_start() -> None:
    engine = new_engine()
    encoder = ConnectFourEncoder()
    obs = engine.observation(AgentId(0))

    mask = encoder.action_mask(obs)
    expected = engine.legal_actions(AgentId(0))

    assert mask.shape == (NUM_COLUMNS,)
    assert mask.dtype == np.bool_
    assert np.array_equal(mask, expected)


def test_action_mask_matches_engine_legal_actions_with_full_column() -> None:
    engine = new_engine()
    encoder = ConnectFourEncoder()

    # Fill column 0 completely (6 discs, alternating agents).
    for _ in range(NUM_ROWS):
        agent = engine.current_agent()
        engine.step(agent, 0)

    current = engine.current_agent()
    obs = engine.observation(current)
    mask = encoder.action_mask(obs)
    expected = engine.legal_actions(current)

    assert not mask[0]
    assert np.array_equal(mask, expected)


def test_action_mask_matches_engine_legal_actions_at_terminal_state() -> None:
    # Review finding: the encoder's action_mask must equal engine.legal_actions in
    # ALL states, including terminal ones -- not just "columns physically open".
    # Both must report all-false once the game has ended.
    engine = new_engine()
    encoder = ConnectFourEncoder()

    # Agent 0 wins with a horizontal four on the bottom row (columns 0-3); agent 1
    # plays elsewhere. Mirrors the engine-level horizontal win test.
    moves = [
        (0, 0),
        (1, 4),
        (0, 1),
        (1, 4),
        (0, 2),
        (1, 5),
        (0, 3),  # agent 0 completes the horizontal win; game is now terminal
    ]
    for agent, col in moves:
        engine.step(AgentId(agent), col)
    assert engine.is_terminal()

    # Query both agents; a terminal game has no legal actions for anyone.
    for agent_index in (0, 1):
        agent = AgentId(agent_index)
        obs = engine.observation(agent)
        mask = encoder.action_mask(obs)
        expected = engine.legal_actions(agent)

        assert not np.any(mask), f"agent {agent_index}: encoder mask should be all-false"
        assert not np.any(expected), f"agent {agent_index}: engine mask should be all-false"
        assert np.array_equal(mask, expected)
