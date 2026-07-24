"""Minimax Connect Four agent tests -- Phase 2 plan, Slice 2a test list item 8.

See plans/phase-02-drl-selfplay.md for the full spec these pin down.
"""

from __future__ import annotations

import numpy as np

from gamesim.agents.scripted import MinimaxAgent
from gamesim.core.types import AgentId
from gamesim.games.connect_four import ConnectFourEngine
from gamesim.games.connect_four.state import NUM_COLUMNS


def new_engine(seed: int | None = 0) -> ConnectFourEngine:
    engine = ConnectFourEngine()
    engine.reset(seed=seed)
    return engine


def _play(engine: ConnectFourEngine, moves: list[tuple[int, int]]) -> None:
    for agent, col in moves:
        engine.step(AgentId(agent), col)


# --- Never plays an illegal move, over many games ------------------------------------


def test_minimax_never_plays_illegal_move_over_several_games() -> None:
    for seed in range(5):
        engine = new_engine(seed=seed)
        agent = MinimaxAgent(depth=3)
        while not engine.is_terminal():
            current = engine.current_agent()
            mask = engine.legal_actions(current)
            observation = engine.observation(current)
            action = agent.act(observation, mask)
            assert mask[action], "MinimaxAgent chose a masked-illegal action"
            engine.step(current, action)


# --- Takes an immediate winning move when one exists ----------------------------------


def test_minimax_takes_immediate_winning_move() -> None:
    engine = new_engine()
    # Agent 0 has three in a row on the bottom row at columns 0,1,2; column 3 completes
    # the horizontal four. Agent 1's discs are placed harmlessly elsewhere.
    moves = [
        (0, 0),
        (1, 4),
        (0, 1),
        (1, 4),
        (0, 2),
        (1, 5),
        # Agent 0 to move: columns 0,1,2 filled on bottom row, column 3 wins.
    ]
    _play(engine, moves)
    assert engine.current_agent() == AgentId(0)

    agent = MinimaxAgent(depth=4)
    mask = engine.legal_actions(AgentId(0))
    observation = engine.observation(AgentId(0))
    action = agent.act(observation, mask)

    assert action == 3


# --- Blocks an immediate opponent win --------------------------------------------------


def test_minimax_blocks_immediate_opponent_win() -> None:
    engine = new_engine()
    # Agent 1 (opponent, from agent 0's perspective) has three in a row on the bottom
    # row at columns 0,1,2, threatening to win at column 3 next turn. It is agent 0's
    # turn now and must block at column 3.
    moves = [
        (0, 5),
        (1, 0),
        (0, 6),
        (1, 1),
        (0, 5),
        (1, 2),
        # Agent 0 to move: must block column 3.
    ]
    _play(engine, moves)
    assert engine.current_agent() == AgentId(0)

    agent = MinimaxAgent(depth=4)
    mask = engine.legal_actions(AgentId(0))
    observation = engine.observation(AgentId(0))
    action = agent.act(observation, mask)

    assert action == 3


# --- Deterministic tie-breaking (fixed seed / no RNG) ---------------------------------


def test_minimax_is_deterministic() -> None:
    engine_a = new_engine()
    engine_b = new_engine()
    agent_a = MinimaxAgent(depth=3)
    agent_b = MinimaxAgent(depth=3)

    actions_a = []
    actions_b = []
    while not engine_a.is_terminal():
        current = engine_a.current_agent()
        mask = engine_a.legal_actions(current)
        action = agent_a.act(engine_a.observation(current), mask)
        actions_a.append(action)
        engine_a.step(current, action)

    while not engine_b.is_terminal():
        current = engine_b.current_agent()
        mask = engine_b.legal_actions(current)
        action = agent_b.act(engine_b.observation(current), mask)
        actions_b.append(action)
        engine_b.step(current, action)

    assert actions_a == actions_b


def test_minimax_action_mask_alignment() -> None:
    engine = new_engine()
    agent = MinimaxAgent(depth=2)
    mask = engine.legal_actions(AgentId(0))
    assert len(mask) == NUM_COLUMNS
    action = agent.act(engine.observation(AgentId(0)), mask)
    assert isinstance(action, int) or isinstance(action, np.integer)
    assert 0 <= action < NUM_COLUMNS
