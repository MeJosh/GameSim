"""Evaluation harness tests -- Phase 2 plan, Slice 2a test list items 9-10.

See plans/phase-02-drl-selfplay.md for the full spec these pin down.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gamesim.agents.scripted import MinimaxAgent
from gamesim.core.agent import RandomAgent
from gamesim.core.types import ActionMask
from gamesim.games.connect_four import ConnectFourEngine
from gamesim.games.connect_four.engine import ConnectFourObservation
from gamesim.games.connect_four.state import EMPTY
from gamesim.rl.evaluate import evaluate


class _FirstMoveLabeler:
    """Wraps an agent and records ``label`` whenever it's handed a fresh (empty)
    board -- i.e. whenever it is the first mover of a game -- so tests can observe
    which logical agent (a/b) went first across a run of ``evaluate()`` games."""

    def __init__(self, inner: Any, label: str, log: list[str]) -> None:
        self._inner = inner
        self._label = label
        self._log = log

    def act(self, observation: ConnectFourObservation, mask: ActionMask) -> int:
        if np.all(observation.board == EMPTY):
            self._log.append(self._label)
        result: int = self._inner.act(observation, mask)
        return result


# --- 9. Minimax beats RandomAgent well above 50%, reproducibly, alternating first ----


def test_minimax_beats_random_well_above_half_and_is_reproducible() -> None:
    engine = ConnectFourEngine()
    minimax = MinimaxAgent(depth=3)
    random_agent: RandomAgent[ConnectFourObservation] = RandomAgent(seed=123)

    result_1 = evaluate(engine, minimax, random_agent, num_games=20, seed=7)
    assert result_1.win_rate_a > 0.5

    # Reproducible: same seed, fresh engine/agents -> identical outcome counts.
    engine_2 = ConnectFourEngine()
    minimax_2 = MinimaxAgent(depth=3)
    random_agent_2: RandomAgent[ConnectFourObservation] = RandomAgent(seed=123)
    result_2 = evaluate(engine_2, minimax_2, random_agent_2, num_games=20, seed=7)

    assert result_1.wins_a == result_2.wins_a
    assert result_1.wins_b == result_2.wins_b
    assert result_1.draws == result_2.draws


def test_evaluate_alternates_first_mover() -> None:
    engine = ConnectFourEngine()
    log: list[str] = []
    agent_a = _FirstMoveLabeler(MinimaxAgent(depth=1), "a", log)
    agent_b = _FirstMoveLabeler(RandomAgent(seed=1), "b", log)

    evaluate(engine, agent_a, agent_b, num_games=6, seed=0)

    assert log == ["a", "b", "a", "b", "a", "b"]


# --- 10. Coherent win/loss/draw counts ------------------------------------------------


def test_evaluate_returns_coherent_counts() -> None:
    engine = ConnectFourEngine()
    minimax = MinimaxAgent(depth=2)
    random_agent: RandomAgent[ConnectFourObservation] = RandomAgent(seed=42)

    result = evaluate(engine, minimax, random_agent, num_games=10, seed=99)

    assert result.games == 10
    assert result.wins_a + result.wins_b + result.draws == 10
    assert 0.0 <= result.win_rate_a <= 1.0
    assert 0.0 <= result.win_rate_b <= 1.0
