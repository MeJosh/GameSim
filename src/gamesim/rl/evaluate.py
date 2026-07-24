"""Evaluation harness: play N games between two agents, alternating who moves first.

Built on the existing ``core.runner.run_game``, so it works with any ``Engine`` +
two ``Agent`` pairing (not just Connect Four's), and inherits the engine's
determinism -- a fixed ``seed`` here makes an entire evaluation run reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gamesim.core.agent import Agent
from gamesim.core.engine import Engine
from gamesim.core.runner import run_game
from gamesim.core.types import ActionT, AgentId, Observation


@dataclass(frozen=True)
class EvaluationResult:
    """Outcome of an ``evaluate()`` run, from ``agent_a``'s point of view.

    ``wins_a``/``wins_b``/``draws`` always sum to ``games``.
    """

    games: int
    wins_a: int
    wins_b: int
    draws: int

    @property
    def win_rate_a(self) -> float:
        return self.wins_a / self.games if self.games else 0.0

    @property
    def win_rate_b(self) -> float:
        return self.wins_b / self.games if self.games else 0.0


def evaluate(
    engine: Engine[Observation, ActionT],
    agent_a: Agent[Observation, ActionT],
    agent_b: Agent[Observation, ActionT],
    *,
    num_games: int,
    seed: int | None = None,
) -> EvaluationResult:
    """Play ``num_games`` between ``agent_a`` and ``agent_b`` via ``run_game``.

    Which agent occupies seat ``AgentId(0)`` (moves first) alternates every game --
    ``agent_a`` goes first on even-indexed games (0, 2, 4, ...), ``agent_b`` on
    odd-indexed ones -- so neither agent accrues a first-move advantage over the
    run. Per-game seeds are drawn from a ``numpy`` generator seeded with ``seed``,
    so the whole run (outcomes, and every individual game) is reproducible.
    """
    rng = np.random.default_rng(seed)
    wins_a = 0
    wins_b = 0
    draws = 0

    for game_index in range(num_games):
        game_seed = int(rng.integers(0, 2**31 - 1))
        a_moves_first = game_index % 2 == 0
        if a_moves_first:
            seat_a, seat_b = AgentId(0), AgentId(1)
        else:
            seat_a, seat_b = AgentId(1), AgentId(0)

        agents = {seat_a: agent_a, seat_b: agent_b}
        rewards = run_game(engine, agents, seed=game_seed)

        reward_a = rewards[seat_a]
        reward_b = rewards[seat_b]
        if reward_a > reward_b:
            wins_a += 1
        elif reward_b > reward_a:
            wins_b += 1
        else:
            draws += 1

    return EvaluationResult(games=num_games, wins_a=wins_a, wins_b=wins_b, draws=draws)
