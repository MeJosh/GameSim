"""Run a two-agent Connect Four match and preserve every game for replay."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from gamesim.core.agent import Agent
from gamesim.core.replay import GameLog
from gamesim.core.runner import run_game
from gamesim.core.types import AgentId
from gamesim.games.connect_four import ConnectFourEngine, ConnectFourObservation

from .match_log import MatchGameLog, MatchLog, MatchOutcome
from .recorder import EventCollector


def record_match(
    agent_a: Agent[ConnectFourObservation, int],
    agent_b: Agent[ConnectFourObservation, int],
    *,
    agent_a_name: str,
    agent_b_name: str,
    num_games: int = 100,
    seed: int = 0,
) -> MatchLog:
    """Play and record a reproducible match, alternating which agent moves first."""
    if num_games < 1:
        raise ValueError("num_games must be at least 1")
    rng = np.random.default_rng(seed)
    games: list[MatchGameLog] = []
    for index in range(num_games):
        game_seed = int(rng.integers(0, 2**31 - 1))
        a_moves_first = index % 2 == 0
        seats = (agent_a_name, agent_b_name) if a_moves_first else (agent_b_name, agent_a_name)
        agents = (
            {AgentId(0): agent_a, AgentId(1): agent_b}
            if a_moves_first
            else {AgentId(0): agent_b, AgentId(1): agent_a}
        )
        collector = EventCollector()
        rewards = run_game(ConnectFourEngine(), agents, seed=game_seed, recorder=collector)
        outcome = _outcome(rewards, a_moves_first)
        game_log = GameLog.from_events(collector.events)
        games.append(
            MatchGameLog(
                index=index,
                seed=game_seed,
                seats=seats,
                actions=tuple((int(agent), int(action)) for agent, action in game_log.actions),
                outcome=outcome,
            )
        )
    return MatchLog(agent_a=agent_a_name, agent_b=agent_b_name, games=tuple(games))


def _outcome(rewards: Mapping[AgentId, float], a_moves_first: bool) -> MatchOutcome:
    typed_rewards = rewards
    if typed_rewards[AgentId(0)] == typed_rewards[AgentId(1)]:
        return "draw"
    winning_seat = (
        AgentId(0) if typed_rewards[AgentId(0)] > typed_rewards[AgentId(1)] else AgentId(1)
    )
    agent_a_seat = AgentId(0) if a_moves_first else AgentId(1)
    return "agent_a" if winning_seat == agent_a_seat else "agent_b"
