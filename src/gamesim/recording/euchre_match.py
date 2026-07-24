"""Run a 4-agent Euchre match and preserve every hand for replay.

Mirrors ``match.py``'s shape (record a reproducible batch, alternate seating for
fairness, return a versioned log) but can't reuse ``core.runner.run_game`` directly:
``run_game`` only forwards ``seed`` to ``Engine.reset``, while ``EuchreEngine.reset``
also takes ``dealer``/``rules`` (see ``games.euchre.engine.EuchreEngine.reset``'s
docstring on why those are engine-specific extras, not part of the base protocol).
``_play_one_hand`` below is the same loop as ``run_game``, with those two extra
knobs threaded through.
"""

from __future__ import annotations

import numpy as np

from gamesim.core.agent import Agent
from gamesim.core.events import GameStarted
from gamesim.core.replay import GameLog
from gamesim.core.types import AgentId
from gamesim.games.euchre import EuchreEngine, EuchreObservation, EuchreRules
from gamesim.games.euchre.actions import Action

from .euchre_match_log import EuchreMatchGameLog, EuchreMatchLog, EuchreMatchOutcome
from .recorder import EventCollector


def record_euchre_match(
    team_a_agent: Agent[EuchreObservation, Action],
    team_b_agent: Agent[EuchreObservation, Action],
    *,
    team_a_name: str,
    team_b_name: str,
    num_hands: int = 100,
    seed: int = 0,
    rules: EuchreRules | None = None,
) -> EuchreMatchLog:
    """Play and record a reproducible batch of hands.

    Partnerships are fixed by seat (0&2 vs 1&3 -- see
    ``games.euchre.state.team_of``), so which *team* wins is partly a function of
    seating (the dealer's left-hand seat bids first each hand). To keep the match
    fair rather than confounding "team_a" with "always seats 0&2", ``team_a``
    alternates between the even and odd seats across hands, same idea as
    ``record_match`` alternating who moves first in Connect Four. The dealer seat
    itself stays fixed at 0 for every hand (only which *team* sits there varies).
    """
    if num_hands < 1:
        raise ValueError("num_hands must be at least 1")
    active_rules = rules if rules is not None else EuchreRules()
    rng = np.random.default_rng(seed)
    games: list[EuchreMatchGameLog] = []
    for index in range(num_hands):
        hand_seed = int(rng.integers(0, 2**31 - 1))
        team_a_on_even_seats = index % 2 == 0
        if team_a_on_even_seats:
            seat_agents = {
                AgentId(0): team_a_agent,
                AgentId(1): team_b_agent,
                AgentId(2): team_a_agent,
                AgentId(3): team_b_agent,
            }
            seats = (team_a_name, team_b_name, team_a_name, team_b_name)
        else:
            seat_agents = {
                AgentId(0): team_b_agent,
                AgentId(1): team_a_agent,
                AgentId(2): team_b_agent,
                AgentId(3): team_a_agent,
            }
            seats = (team_b_name, team_a_name, team_b_name, team_a_name)

        collector = EventCollector()
        engine = _play_one_hand(
            seat_agents, seed=hand_seed, dealer=0, rules=active_rules, recorder=collector
        )
        final = engine.observation(AgentId(0))
        assert final.terminal and final.maker is not None
        assert final.scoring_team is not None
        assert final.points is not None and final.trump is not None

        outcome = _team_name(final.scoring_team, team_a_on_even_seats)
        maker_team = _team_name(int(final.maker) % 2, team_a_on_even_seats)

        game_log = GameLog.from_events(collector.events)
        games.append(
            EuchreMatchGameLog(
                index=index,
                seed=hand_seed,
                dealer=0,
                stick_the_dealer=active_rules.stick_the_dealer,
                seats=seats,
                actions=tuple((int(agent), int(action)) for agent, action in game_log.actions),
                outcome=outcome,
                points=final.points,
                maker_team=maker_team,
                trump=int(final.trump),
                alone=final.alone,
            )
        )
    return EuchreMatchLog(team_a=team_a_name, team_b=team_b_name, games=tuple(games))


def _play_one_hand(
    agents: dict[AgentId, Agent[EuchreObservation, Action]],
    *,
    seed: int,
    dealer: int,
    rules: EuchreRules,
    recorder: EventCollector,
) -> EuchreEngine:
    """The Euchre analogue of ``core.runner.run_game``, with ``dealer``/``rules``
    threaded through to ``EuchreEngine.reset`` (see module docstring)."""
    engine = EuchreEngine()
    engine.reset(seed=seed, dealer=dealer, rules=rules)
    recorder.record(GameStarted(seed=seed, agents=tuple(engine.agents())))

    while not engine.is_terminal():
        current = engine.current_agent()
        agent = agents[current]
        observation = engine.observation(current)
        mask = engine.legal_actions(current)
        action = agent.act(observation, mask)
        result = engine.step(current, action)
        for event in result.events:
            recorder.record(event)

    return engine


def _team_name(parity: int, team_a_on_even_seats: bool) -> EuchreMatchOutcome:
    """``parity`` is a seat parity (0 or 1, see ``games.euchre.state.team_of``)."""
    is_team_a = (parity == 0) == team_a_on_even_seats
    return "team_a" if is_team_a else "team_b"
