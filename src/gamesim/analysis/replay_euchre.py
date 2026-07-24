"""Engine-authoritative, full-visibility Euchre state reconstruction for one hand.

Same principle as ``replay.py``'s Connect Four reconstruction (docs/adr/0009): every
ply is engine-replayed through a fresh ``EuchreEngine``, never derived by this module
or by the report's JavaScript. The one deliberate departure from ``EuchreEngine``'s
public API is the *visibility* of the resulting snapshot.

``EuchreEngine.observation(agent)`` intentionally hides every other seat's hand (the
whole point of Phase 4's hidden-information boundary -- see
plans/phase-04-euchre.md). That boundary exists for **live agents during play**:
``Agent.act()`` must never see what it isn't allowed to. It does not apply here. This
module only ever runs against an *already-completed, already-recorded* hand for
offline human review (the standalone HTML report), so there is no live agent to leak
information to -- showing all four hands is the entire point (the report's "god view"
default), and a per-seat toggle is just hiding columns of already-fully-known data in
the browser, not a real information boundary. Reaching into ``EuchreEngine``'s
internal ``_state`` here (rather than combining four separate ``observation()`` calls)
is a direct, explicit choice for exactly that reason -- see
progress/2026-07-24-phase-4-euchre.md, which flagged this as the design question a
Euchre visualizer would have to resolve that Connect Four's fully-observable board
never raised.
"""

from __future__ import annotations

from dataclasses import dataclass

from gamesim.core.types import AgentId
from gamesim.games.euchre.actions import (
    ORDER_UP,
    ORDER_UP_ALONE,
    PASS,
    Action,
    is_call_suit_action,
    is_call_suit_alone_action,
    suit_of_call_action,
)
from gamesim.games.euchre.cards import card_label, suit_of, suit_symbol
from gamesim.games.euchre.engine import EuchreEngine
from gamesim.games.euchre.state import EuchreRules, EuchreState, Phase
from gamesim.recording.euchre_match_log import EuchreMatchGameLog


@dataclass(frozen=True)
class EuchreAction:
    """The action that produced a given ply, plus a precomputed human-readable
    label -- computed here (with phase context) so the report's JavaScript never
    has to know what an action *means*, only how to display a string."""

    agent: int
    action: int
    label: str


@dataclass(frozen=True)
class EuchrePlySnapshot:
    """Full, god-view Euchre state after ``ply`` actions have been applied.

    Deliberately broader than any single ``EuchreObservation`` -- see module
    docstring. ``hands`` includes all four seats' current cards, sorted for stable
    display; a sitting-out partner's hand stays as originally dealt (they never
    play, per ``EuchreState.sitting_out``).
    """

    ply: int
    phase: str
    dealer: int
    upcard: int
    turned_down_suit: int | None
    trump: int | None
    maker: int | None
    alone: bool
    sitting_out: int | None
    hands: tuple[tuple[int, ...], ...]
    current_trick: tuple[tuple[int, int], ...]
    trick_number: int
    tricks_won: tuple[int, ...]
    to_act: int
    terminal: bool
    scoring_team: int | None
    points: int | None
    last_action: EuchreAction | None


def replay_euchre_match_game(game: EuchreMatchGameLog) -> list[EuchrePlySnapshot]:
    """Replay every action in ``game`` and return the full state after each ply.

    The engine is the sole rules authority: this steps a fresh ``EuchreEngine``
    through ``game.seed``/``game.dealer``/``game.stick_the_dealer`` and
    ``game.actions`` rather than reconstructing bidding/trick logic itself. The
    returned sequence includes the initial dealt hand (ply 0, ``last_action=None``),
    so its length is always ``len(game.actions) + 1``.
    """
    engine = EuchreEngine()
    engine.reset(
        seed=game.seed,
        dealer=game.dealer,
        rules=EuchreRules(stick_the_dealer=game.stick_the_dealer),
    )
    snapshots = [_snapshot(engine, ply=0, last_action=None)]
    for ply, (agent, action) in enumerate(game.actions, start=1):
        phase_before = _state(engine).phase
        label = _describe_action(phase_before, agent, action)
        engine.step(AgentId(agent), action)
        taken = EuchreAction(agent=agent, action=action, label=label)
        snapshots.append(_snapshot(engine, ply=ply, last_action=taken))
    return snapshots


def _state(engine: EuchreEngine) -> EuchreState:
    """Reach past the observation boundary on purpose -- see module docstring."""
    state = engine._state
    assert state is not None
    return state


def _snapshot(
    engine: EuchreEngine, *, ply: int, last_action: EuchreAction | None
) -> EuchrePlySnapshot:
    state = _state(engine)
    return EuchrePlySnapshot(
        ply=ply,
        phase=state.phase.name,
        dealer=state.dealer,
        upcard=state.upcard,
        turned_down_suit=int(suit_of(state.upcard)) if state.turned_down else None,
        trump=int(state.trump) if state.trump is not None else None,
        maker=state.maker,
        alone=state.alone,
        sitting_out=state.sitting_out,
        hands=tuple(tuple(sorted(hand)) for hand in state.hands),
        current_trick=tuple((agent, card) for agent, card in state.current_trick),
        trick_number=state.trick_number,
        tricks_won=tuple(state.tricks_won),
        to_act=state.to_act,
        terminal=state.terminal,
        scoring_team=state.scoring_team,
        points=state.points if state.terminal else None,
        last_action=last_action,
    )


def _describe_action(phase: Phase, agent: int, action: Action) -> str:
    if action == PASS:
        return f"P{agent} passes"
    if action == ORDER_UP:
        return f"P{agent} orders it up"
    if action == ORDER_UP_ALONE:
        return f"P{agent} orders it up, alone"
    if is_call_suit_action(action):
        return f"P{agent} calls {suit_symbol(suit_of_call_action(action))}"
    if is_call_suit_alone_action(action):
        return f"P{agent} calls {suit_symbol(suit_of_call_action(action))}, alone"
    verb = "discards" if phase == Phase.DEALER_DISCARD else "plays"
    return f"P{agent} {verb} {card_label(action)}"
