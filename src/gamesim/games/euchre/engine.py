"""Euchre engine: bidding, discard, trick play, scoring -- one hand per episode.

Implements the ``Engine`` protocol from ``gamesim.core.engine`` (see
plans/phase-04-euchre.md for the full spec). Plain state, no ECS -- same baseline
decision as Connect Four.

Phase machine (``EuchreState.phase``):
``BID_ROUND_1`` -> (order-up) -> ``DEALER_DISCARD`` -> ``TRICK_PLAY``
``BID_ROUND_1`` -> (all pass) -> ``BID_ROUND_2`` -> (call suit) -> ``TRICK_PLAY``
``BID_ROUND_2`` -> (all pass, stick-the-dealer off) -> redeal -> ``BID_ROUND_1``
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from gamesim.core.engine import StepResult
from gamesim.core.events import ActionTaken, Event, GameEnded
from gamesim.core.types import ActionMask, AgentId

from .actions import (
    NUM_ACTIONS,
    ORDER_UP,
    ORDER_UP_ALONE,
    PASS,
    Action,
    call_suit_action,
    call_suit_alone_action,
    is_call_suit_alone_action,
    suit_of_call_action,
)
from .cards import Card, Suit, effective_suit, plain_rank, suit_of, trump_rank
from .state import (
    NUM_PLAYERS,
    TRICKS_PER_HAND,
    EuchreRules,
    EuchreState,
    Phase,
    deal_hand,
    partner_of,
    team_of,
)


@dataclass(frozen=True)
class EuchreObservation:
    """What a specific agent sees, from that agent's own perspective.

    Unlike Connect Four (no hidden information), ``hand`` is genuinely per-agent: it
    contains only ``perspective_agent``'s own cards. Other seats' hand *sizes* are
    public (a real player can see how many cards everyone holds) but their contents
    never appear here. ``upcard`` is populated only while it's actually a live public
    card (round-1 bidding and the discard step); once turned down it becomes ``None``
    and only ``turned_down_suit`` remains (that much is public knowledge in a real
    game -- everyone saw which suit was rejected).
    """

    perspective_agent: AgentId
    hand: tuple[Card, ...]
    hand_sizes: tuple[int, ...]
    upcard: Card | None
    turned_down_suit: Suit | None
    trump: Suit | None
    dealer: AgentId
    phase: Phase
    current_trick: tuple[tuple[AgentId, Card], ...]
    trick_number: int
    tricks_won: tuple[int, ...]
    maker: AgentId | None
    alone: bool
    sitting_out: AgentId | None
    to_act: AgentId
    legal_actions: ActionMask
    terminal: bool
    scoring_team: int | None
    points: int | None


def _beats(challenger: Card, incumbent: Card, trump: Suit, led_suit: Suit) -> bool:
    """Whether ``challenger`` beats ``incumbent`` as the best card so far in a trick.

    Standard trick-evaluation algorithm: compare each new card against the current
    best via one pairwise rule. Correct regardless of play order because "trump beats
    non-trump" and "higher trump beats lower trump" / "higher led-suit beats lower
    led-suit" are consistent total orders within their groups, and a card that is
    neither trump nor the suit led can never win a trick.
    """
    c_eff, i_eff = effective_suit(challenger, trump), effective_suit(incumbent, trump)
    c_is_trump, i_is_trump = c_eff == trump, i_eff == trump

    if c_is_trump and i_is_trump:
        return trump_rank(challenger, trump) > trump_rank(incumbent, trump)
    if c_is_trump:
        return True
    if i_is_trump:
        return False
    if c_eff == led_suit and i_eff == led_suit:
        return plain_rank(challenger) > plain_rank(incumbent)
    return False


class EuchreEngine:
    """Authoritative Euchre simulator: 4 agents, fixed partnerships, one hand/episode."""

    def __init__(self) -> None:
        self._state: EuchreState | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        rules: EuchreRules | None = None,
        dealer: int = 0,
    ) -> None:
        """Deal a fresh hand. ``dealer``/``rules`` are Euchre-specific extras beyond
        the base ``Engine`` protocol (both optional, defaulted) -- see
        plans/phase-04-euchre.md's "one hand = one episode" scoping note for why
        dealer rotation across hands is a match-level (not engine-level) concern."""
        self._state = EuchreState.new_game(seed=seed, rules=rules, dealer=dealer)

    def _require_state(self) -> EuchreState:
        if self._state is None:
            raise RuntimeError("EuchreEngine.reset() must be called before use")
        return self._state

    def agents(self) -> Sequence[AgentId]:
        return tuple(AgentId(i) for i in range(NUM_PLAYERS))

    def current_agent(self) -> AgentId:
        return AgentId(self._require_state().to_act)

    def is_terminal(self) -> bool:
        return self._require_state().terminal

    # -- masking ---------------------------------------------------------------

    def legal_actions(self, agent: AgentId) -> ActionMask:
        state = self._require_state()
        mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        if state.terminal or agent != state.to_act:
            return mask

        if state.phase == Phase.BID_ROUND_1:
            mask[PASS] = True
            mask[ORDER_UP] = True
            mask[ORDER_UP_ALONE] = True
        elif state.phase == Phase.BID_ROUND_2:
            excluded_suit = suit_of(state.upcard)
            for suit in Suit:
                if suit == excluded_suit:
                    continue
                mask[call_suit_action(suit)] = True
                mask[call_suit_alone_action(suit)] = True
            dealer_is_forced = (
                state.rules.stick_the_dealer
                and agent == state.dealer
                and state.bid_position == NUM_PLAYERS - 1
            )
            mask[PASS] = not dealer_is_forced
        elif state.phase == Phase.DEALER_DISCARD:
            for card in state.hands[agent]:
                mask[card] = True
        elif state.phase == Phase.TRICK_PLAY:
            for card in self._legal_trick_cards(state, agent):
                mask[card] = True
        return mask

    def _legal_trick_cards(self, state: EuchreState, agent: int) -> list[Card]:
        hand = state.hands[agent]
        if not state.current_trick or state.trump is None:
            return list(hand)
        led_suit = effective_suit(state.current_trick[0][1], state.trump)
        following = [c for c in hand if effective_suit(c, state.trump) == led_suit]
        return following if following else list(hand)

    # -- step --------------------------------------------------------------

    def step(self, agent: AgentId, action: Action) -> StepResult:
        state = self._require_state()
        if state.terminal:
            raise ValueError("cannot step: the hand has already ended")
        if agent != state.to_act:
            raise ValueError(f"it is agent {state.to_act}'s turn, agent {agent} cannot act")
        mask = self.legal_actions(agent)
        if not (0 <= action < NUM_ACTIONS) or not mask[action]:
            raise ValueError(f"illegal action {action} for agent {agent} in phase {state.phase}")

        events: list[Event] = [ActionTaken(agent=agent, action=int(action))]

        if state.phase == Phase.BID_ROUND_1:
            self._apply_bid_round_1(state, agent, action)
        elif state.phase == Phase.BID_ROUND_2:
            self._apply_bid_round_2(state, agent, action)
        elif state.phase == Phase.DEALER_DISCARD:
            self._apply_discard(state, agent, action)
        elif state.phase == Phase.TRICK_PLAY:
            self._apply_play(state, agent, action)

        if state.terminal:
            events.append(GameEnded(rewards=dict(self.rewards())))

        return StepResult(terminal=state.terminal, rewards=self.rewards(), events=tuple(events))

    def _apply_bid_round_1(self, state: EuchreState, agent: int, action: Action) -> None:
        if action == PASS:
            state.bid_position += 1
            if state.bid_position == NUM_PLAYERS:
                state.phase = Phase.BID_ROUND_2
                state.turned_down = True
                state.bid_position = 0
            state.to_act = (state.bid_start + state.bid_position) % NUM_PLAYERS
            return

        # ORDER_UP or ORDER_UP_ALONE: trump = upcard's suit, dealer picks it up.
        self._set_maker(state, agent, suit_of(state.upcard), alone=(action == ORDER_UP_ALONE))
        state.hands[state.dealer].append(state.upcard)
        state.phase = Phase.DEALER_DISCARD
        state.to_act = state.dealer

    def _apply_bid_round_2(self, state: EuchreState, agent: int, action: Action) -> None:
        if action == PASS:
            state.bid_position += 1
            if state.bid_position == NUM_PLAYERS:
                self._redeal(state)
                return
            state.to_act = (state.bid_start + state.bid_position) % NUM_PLAYERS
            return

        suit = suit_of_call_action(action)
        alone = is_call_suit_alone_action(action)
        self._set_maker(state, agent, suit, alone=alone)
        self._start_trick_play(state)

    def _set_maker(self, state: EuchreState, agent: int, trump: Suit, *, alone: bool) -> None:
        state.trump = trump
        state.maker = agent
        state.alone = alone
        state.sitting_out = partner_of(agent) if alone else None

    def _redeal(self, state: EuchreState) -> None:
        """All 4 passed round 2 with stick-the-dealer off: reshuffle and re-deal."""
        new_dealer = (state.dealer + 1) % NUM_PLAYERS
        hands, upcard = deal_hand(state.rng)
        bid_start = (new_dealer + 1) % NUM_PLAYERS
        state.hands = hands
        state.upcard = upcard
        state.dealer = new_dealer
        state.phase = Phase.BID_ROUND_1
        state.turned_down = False
        state.trump = None
        state.maker = None
        state.alone = False
        state.sitting_out = None
        state.bid_start = bid_start
        state.bid_position = 0
        state.to_act = bid_start
        state.redeals += 1

    def _apply_discard(self, state: EuchreState, agent: int, action: Action) -> None:
        state.hands[agent].remove(action)
        self._start_trick_play(state)

    def _start_trick_play(self, state: EuchreState) -> None:
        state.phase = Phase.TRICK_PLAY
        leader = (state.dealer + 1) % NUM_PLAYERS
        if leader == state.sitting_out:
            leader = state.next_trick_seat(leader)
        state.trick_leader = leader
        state.to_act = leader
        state.trick_number = 0
        state.current_trick = []
        state.tricks_won = [0, 0, 0, 0]

    def _apply_play(self, state: EuchreState, agent: int, action: Action) -> None:
        card = action
        state.hands[agent].remove(card)
        state.current_trick.append((agent, card))

        if len(state.current_trick) < state.active_player_count():
            state.to_act = state.next_trick_seat(agent)
            return

        winner = self._resolve_trick(state)
        state.tricks_won[winner] += 1
        state.trick_number += 1
        state.current_trick = []

        if state.trick_number == TRICKS_PER_HAND:
            self._score_hand(state)
        else:
            state.trick_leader = winner
            state.to_act = winner

    def _resolve_trick(self, state: EuchreState) -> int:
        assert state.trump is not None
        led_suit = effective_suit(state.current_trick[0][1], state.trump)
        best_agent, best_card = state.current_trick[0]
        for a, c in state.current_trick[1:]:
            if _beats(c, best_card, state.trump, led_suit):
                best_agent, best_card = a, c
        return best_agent

    def _score_hand(self, state: EuchreState) -> None:
        assert state.maker is not None
        maker_team = team_of(state.maker)
        maker_tricks = sum(t for i, t in enumerate(state.tricks_won) if team_of(i) == maker_team)

        if maker_tricks >= 3:
            scoring_team = maker_team
            if maker_tricks == TRICKS_PER_HAND:
                points = 4 if state.alone else 2
            else:
                points = 1
        else:
            scoring_team = 1 - maker_team
            points = 2

        state.scoring_team = scoring_team
        state.points = points
        state.terminal = True

    # -- observation / rewards --------------------------------------------------

    def observation(self, agent: AgentId) -> EuchreObservation:
        state = self._require_state()
        is_on_turn = (not state.terminal) and agent == state.to_act
        mask = self.legal_actions(agent) if is_on_turn else np.zeros(NUM_ACTIONS, dtype=np.bool_)

        upcard_visible = state.phase in (Phase.BID_ROUND_1, Phase.DEALER_DISCARD)
        return EuchreObservation(
            perspective_agent=agent,
            hand=tuple(sorted(state.hands[agent])),
            hand_sizes=tuple(len(h) for h in state.hands),
            upcard=state.upcard if upcard_visible else None,
            turned_down_suit=suit_of(state.upcard) if state.turned_down else None,
            trump=state.trump,
            dealer=AgentId(state.dealer),
            phase=state.phase,
            current_trick=tuple((AgentId(a), c) for a, c in state.current_trick),
            trick_number=state.trick_number,
            tricks_won=tuple(state.tricks_won),
            maker=AgentId(state.maker) if state.maker is not None else None,
            alone=state.alone,
            sitting_out=AgentId(state.sitting_out) if state.sitting_out is not None else None,
            to_act=AgentId(state.to_act),
            legal_actions=mask,
            terminal=state.terminal,
            scoring_team=state.scoring_team,
            points=state.points if state.terminal else None,
        )

    def rewards(self) -> Mapping[AgentId, float]:
        state = self._require_state()
        if not state.terminal or state.scoring_team is None:
            return {AgentId(i): 0.0 for i in range(NUM_PLAYERS)}
        rewards = {}
        for i in range(NUM_PLAYERS):
            sign = 1.0 if team_of(i) == state.scoring_team else -1.0
            rewards[AgentId(i)] = sign * state.points
        return rewards
