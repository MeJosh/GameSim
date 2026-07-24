"""Euchre engine tests -- plan groups A-K. See plans/phase-04-euchre.md for the spec.

Several tests (trick-winner logic, going-alone scoring, the isolated scoring-formula
checks) construct ``EuchreState`` fields directly after a normal ``reset()`` rather
than driving every card through bidding first. This is the Euchre analogue of Connect
Four's hardcoded verified-draw fixture (see test_connect_four.py): dealt hands are
random, so pinning down an exact bower/trick/going-alone scenario via real play would
mean fighting the RNG instead of testing the rule. The bidding/discard/masking state
machine itself (groups C-E) is driven through ``step()`` for real, since that's exactly
what needs exercising end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

from gamesim.core.agent import Agent, RandomAgent
from gamesim.core.runner import run_game
from gamesim.core.types import ActionMask, AgentId
from gamesim.games.euchre import EuchreEngine, EuchreObservation, EuchreRules, Phase, Suit
from gamesim.games.euchre.actions import (
    NUM_ACTIONS,
    ORDER_UP,
    ORDER_UP_ALONE,
    PASS,
    Action,
    call_suit_action,
    call_suit_alone_action,
)
from gamesim.games.euchre.cards import (
    Rank,
    effective_suit,
    full_deck,
    make_card,
    plain_rank,
    same_color_suit,
    suit_of,
    trump_rank,
)
from gamesim.games.euchre.state import partner_of


def new_engine(
    seed: int | None = 0, dealer: int = 0, rules: EuchreRules | None = None
) -> EuchreEngine:
    engine = EuchreEngine()
    engine.reset(seed=seed, dealer=dealer, rules=rules)
    return engine


# --- A. Cards & bower logic ---------------------------------------------------------


def test_deck_has_24_unique_cards() -> None:
    deck = full_deck()
    assert len(deck) == 24
    assert len(set(deck)) == 24


def test_left_bower_effective_suit_is_trump() -> None:
    left_bower = make_card(Suit.DIAMONDS, Rank.JACK)  # trump=hearts -> diamonds' jack
    assert effective_suit(left_bower, Suit.HEARTS) == Suit.HEARTS
    assert suit_of(left_bower) == Suit.DIAMONDS  # printed suit unchanged


def test_trump_ranking_order() -> None:
    trump = Suit.HEARTS
    right_bower = make_card(Suit.HEARTS, Rank.JACK)
    left_bower = make_card(Suit.DIAMONDS, Rank.JACK)
    ace = make_card(Suit.HEARTS, Rank.ACE)
    king = make_card(Suit.HEARTS, Rank.KING)
    queen = make_card(Suit.HEARTS, Rank.QUEEN)
    ten = make_card(Suit.HEARTS, Rank.TEN)
    nine = make_card(Suit.HEARTS, Rank.NINE)
    ranks = [trump_rank(c, trump) for c in (right_bower, left_bower, ace, king, queen, ten, nine)]
    assert ranks == sorted(ranks, reverse=True)
    assert len(set(ranks)) == 7  # strictly decreasing, no ties


def test_off_color_suit_missing_its_jack_ranks_without_it() -> None:
    # trump=hearts -> diamonds' jack becomes the left bower, so a *plain* diamonds
    # hand only ever has 9/10/Q/K/A; ranking among those follows plain_rank directly.
    trump = Suit.HEARTS
    diamond_ace = make_card(Suit.DIAMONDS, Rank.ACE)
    diamond_king = make_card(Suit.DIAMONDS, Rank.KING)
    assert effective_suit(diamond_ace, trump) == Suit.DIAMONDS
    assert plain_rank(diamond_ace) > plain_rank(diamond_king)
    # An unaffected suit (spades, unrelated color) keeps its own jack as a plain card.
    spade_jack = make_card(Suit.SPADES, Rank.JACK)
    assert effective_suit(spade_jack, trump) == Suit.SPADES


@pytest.mark.parametrize("trump", [Suit.HEARTS, Suit.DIAMONDS, Suit.SPADES, Suit.CLUBS])
def test_bower_rule_symmetric_for_every_trump_suit(trump: Suit) -> None:
    partner_suit = same_color_suit(trump)
    assert partner_suit != trump
    assert same_color_suit(partner_suit) == trump  # symmetric pairing
    left_bower = make_card(partner_suit, Rank.JACK)
    right_bower = make_card(trump, Rank.JACK)
    assert effective_suit(left_bower, trump) == trump
    assert effective_suit(right_bower, trump) == trump
    assert trump_rank(right_bower, trump) > trump_rank(left_bower, trump)


# --- B. Deal & construction ----------------------------------------------------------


def test_fresh_hand_deals_five_cards_each_and_a_disjoint_upcard() -> None:
    engine = new_engine()
    state = engine._state
    assert state is not None
    assert [len(h) for h in state.hands] == [5, 5, 5, 5]
    dealt = {c for h in state.hands for c in h}
    assert len(dealt) == 20
    assert state.upcard not in dealt


def test_agents_and_first_bidder_is_left_of_dealer() -> None:
    engine = new_engine(dealer=2)
    assert list(engine.agents()) == [AgentId(i) for i in range(4)]
    assert engine.current_agent() == AgentId(3)


def test_fresh_hand_not_terminal_and_zero_rewards() -> None:
    engine = new_engine()
    assert engine.is_terminal() is False
    rewards = engine.rewards()
    assert all(rewards[AgentId(i)] == 0 for i in range(4))


# --- C. Round-1 bidding (order-up) ----------------------------------------------------


def test_round1_mask_offers_exactly_pass_order_up_order_up_alone() -> None:
    engine = new_engine()
    mask = engine.legal_actions(engine.current_agent())
    assert mask.sum() == 3
    assert mask[PASS] and mask[ORDER_UP] and mask[ORDER_UP_ALONE]


def test_order_up_sets_trump_and_transitions_to_discard() -> None:
    engine = new_engine(dealer=0)
    bidder = engine.current_agent()
    state = engine._state
    assert state is not None
    upcard_suit = suit_of(state.upcard)
    engine.step(bidder, ORDER_UP)
    assert state.phase == Phase.DEALER_DISCARD
    assert state.trump == upcard_suit
    assert state.maker == bidder
    assert len(state.hands[0]) == 6  # dealer picked up the upcard
    assert engine.current_agent() == AgentId(0)


def test_order_up_alone_marks_partner_sitting_out() -> None:
    engine = new_engine(dealer=0)
    bidder = engine.current_agent()
    engine.step(bidder, ORDER_UP_ALONE)
    state = engine._state
    assert state is not None
    assert state.alone is True
    assert state.sitting_out == partner_of(bidder)


def test_dealer_orders_up_alone_on_own_upcard() -> None:
    """Edge case flagged in review: the dealer can order up (and go alone on) their
    own turned-up card, since they're the last to act in round 1 too."""
    engine = new_engine(dealer=0)
    for _ in range(3):  # agents 1, 2, 3 pass; agent 0 (the dealer) is left
        engine.step(engine.current_agent(), PASS)
    assert engine.current_agent() == AgentId(0)
    state = engine._state
    assert state is not None
    upcard_suit = suit_of(state.upcard)
    engine.step(AgentId(0), ORDER_UP_ALONE)
    assert state.trump == upcard_suit
    assert state.maker == 0
    assert state.alone is True
    assert state.sitting_out == partner_of(0)
    assert state.phase == Phase.DEALER_DISCARD
    assert engine.current_agent() == AgentId(0)  # dealer discards their own pickup
    assert len(state.hands[0]) == 6


def test_dealer_still_discards_when_sitting_out_as_makers_partner() -> None:
    """Edge case flagged in review: if the maker's partner *is* the dealer, the
    dealer must still pick up and discard (only trick play skips them), and trick
    play correctly skips them afterwards."""
    engine = new_engine(dealer=2)  # dealer's partner is agent 0
    engine.step(engine.current_agent(), PASS)  # agent 3 passes
    assert engine.current_agent() == AgentId(0)
    engine.step(AgentId(0), ORDER_UP_ALONE)
    state = engine._state
    assert state is not None
    assert state.sitting_out == 2
    assert state.sitting_out == state.dealer  # the sitting-out partner is the dealer
    assert state.phase == Phase.DEALER_DISCARD
    assert engine.current_agent() == AgentId(2)  # dealer discards despite sitting out
    assert len(state.hands[2]) == 6

    discard = state.hands[2][0]
    engine.step(AgentId(2), discard)
    state = engine._state  # re-bind: resets mypy's narrowing of the mutated phase
    assert state is not None
    assert state.phase == Phase.TRICK_PLAY
    assert engine.current_agent() != AgentId(2)  # now correctly skipped as sitting-out


def test_all_pass_round1_moves_to_round2() -> None:
    engine = new_engine(dealer=0)
    for _ in range(4):
        engine.step(engine.current_agent(), PASS)
    state = engine._state
    assert state is not None
    assert state.phase == Phase.BID_ROUND_2
    assert state.turned_down is True
    assert state.trump is None
    assert engine.current_agent() == AgentId(1)  # bidding restarts left of dealer


# --- D. Round-2 bidding (name trump) + stick the dealer --------------------------------


def _pass_round1(engine: EuchreEngine) -> None:
    for _ in range(4):
        engine.step(engine.current_agent(), PASS)


def test_round2_mask_excludes_turned_down_suit_offers_others_and_pass() -> None:
    engine = new_engine(dealer=0)
    state = engine._state
    assert state is not None
    excluded = suit_of(state.upcard)
    _pass_round1(engine)
    mask = engine.legal_actions(engine.current_agent())
    assert mask[PASS]
    for suit in Suit:
        expected = suit != excluded
        assert mask[call_suit_action(suit)] == expected
        assert mask[call_suit_alone_action(suit)] == expected


def test_calling_suit_round2_sets_trump_and_maker_without_discard() -> None:
    engine = new_engine(dealer=0)
    state = engine._state
    assert state is not None
    excluded = suit_of(state.upcard)
    called = next(s for s in Suit if s != excluded)
    _pass_round1(engine)
    bidder = engine.current_agent()
    dealer_hand_before = len(state.hands[state.dealer])
    engine.step(bidder, call_suit_action(called))
    assert state.trump == called
    assert state.maker == bidder
    assert state.phase == Phase.TRICK_PLAY  # straight to play, no discard
    assert len(state.hands[state.dealer]) == dealer_hand_before


def test_stick_the_dealer_forces_call_on_dealers_last_turn() -> None:
    engine = new_engine(dealer=0)
    _pass_round1(engine)
    for _ in range(3):
        engine.step(engine.current_agent(), PASS)
    assert engine.current_agent() == AgentId(0)  # dealer, last to act
    mask = engine.legal_actions(AgentId(0))
    assert not mask[PASS]
    assert mask.sum() > 0  # some suit must be callable


def test_redeal_when_stick_the_dealer_disabled_and_all_pass_round2() -> None:
    engine = new_engine(dealer=0, rules=EuchreRules(stick_the_dealer=False))
    _pass_round1(engine)
    for _ in range(4):
        mask = engine.legal_actions(engine.current_agent())
        assert mask[PASS]  # dealer is never forced under this rule set
        engine.step(engine.current_agent(), PASS)
    state = engine._state
    assert state is not None
    assert state.redeals == 1
    assert state.dealer == 1
    assert state.phase == Phase.BID_ROUND_1
    assert state.trump is None
    assert [len(h) for h in state.hands] == [5, 5, 5, 5]
    assert engine.current_agent() == AgentId(2)  # left of the new dealer


# --- E. Dealer discard ----------------------------------------------------------------


def test_discard_mask_is_exactly_dealers_current_hand() -> None:
    engine = new_engine(dealer=0)
    engine.step(engine.current_agent(), ORDER_UP)
    state = engine._state
    assert state is not None
    mask = engine.legal_actions(AgentId(0))
    legal_cards = {i for i in range(NUM_ACTIONS) if mask[i]}
    assert legal_cards == set(state.hands[0])
    assert len(legal_cards) == 6


def test_discard_returns_to_five_and_starts_trick_play() -> None:
    engine = new_engine(dealer=0)
    engine.step(engine.current_agent(), ORDER_UP)
    state = engine._state
    assert state is not None
    discard = state.hands[0][0]
    engine.step(AgentId(0), discard)
    assert len(state.hands[0]) == 5
    assert state.phase == Phase.TRICK_PLAY
    assert engine.current_agent() == AgentId(1)  # left of dealer leads


# --- F. Trick play: follow-suit masking -------------------------------------------------


def _setup_trick_play(
    engine: EuchreEngine,
    *,
    trump: Suit,
    leader: int,
    hands: dict[int, list[int]],
    maker: int | None = None,
    alone: bool = False,
    sitting_out: int | None = None,
) -> None:
    """White-box scenario setup: jump straight into TRICK_PLAY with chosen hands.

    See the module docstring for why: dealt hands are random, and these tests are
    about trick/follow-suit logic, not the bidding state machine (covered above).
    """
    state = engine._state
    assert state is not None
    state.hands = [list(hands.get(i, [])) for i in range(4)]
    state.trump = trump
    state.maker = maker if maker is not None else leader
    state.alone = alone
    state.sitting_out = sitting_out
    state.phase = Phase.TRICK_PLAY
    state.trick_leader = leader
    state.to_act = leader
    state.trick_number = 0
    state.current_trick = []
    state.tricks_won = [0, 0, 0, 0]


def test_must_follow_led_suit_when_holding_it() -> None:
    engine = new_engine()
    king_hearts = make_card(Suit.HEARTS, Rank.KING)
    ace_hearts = make_card(Suit.HEARTS, Rank.ACE)
    nine_clubs = make_card(Suit.CLUBS, Rank.NINE)
    ten_spades = make_card(Suit.SPADES, Rank.TEN)
    _setup_trick_play(
        engine,
        trump=Suit.SPADES,
        leader=0,
        hands={0: [king_hearts], 1: [ace_hearts, nine_clubs, ten_spades]},
    )
    engine.step(AgentId(0), king_hearts)
    mask = engine.legal_actions(AgentId(1))
    legal = {i for i in range(NUM_ACTIONS) if mask[i]}
    assert legal == {ace_hearts}


def test_may_play_any_card_when_void_in_led_suit() -> None:
    engine = new_engine()
    king_hearts = make_card(Suit.HEARTS, Rank.KING)
    nine_clubs = make_card(Suit.CLUBS, Rank.NINE)
    ten_diamonds = make_card(Suit.DIAMONDS, Rank.TEN)
    queen_spades = make_card(Suit.SPADES, Rank.QUEEN)
    _setup_trick_play(
        engine,
        trump=Suit.SPADES,
        leader=0,
        hands={0: [king_hearts], 1: [nine_clubs, ten_diamonds, queen_spades]},
    )
    engine.step(AgentId(0), king_hearts)
    mask = engine.legal_actions(AgentId(1))
    legal = {i for i in range(NUM_ACTIONS) if mask[i]}
    assert legal == {nine_clubs, ten_diamonds, queen_spades}


def test_left_bower_is_legal_and_required_when_trump_is_led() -> None:
    trump = Suit.HEARTS
    right_bower = make_card(Suit.HEARTS, Rank.JACK)
    left_bower = make_card(Suit.DIAMONDS, Rank.JACK)
    nine_clubs = make_card(Suit.CLUBS, Rank.NINE)
    ten_spades = make_card(Suit.SPADES, Rank.TEN)
    engine = new_engine()
    _setup_trick_play(
        engine,
        trump=trump,
        leader=0,
        hands={0: [right_bower], 1: [left_bower, nine_clubs, ten_spades]},
    )
    engine.step(AgentId(0), right_bower)  # leads trump
    mask = engine.legal_actions(AgentId(1))
    legal = {i for i in range(NUM_ACTIONS) if mask[i]}
    assert legal == {left_bower}  # must follow trump; left bower counts as trump


# --- G. Trick play: winner determination ------------------------------------------------


def _play_out_trick(engine: EuchreEngine, order: list[int], cards: dict[int, int]) -> None:
    for agent in order:
        engine.step(AgentId(agent), cards[agent])


def test_right_bower_beats_everything() -> None:
    trump = Suit.HEARTS
    ace_hearts = make_card(Suit.HEARTS, Rank.ACE)
    right_bower = make_card(Suit.HEARTS, Rank.JACK)
    left_bower = make_card(Suit.DIAMONDS, Rank.JACK)
    king_hearts = make_card(Suit.HEARTS, Rank.KING)
    engine = new_engine()
    hands = {0: [ace_hearts], 1: [right_bower], 2: [left_bower], 3: [king_hearts]}
    _setup_trick_play(engine, trump=trump, leader=0, hands=hands)
    _play_out_trick(
        engine, [0, 1, 2, 3], {0: ace_hearts, 1: right_bower, 2: left_bower, 3: king_hearts}
    )
    state = engine._state
    assert state is not None
    assert state.trick_leader == 1  # winner leads the next trick
    assert state.tricks_won == [0, 1, 0, 0]


def test_left_bower_beats_all_trump_except_right_bower() -> None:
    trump = Suit.HEARTS
    king_hearts = make_card(Suit.HEARTS, Rank.KING)
    left_bower = make_card(Suit.DIAMONDS, Rank.JACK)
    ace_hearts = make_card(Suit.HEARTS, Rank.ACE)
    nine_hearts = make_card(Suit.HEARTS, Rank.NINE)
    engine = new_engine()
    hands = {0: [king_hearts], 1: [left_bower], 2: [ace_hearts], 3: [nine_hearts]}
    _setup_trick_play(engine, trump=trump, leader=0, hands=hands)
    _play_out_trick(
        engine, [0, 1, 2, 3], {0: king_hearts, 1: left_bower, 2: ace_hearts, 3: nine_hearts}
    )
    state = engine._state
    assert state is not None
    assert state.trick_leader == 1


def test_highest_led_suit_wins_when_no_trump_played() -> None:
    trump = Suit.SPADES
    king_hearts = make_card(Suit.HEARTS, Rank.KING)
    nine_clubs = make_card(Suit.CLUBS, Rank.NINE)
    ace_hearts = make_card(Suit.HEARTS, Rank.ACE)
    queen_hearts = make_card(Suit.HEARTS, Rank.QUEEN)
    engine = new_engine()
    hands = {0: [king_hearts], 1: [nine_clubs], 2: [ace_hearts], 3: [queen_hearts]}
    _setup_trick_play(engine, trump=trump, leader=0, hands=hands)
    _play_out_trick(
        engine, [0, 1, 2, 3], {0: king_hearts, 1: nine_clubs, 2: ace_hearts, 3: queen_hearts}
    )
    state = engine._state
    assert state is not None
    assert state.trick_leader == 2  # ace of hearts, the led suit


def test_trump_played_off_lead_still_wins() -> None:
    trump = Suit.CLUBS
    ace_hearts = make_card(Suit.HEARTS, Rank.ACE)
    nine_clubs = make_card(Suit.CLUBS, Rank.NINE)  # low trump
    king_hearts = make_card(Suit.HEARTS, Rank.KING)
    queen_hearts = make_card(Suit.HEARTS, Rank.QUEEN)
    engine = new_engine()
    hands = {0: [ace_hearts], 1: [nine_clubs], 2: [king_hearts], 3: [queen_hearts]}
    _setup_trick_play(engine, trump=trump, leader=0, hands=hands)
    _play_out_trick(
        engine, [0, 1, 2, 3], {0: ace_hearts, 1: nine_clubs, 2: king_hearts, 3: queen_hearts}
    )
    state = engine._state
    assert state is not None
    assert state.trick_leader == 1  # even a low trump beats a non-trump ace


# --- H. Going alone --------------------------------------------------------------------


def _drive_to_terminal(engine: EuchreEngine, seed: int) -> list[int]:
    """Play out the rest of the hand with random legal actions; return agents seen."""
    agent_rng: RandomAgent[EuchreObservation] = RandomAgent(seed=seed)
    seen: list[int] = []
    while not engine.is_terminal():
        current = engine.current_agent()
        seen.append(int(current))
        mask = engine.legal_actions(current)
        action = agent_rng.act(engine.observation(current), mask)
        engine.step(current, action)
    return seen


def test_sitting_out_partner_never_acts_and_lone_march_scores_four() -> None:
    trump = Suit.SPADES
    maker_cards = [
        make_card(Suit.SPADES, Rank.JACK),  # right bower
        make_card(Suit.CLUBS, Rank.JACK),  # left bower
        make_card(Suit.SPADES, Rank.ACE),
        make_card(Suit.SPADES, Rank.KING),
        make_card(Suit.SPADES, Rank.QUEEN),
    ]
    remaining = [c for c in full_deck() if c not in maker_cards]
    hands = {
        0: maker_cards,
        1: remaining[0:5],
        2: remaining[5:10],
        3: remaining[10:15],
    }
    engine = new_engine(dealer=3)  # leader = (3+1)%4 = 0 = maker
    _setup_trick_play(
        engine, trump=trump, leader=0, hands=hands, maker=0, alone=True, sitting_out=2
    )

    seen = _drive_to_terminal(engine, seed=7)

    assert 2 not in seen  # the sitting-out partner is never asked to act
    state = engine._state
    assert state is not None
    assert state.tricks_won[0] == 5
    rewards = engine.rewards()
    assert rewards[AgentId(0)] == 4.0
    assert rewards[AgentId(2)] == 4.0
    assert rewards[AgentId(1)] == -4.0
    assert rewards[AgentId(3)] == -4.0


def test_lone_hand_euchred_still_costs_defenders_standard_two() -> None:
    trump = Suit.SPADES
    # Give the *defender* (agent 1) the unbeatable cards this time, so the lone
    # maker (agent 0) cannot take 3 tricks.
    dominant_cards = [
        make_card(Suit.SPADES, Rank.JACK),
        make_card(Suit.CLUBS, Rank.JACK),
        make_card(Suit.SPADES, Rank.ACE),
        make_card(Suit.SPADES, Rank.KING),
        make_card(Suit.SPADES, Rank.QUEEN),
    ]
    remaining = [c for c in full_deck() if c not in dominant_cards]
    hands = {
        0: remaining[0:5],
        1: dominant_cards,
        2: [],  # sitting out: no cards needed
        3: remaining[5:10],
    }
    engine = new_engine(dealer=3)
    _setup_trick_play(
        engine, trump=trump, leader=0, hands=hands, maker=0, alone=True, sitting_out=2
    )

    _drive_to_terminal(engine, seed=11)

    state = engine._state
    assert state is not None
    assert state.tricks_won[0] == 0  # maker took nothing
    rewards = engine.rewards()
    assert rewards[AgentId(1)] == 2.0
    assert rewards[AgentId(3)] == 2.0
    assert rewards[AgentId(0)] == -2.0  # euchre penalty, not lone-march-sized
    assert rewards[AgentId(2)] == -2.0


# --- I. Scoring & terminal ---------------------------------------------------------------


def _score(engine: EuchreEngine, *, maker: int, alone: bool, tricks_won: list[int]) -> None:
    state = engine._state
    assert state is not None
    state.maker = maker
    state.alone = alone
    state.tricks_won = tricks_won
    engine._score_hand(state)


def test_partnered_three_tricks_scores_one() -> None:
    engine = new_engine()
    _score(engine, maker=0, alone=False, tricks_won=[2, 1, 1, 1])  # team0 = 3
    state = engine._state
    assert state is not None
    assert state.scoring_team == 0
    assert state.points == 1
    rewards = engine.rewards()
    assert rewards[AgentId(0)] == 1.0 and rewards[AgentId(2)] == 1.0
    assert rewards[AgentId(1)] == -1.0 and rewards[AgentId(3)] == -1.0


def test_partnered_four_tricks_scores_one() -> None:
    engine = new_engine()
    _score(engine, maker=0, alone=False, tricks_won=[3, 0, 1, 1])  # team0 = 4
    state = engine._state
    assert state is not None
    assert state.scoring_team == 0
    assert state.points == 1  # 4 tricks is still 1 point -- only a clean 5-0 marches


def test_partnered_march_scores_two() -> None:
    engine = new_engine()
    _score(engine, maker=1, alone=False, tricks_won=[0, 3, 0, 2])  # team1 = 5
    state = engine._state
    assert state is not None
    assert state.scoring_team == 1
    assert state.points == 2


def test_euchre_scores_two_to_defenders() -> None:
    engine = new_engine()
    _score(engine, maker=0, alone=False, tricks_won=[1, 2, 1, 1])  # team0 = 2 < 3
    state = engine._state
    assert state is not None
    assert state.scoring_team == 1
    assert state.points == 2
    rewards = engine.rewards()
    assert rewards[AgentId(0)] == -2.0 and rewards[AgentId(2)] == -2.0
    assert rewards[AgentId(1)] == 2.0 and rewards[AgentId(3)] == 2.0


def test_terminal_after_scoring_step_raises_and_masks_all_false() -> None:
    engine = new_engine()
    _score(engine, maker=0, alone=False, tricks_won=[3, 0, 1, 1])
    assert engine.is_terminal()
    for i in range(4):
        assert not np.any(engine.legal_actions(AgentId(i)))
    with pytest.raises(ValueError):
        engine.step(AgentId(0), PASS)


# --- J. Observation boundary -------------------------------------------------------------


def test_observation_hides_other_hands_shows_own() -> None:
    engine = new_engine()
    state = engine._state
    assert state is not None
    obs0 = engine.observation(AgentId(0))
    assert set(obs0.hand) == set(state.hands[0])
    assert obs0.hand_sizes == (5, 5, 5, 5)


def test_upcard_hidden_after_turned_down_but_suit_stays_public() -> None:
    engine = new_engine(dealer=0)
    state = engine._state
    assert state is not None
    original_suit = suit_of(state.upcard)
    _pass_round1(engine)
    obs = engine.observation(engine.current_agent())
    assert obs.upcard is None
    assert obs.turned_down_suit == original_suit


def test_perspective_agent_is_honored_and_hands_are_disjoint() -> None:
    engine = new_engine()
    obs0 = engine.observation(AgentId(0))
    obs1 = engine.observation(AgentId(1))
    assert obs0.perspective_agent == AgentId(0)
    assert obs1.perspective_agent == AgentId(1)
    assert set(obs0.hand).isdisjoint(set(obs1.hand))


# --- K. Determinism & the Runner ----------------------------------------------------------


def test_same_seed_same_deal() -> None:
    a, b = new_engine(seed=42), new_engine(seed=42)
    state_a, state_b = a._state, b._state
    assert state_a is not None and state_b is not None
    assert state_a.hands == state_b.hands
    assert state_a.upcard == state_b.upcard


@pytest.mark.parametrize("seed", range(10))
def test_random_agents_play_full_hand_via_runner(seed: int) -> None:
    engine = EuchreEngine()
    agents: dict[AgentId, Agent[EuchreObservation, Action]] = {
        AgentId(i): RandomAgent(seed=seed * 10 + i) for i in range(4)
    }
    rewards = run_game(engine, agents, seed=seed)
    assert rewards[AgentId(0)] == rewards[AgentId(2)]
    assert rewards[AgentId(1)] == rewards[AgentId(3)]
    assert rewards[AgentId(0)] == -rewards[AgentId(1)]
    assert rewards[AgentId(0)] != 0


class _PassBiasedAgent:
    """Passes with high probability during bidding, else picks a random legal action.

    Under uniform-random play, all-pass-twice (the redeal trigger) is astronomically
    rare -- roughly (1/3)^4 * (1/7)^4 per hand -- so a plain ``RandomAgent`` fuzz run
    essentially never exercises the redeal path. Biasing toward pass makes it likely
    within a handful of hands while still using real per-turn randomness (not a
    scripted action sequence), which is a meaningfully different check than the one
    scripted redeal unit test above.
    """

    def __init__(self, seed: int, pass_bias: float = 0.9) -> None:
        self._rng = np.random.default_rng(seed)
        self._pass_bias = pass_bias

    def act(self, observation: EuchreObservation, mask: ActionMask) -> int:
        legal = np.flatnonzero(mask)
        if mask[PASS] and self._rng.random() < self._pass_bias:
            return PASS
        return int(self._rng.choice(legal))


def test_random_play_survives_and_actually_exercises_redeal_when_stick_the_dealer_off() -> None:
    """Review nit: the redeal path had only one dedicated unit test and never fired
    during the (default-rules) random-agent fuzz runs above. This drives many hands
    with stick-the-dealer disabled and a pass-biased (but still randomized) policy,
    and asserts a redeal actually happened at least once -- not just that the code
    path exists in theory."""
    redeal_seen = False
    for seed in range(20):
        engine = EuchreEngine()
        engine.reset(seed=seed, rules=EuchreRules(stick_the_dealer=False))
        agent = _PassBiasedAgent(seed=seed)
        steps = 0
        while not engine.is_terminal():
            steps += 1
            assert steps < 10_000, "runaway hand -- redeal loop likely never terminating"
            current = engine.current_agent()
            mask = engine.legal_actions(current)
            action = agent.act(engine.observation(current), mask)
            engine.step(current, action)
        state = engine._state
        assert state is not None
        redeal_seen = redeal_seen or state.redeals > 0
        rewards = engine.rewards()
        assert rewards[AgentId(0)] == rewards[AgentId(2)]
        assert rewards[AgentId(1)] == rewards[AgentId(3)]
    assert redeal_seen, "expected at least one redeal across 20 seeds -- fixture may be stale"


def test_same_seed_same_actions_same_final_state() -> None:
    def play(seed: int) -> tuple[float, float]:
        engine = EuchreEngine()
        agents: dict[AgentId, Agent[EuchreObservation, Action]] = {
            AgentId(i): RandomAgent(seed=seed * 10 + i) for i in range(4)
        }
        rewards = run_game(engine, agents, seed=seed)
        return rewards[AgentId(0)], rewards[AgentId(1)]

    assert play(99) == play(99)
