"""Euchre hand state.

A plain, typed mutable state object -- no ECS, same precedent as Connect Four (see
``connect_four/state.py`` and docs/architecture.md: ECS is reserved for games whose
complexity justifies it, and one hand of Euchre doesn't).

**Scope reminder** (see plans/phase-04-euchre.md): one ``EuchreState`` is one *hand*,
not a full first-to-10 match. ``to_act`` is the single source of truth for whose turn
it is across every phase (bidding, discard, trick play) -- the engine updates it
directly rather than re-deriving it, mirroring how ``ConnectFourState.current_agent_
index`` works but generalized past strict alternation (see docs/adr/0002).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from .cards import Card, Suit, full_deck, suit_of

NUM_PLAYERS = 4
CARDS_PER_HAND = 5
TRICKS_PER_HAND = 5


class Phase(Enum):
    BID_ROUND_1 = auto()
    BID_ROUND_2 = auto()
    DEALER_DISCARD = auto()
    TRICK_PLAY = auto()


@dataclass(frozen=True)
class EuchreRules:
    """Configurable rule variants. Only ``stick_the_dealer`` is exposed for now.

    See plans/phase-04-euchre.md: the user confirmed standard rules + stick-the-dealer
    as the baseline. The flag exists (rather than hardcoding stick-the-dealer=True) so
    the redeal path (round 2 all-pass with stick-the-dealer *off*) is representable and
    testable, not because both variants are equally supported end-to-end yet.
    """

    stick_the_dealer: bool = True


def partner_of(agent: int) -> int:
    """The agent seated across the table (fixed partnerships, see ADR 0002)."""
    return (agent + 2) % NUM_PLAYERS


def team_of(agent: int) -> int:
    """0 or 1: agents 0&2 are team 0, agents 1&3 are team 1 (seating order)."""
    return agent % 2


@dataclass
class EuchreState:
    """Mutable state for one Euchre hand in progress."""

    hands: list[list[Card]]
    dealer: int
    to_act: int
    phase: Phase
    upcard: Card
    turned_down: bool
    trump: Suit | None
    maker: int | None
    alone: bool
    sitting_out: int | None
    bid_start: int
    bid_position: int
    trick_leader: int
    current_trick: list[tuple[int, Card]]
    tricks_won: list[int]
    trick_number: int
    terminal: bool
    scoring_team: int | None
    points: int
    rules: EuchreRules
    seed: int | None
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    redeals: int = 0

    @classmethod
    def new_game(
        cls,
        *,
        seed: int | None = None,
        rules: EuchreRules | None = None,
        dealer: int = 0,
    ) -> EuchreState:
        """Deal a fresh hand. ``dealer`` is which seat deals first (default 0)."""
        rng = np.random.default_rng(seed)
        hands, upcard = deal_hand(rng)
        bid_start = (dealer + 1) % NUM_PLAYERS
        return cls(
            hands=hands,
            dealer=dealer,
            to_act=bid_start,
            phase=Phase.BID_ROUND_1,
            upcard=upcard,
            turned_down=False,
            trump=None,
            maker=None,
            alone=False,
            sitting_out=None,
            bid_start=bid_start,
            bid_position=0,
            trick_leader=bid_start,
            current_trick=[],
            tricks_won=[0, 0, 0, 0],
            trick_number=0,
            terminal=False,
            scoring_team=None,
            points=0,
            rules=rules if rules is not None else EuchreRules(),
            seed=seed,
            rng=rng,
        )

    def active_player_count(self) -> int:
        """3 if the maker went alone (partner sits out), else 4."""
        return 3 if self.alone else 4

    def next_trick_seat(self, agent: int) -> int:
        """The next seat after ``agent`` in trick play, skipping a sitting-out partner."""
        nxt = (agent + 1) % NUM_PLAYERS
        while nxt == self.sitting_out:
            nxt = (nxt + 1) % NUM_PLAYERS
        return nxt


def deal_hand(rng: np.random.Generator) -> tuple[list[list[Card]], Card]:
    """Shuffle the 24-card deck, deal 5 cards to each of 4 seats, turn up 1 more.

    3 cards remain in the kitty, undealt (24 - 4*5 - 1 = 3) -- unused this hand,
    consistent with a real Euchre pack. Deal order (which seat gets which block) has
    no rules significance and no fairness implication under a uniform shuffle, so
    seats are filled in index order (0, 1, 2, 3) rather than physically simulating
    left-of-dealer dealing order.
    """
    # ``rng.permutation`` returns a numpy array (elements are ``np.int64``, not
    # plain ``int``) -- cast explicitly so ``Card`` (== ``int``) is a real Python
    # int everywhere downstream, not a numpy scalar that happens to duck-type as
    # one. Without this, values compare/hash/step correctly (numpy ints behave
    # like ints for arithmetic and equality) but silently fail ``json.dumps`` the
    # first time something needs to serialize a hand -- e.g. the match-report
    # snapshots in ``analysis.replay_euchre``.
    deck = [int(card) for card in rng.permutation(full_deck())]
    hands = [list(deck[i * CARDS_PER_HAND : (i + 1) * CARDS_PER_HAND]) for i in range(NUM_PLAYERS)]
    upcard = deck[NUM_PLAYERS * CARDS_PER_HAND]
    return hands, upcard


# Re-exported for convenience at call sites that only need the suit, not the state.
def upcard_suit(upcard: Card) -> Suit:
    return suit_of(upcard)
