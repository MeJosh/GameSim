"""Euchre cards: encoding, the deck, and bower-aware suit/rank logic.

A card is a plain ``int`` (like Connect Four's ``Action`` -- see
``connect_four/actions.py``): ``card_id = suit * RANKS_PER_SUIT + rank``. Suits and
ranks are ``IntEnum`` so arithmetic stays cheap while call sites stay readable.

The one piece of genuinely new logic Euchre needs beyond "compare two ints" is the
**bower rule**: the jack of the trump suit (the *right bower*) and the jack of the
same-color suit (the *left bower*) are both trump, ranked above every other trump card,
and the left bower is treated as trump -- not as a member of its printed suit -- for
both follow-suit and trick-winner comparison. Centralizing that here means the engine
and any future encoder/renderer share one source of truth instead of re-deriving it.
"""

from __future__ import annotations

from enum import IntEnum

Card = int


class Suit(IntEnum):
    SPADES = 0
    HEARTS = 1
    DIAMONDS = 2
    CLUBS = 3


class Rank(IntEnum):
    NINE = 0
    TEN = 1
    JACK = 2
    QUEEN = 3
    KING = 4
    ACE = 5


RANKS_PER_SUIT = len(Rank)
NUM_CARDS = len(Suit) * RANKS_PER_SUIT  # 24

# Color pairing that determines the left bower: hearts<->diamonds (red), spades<->clubs
# (black). Each suit's partner is the *other* suit of the same color.
_SAME_COLOR: dict[Suit, Suit] = {
    Suit.SPADES: Suit.CLUBS,
    Suit.CLUBS: Suit.SPADES,
    Suit.HEARTS: Suit.DIAMONDS,
    Suit.DIAMONDS: Suit.HEARTS,
}

# Trump-group strength for the two bowers; both outrank every plain trump card, whose
# strength is just its ``Rank`` value (0..5, JACK excluded -- see ``trump_rank``).
_RIGHT_BOWER_STRENGTH = 100
_LEFT_BOWER_STRENGTH = 99


def make_card(suit: Suit, rank: Rank) -> Card:
    return int(suit) * RANKS_PER_SUIT + int(rank)


def suit_of(card: Card) -> Suit:
    """The card's *printed* suit (not bower-adjusted; see ``effective_suit``)."""
    return Suit(card // RANKS_PER_SUIT)


def rank_of(card: Card) -> Rank:
    return Rank(card % RANKS_PER_SUIT)


def full_deck() -> tuple[Card, ...]:
    """All 24 cards, in ``card_id`` order."""
    return tuple(range(NUM_CARDS))


def same_color_suit(suit: Suit) -> Suit:
    """The other suit of the same color (the source of that suit's left bower)."""
    return _SAME_COLOR[suit]


def is_right_bower(card: Card, trump: Suit) -> bool:
    return suit_of(card) == trump and rank_of(card) == Rank.JACK


def is_left_bower(card: Card, trump: Suit) -> bool:
    return suit_of(card) == same_color_suit(trump) and rank_of(card) == Rank.JACK


def is_bower(card: Card, trump: Suit) -> bool:
    return is_right_bower(card, trump) or is_left_bower(card, trump)


def effective_suit(card: Card, trump: Suit) -> Suit:
    """The suit ``card`` counts as for follow-suit and trick comparison.

    Identical to ``suit_of`` except for the left bower, which counts as trump even
    though its printed suit is trump's same-color partner.
    """
    if is_left_bower(card, trump):
        return trump
    return suit_of(card)


def trump_rank(card: Card, trump: Suit) -> int:
    """Strength of ``card`` within the trump group (higher wins).

    Only meaningful when ``effective_suit(card, trump) == trump``; callers must check
    that first (or only call this on cards already known to be trump).
    """
    if is_right_bower(card, trump):
        return _RIGHT_BOWER_STRENGTH
    if is_left_bower(card, trump):
        return _LEFT_BOWER_STRENGTH
    # Any other trump card is a plain trump-suit card (never the trump suit's own
    # jack -- that's always the right bower, caught above). Rank values 0..5 (minus
    # JACK=2, which can't occur here) already sort correctly: 9 < 10 < Q < K < A.
    return int(rank_of(card))


def plain_rank(card: Card) -> int:
    """Strength of ``card`` within its own (non-trump) suit (higher wins).

    Rank values already sort correctly for a plain suit: 9 < 10 < J < Q < K < A. Not
    valid for a card whose ``effective_suit`` is trump (use ``trump_rank`` instead).
    """
    return int(rank_of(card))
