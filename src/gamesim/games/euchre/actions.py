"""Euchre actions: one fixed-size discrete space shared across every phase.

Connect Four gets away with "an action is just the column index" because it has one
action kind (see ``connect_four/actions.py``). Euchre has several heterogeneous kinds
-- pass, order-up, name-trump, discard, play-a-card -- but rather than a tagged union
per phase, they're packed into a single flat ``int`` space (see
plans/phase-04-euchre.md "Action space"): the *phase* (carried in ``EuchreState``, not
in the action) determines which slice of the space is meaningful, and
``EuchreEngine.legal_actions`` masks accordingly, exactly like Connect Four's mask
narrows by board state rather than the action type changing shape.

Card actions (0-23) are reused for both the dealer's discard and ordinary trick-play --
same reuse-by-phase idea, not two separate encodings.
"""

from __future__ import annotations

from .cards import NUM_CARDS, Suit

Action = int

# 0..23: play/discard card_id (see cards.py). NUM_CARDS == 24.
PASS: Action = NUM_CARDS  # 24
ORDER_UP: Action = NUM_CARDS + 1  # 25
ORDER_UP_ALONE: Action = NUM_CARDS + 2  # 26
CALL_SUIT_BASE: Action = NUM_CARDS + 3  # 27..30
CALL_SUIT_ALONE_BASE: Action = NUM_CARDS + 3 + len(Suit)  # 31..34

NUM_ACTIONS = NUM_CARDS + 3 + 2 * len(Suit)  # 35


def call_suit_action(suit: Suit) -> Action:
    return CALL_SUIT_BASE + int(suit)


def call_suit_alone_action(suit: Suit) -> Action:
    return CALL_SUIT_ALONE_BASE + int(suit)


def is_call_suit_action(action: Action) -> bool:
    return CALL_SUIT_BASE <= action < CALL_SUIT_BASE + len(Suit)


def is_call_suit_alone_action(action: Action) -> bool:
    return CALL_SUIT_ALONE_BASE <= action < CALL_SUIT_ALONE_BASE + len(Suit)


def suit_of_call_action(action: Action) -> Suit:
    """The suit named by a round-2 call action (alone or not).

    Raises if ``action`` isn't a call-suit action -- callers only invoke this after
    confirming the action is one of the two call ranges (mirrors the rest of the
    engine: illegal/malformed actions are rejected loudly, not silently coerced).
    """
    if is_call_suit_action(action):
        return Suit(action - CALL_SUIT_BASE)
    if is_call_suit_alone_action(action):
        return Suit(action - CALL_SUIT_ALONE_BASE)
    raise ValueError(f"action {action} does not name a suit")
