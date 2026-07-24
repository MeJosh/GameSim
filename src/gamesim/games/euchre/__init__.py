"""Euchre: the Phase 4 second game.

4 players, 2 fixed partnerships, 24-card deck. Two-round bidding (order-up / name
trump, with stick-the-dealer), right/left bower, going alone, standard scoring. One
``EuchreEngine`` episode = one hand (not a first-to-10 match) -- see
plans/phase-04-euchre.md. Plain state, no ECS -- see docs/architecture.md.
"""

from __future__ import annotations

from .actions import NUM_ACTIONS, Action
from .cards import Card, Suit
from .engine import EuchreEngine, EuchreObservation
from .state import EuchreRules, EuchreState, Phase

__all__ = [
    "NUM_ACTIONS",
    "Action",
    "Card",
    "EuchreEngine",
    "EuchreObservation",
    "EuchreRules",
    "EuchreState",
    "Phase",
    "Suit",
]
