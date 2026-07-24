"""Torch-free offline analysis of recorded matches (see docs/adr/0009).

Consumes ``MatchLog`` artifacts (``gamesim.recording``) and, where board state is
needed, reconstructs it by replaying actions through ``ConnectFourEngine`` --
never deriving state itself. Fully usable without the DRL stack installed;
nothing in this package imports torch.
"""

from .replay import replay_match_game
from .replay_euchre import EuchreAction, EuchrePlySnapshot, replay_euchre_match_game
from .summary import MatchSummary, summarize_match
from .summary_euchre import EuchreMatchSummary, summarize_euchre_match

__all__ = [
    "MatchSummary",
    "summarize_match",
    "replay_match_game",
    "EuchreAction",
    "EuchrePlySnapshot",
    "replay_euchre_match_game",
    "EuchreMatchSummary",
    "summarize_euchre_match",
]
