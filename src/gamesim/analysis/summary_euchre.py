"""Pure, torch-free summary statistics over a recorded ``EuchreMatchLog``.

Same principle as ``summary.py``: everything here is computable directly from each
hand's logged outcome/points/maker_team/trump/alone metadata (see
``EuchreMatchGameLog``), with no engine replay needed -- board/trick reconstruction
for the interactive report is ``analysis.replay_euchre.replay_euchre_match_game``'s
job (docs/adr/0009-offline-analysis-and-reporting.md).
"""

from __future__ import annotations

from dataclasses import dataclass

from gamesim.recording.euchre_match_log import EuchreMatchLog


@dataclass(frozen=True)
class EuchreMatchSummary:
    """Aggregate statistics for a batch of recorded Euchre hands.

    ``*_distribution`` fields are ``(key, count)`` pairs sorted by key -- plain
    tuples rather than dicts, matching ``MatchSummary``'s convention, so this stays
    hashable/frozen with a deterministic iteration order for display and tests.
    There is no ``draws``/``draw_rate`` field: unlike Connect Four, a single Euchre
    hand always has a scoring team (see ``EuchreEngine._score_hand``).
    """

    team_a: str
    team_b: str
    total_hands: int

    team_a_wins: int
    team_b_wins: int
    team_a_win_rate: float
    team_b_win_rate: float

    # A "march" is the maker's team taking all 5 tricks (2 points, or 4 alone); a
    # "euchre" is the maker's team failing to take 3 (the *defenders* score 2). Both
    # score 2 points, so points alone can't distinguish them -- outcome vs.
    # maker_team can.
    march_count: int
    lone_march_count: int
    euchre_count: int
    maker_success_rate: float  # fraction of hands NOT euchred

    alone_call_count: int
    alone_call_rate: float

    points_distribution: tuple[tuple[int, int], ...]
    trump_suit_distribution: tuple[tuple[int, int], ...]


def summarize_euchre_match(log: EuchreMatchLog) -> EuchreMatchSummary:
    """Compute :class:`EuchreMatchSummary` statistics purely from logged metadata."""
    total_hands = len(log.games)
    team_a_wins = sum(game.outcome == "team_a" for game in log.games)
    team_b_wins = sum(game.outcome == "team_b" for game in log.games)

    march_count = 0
    lone_march_count = 0
    euchre_count = 0
    alone_call_count = 0
    points_counts: dict[int, int] = {}
    trump_counts: dict[int, int] = {}

    for game in log.games:
        maker_succeeded = game.outcome == game.maker_team
        if maker_succeeded:
            if game.points in (2, 4):
                march_count += 1
                if game.alone and game.points == 4:
                    lone_march_count += 1
        else:
            euchre_count += 1
        if game.alone:
            alone_call_count += 1
        points_counts[game.points] = points_counts.get(game.points, 0) + 1
        trump_counts[game.trump] = trump_counts.get(game.trump, 0) + 1

    return EuchreMatchSummary(
        team_a=log.team_a,
        team_b=log.team_b,
        total_hands=total_hands,
        team_a_wins=team_a_wins,
        team_b_wins=team_b_wins,
        team_a_win_rate=_rate(team_a_wins, total_hands),
        team_b_win_rate=_rate(team_b_wins, total_hands),
        march_count=march_count,
        lone_march_count=lone_march_count,
        euchre_count=euchre_count,
        maker_success_rate=_rate(total_hands - euchre_count, total_hands),
        alone_call_count=alone_call_count,
        alone_call_rate=_rate(alone_call_count, total_hands),
        points_distribution=tuple(sorted(points_counts.items())),
        trump_suit_distribution=tuple(sorted(trump_counts.items())),
    )


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0
