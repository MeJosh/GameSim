"""Tests for EuchreMatchSummary.

Builds EuchreMatchLog values by hand (no engine/agents involved) so the arithmetic
is pinned down independent of any particular match-recording run -- same approach
as tests/analysis/test_summary.py.
"""

from __future__ import annotations

from gamesim.analysis import EuchreMatchSummary, summarize_euchre_match
from gamesim.recording.euchre_match_log import EuchreMatchGameLog, EuchreMatchLog


def _hand(
    index: int,
    *,
    outcome: str,
    maker_team: str,
    points: int,
    trump: int,
    alone: bool = False,
) -> EuchreMatchGameLog:
    return EuchreMatchGameLog(
        index=index,
        seed=index,
        dealer=0,
        stick_the_dealer=True,
        seats=("alpha", "beta", "alpha", "beta"),
        actions=((0, 25), (0, 3)),  # content doesn't matter for summary stats
        outcome=outcome,  # type: ignore[arg-type]
        points=points,
        maker_team=maker_team,  # type: ignore[arg-type]
        trump=trump,
        alone=alone,
    )


def _hand_built_log() -> EuchreMatchLog:
    return EuchreMatchLog(
        team_a="alpha",
        team_b="beta",
        games=(
            # Hand 0: alpha makes, wins 1 point (3-4 tricks).
            _hand(0, outcome="team_a", maker_team="team_a", points=1, trump=0),
            # Hand 1: beta makes, marches (2 points, not alone).
            _hand(1, outcome="team_b", maker_team="team_b", points=2, trump=1),
            # Hand 2: alpha makes alone, lone march (4 points).
            _hand(2, outcome="team_a", maker_team="team_a", points=4, trump=0, alone=True),
            # Hand 3: beta makes but gets euchred -- alpha (defenders) scores 2.
            _hand(3, outcome="team_a", maker_team="team_b", points=2, trump=2),
            # Hand 4: alpha makes alone but gets euchred -- beta scores 2.
            _hand(4, outcome="team_b", maker_team="team_a", points=2, trump=0, alone=True),
        ),
    )


def test_summarize_euchre_match_win_counts_and_rates() -> None:
    summary = summarize_euchre_match(_hand_built_log())

    assert isinstance(summary, EuchreMatchSummary)
    assert summary.total_hands == 5
    # Hand outcomes: team_a, team_b, team_a, team_a, team_b
    assert summary.team_a_wins == 3
    assert summary.team_b_wins == 2
    assert summary.team_a_win_rate == 0.6
    assert summary.team_b_win_rate == 0.4


def test_summarize_euchre_match_march_lone_march_and_euchre_counts() -> None:
    summary = summarize_euchre_match(_hand_built_log())

    # Hand 0: maker succeeded, 1 point -- not a march.
    # Hand 1: maker succeeded, march (2 pts, not alone).
    # Hand 2: maker succeeded, lone march (4 pts, alone).
    # Hand 3: maker (beta) failed -- euchre.
    # Hand 4: maker (alpha, alone) failed -- euchre.
    assert summary.march_count == 2
    assert summary.lone_march_count == 1
    assert summary.euchre_count == 2
    assert summary.maker_success_rate == 3 / 5


def test_summarize_euchre_match_alone_call_rate() -> None:
    summary = summarize_euchre_match(_hand_built_log())

    # Hands 2 and 4 went alone.
    assert summary.alone_call_count == 2
    assert summary.alone_call_rate == 2 / 5


def test_summarize_euchre_match_points_and_trump_distributions() -> None:
    summary = summarize_euchre_match(_hand_built_log())

    # Points: 1, 2, 4, 2, 2 -> {1:1, 2:3, 4:1}
    assert summary.points_distribution == ((1, 1), (2, 3), (4, 1))
    # Trump: 0, 1, 0, 2, 0 -> {0:3, 1:1, 2:1}
    assert summary.trump_suit_distribution == ((0, 3), (1, 1), (2, 1))


def test_summarize_euchre_match_handles_an_empty_log_without_dividing_by_zero() -> None:
    empty_log = EuchreMatchLog(team_a="alpha", team_b="beta", games=())

    summary = summarize_euchre_match(empty_log)

    assert summary.total_hands == 0
    assert summary.team_a_win_rate == 0.0
    assert summary.maker_success_rate == 0.0
    assert summary.alone_call_rate == 0.0
    assert summary.points_distribution == ()
    assert summary.trump_suit_distribution == ()
