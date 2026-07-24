"""Tests for MatchSummary -- plan Slice 3a, tests 2-5.

Builds MatchLog values by hand (no engine/agents involved) so the arithmetic is
pinned down independent of any particular match-recording run.
"""

from __future__ import annotations

from gamesim.analysis import MatchSummary, summarize_match
from gamesim.recording.match_log import MatchGameLog, MatchLog, MatchOutcome


def _game(
    index: int, seats: tuple[str, str], moves: list[int], outcome: MatchOutcome
) -> MatchGameLog:
    """Build a MatchGameLog whose actions alternate agent 0/1, starting with 0."""
    actions = tuple((move_index % 2, column) for move_index, column in enumerate(moves))
    return MatchGameLog(index=index, seed=index, seats=seats, actions=actions, outcome=outcome)


def _hand_built_log() -> MatchLog:
    # Four games, seats alternate first-mover like record_match does.
    return MatchLog(
        agent_a="alpha",
        agent_b="beta",
        games=(
            # Game 0: alpha moves first (seats[0]="alpha") and wins in 3 plies.
            _game(0, ("alpha", "beta"), [3, 2, 3], "agent_a"),
            # Game 1: beta moves first (seats[0]="beta") and wins in 5 plies.
            _game(1, ("beta", "alpha"), [0, 1, 0, 1, 0], "agent_b"),
            # Game 2: alpha moves first and loses (beta wins) in 4 plies.
            _game(2, ("alpha", "beta"), [4, 5, 4, 5], "agent_b"),
            # Game 3: beta moves first and it's a draw, 2 plies.
            _game(3, ("beta", "alpha"), [6, 6], "draw"),
        ),
    )


def test_summarize_match_outcome_counts_and_win_rates() -> None:
    summary = summarize_match(_hand_built_log())

    assert isinstance(summary, MatchSummary)
    assert summary.total_games == 4
    assert summary.agent_a_wins == 1
    assert summary.agent_b_wins == 2
    assert summary.draws == 1
    assert summary.agent_a_win_rate == 0.25
    assert summary.agent_b_win_rate == 0.5
    assert summary.draw_rate == 0.25


def test_summarize_match_first_mover_breakdown() -> None:
    summary = summarize_match(_hand_built_log())

    # Game 0: first mover (alpha) won.       -> first-mover win
    # Game 1: first mover (beta) won.        -> first-mover win
    # Game 2: first mover (alpha) lost.      -> first-mover loss
    # Game 3: first mover (beta) drew.       -> first-mover draw
    assert summary.first_mover_wins == 2
    assert summary.first_mover_losses == 1
    assert summary.first_mover_draws == 1
    assert summary.first_mover_win_rate == 0.5


def test_summarize_match_game_length_stats_and_histogram() -> None:
    summary = summarize_match(_hand_built_log())

    # Lengths: 3, 5, 4, 2
    assert summary.game_length_min == 2
    assert summary.game_length_max == 5
    assert summary.game_length_mean == (3 + 5 + 4 + 2) / 4
    assert summary.game_length_histogram == ((2, 1), (3, 1), (4, 1), (5, 1))


def test_summarize_match_opening_and_column_usage_distributions() -> None:
    summary = summarize_match(_hand_built_log())

    # Opening columns: 3 (game 0), 0 (game 1), 4 (game 2), 6 (game 3).
    assert summary.opening_move_distribution == ((0, 1), (3, 1), (4, 1), (6, 1))

    # All columns played: 3,2,3 | 0,1,0,1,0 | 4,5,4,5 | 6,6
    # col0: 3, col1: 2, col2: 1, col3: 2, col4: 2, col5: 2, col6: 2
    assert summary.column_usage_distribution == (
        (0, 3),
        (1, 2),
        (2, 1),
        (3, 2),
        (4, 2),
        (5, 2),
        (6, 2),
    )


def test_summarize_match_handles_an_empty_log_without_dividing_by_zero() -> None:
    empty_log = MatchLog(agent_a="alpha", agent_b="beta", games=())

    summary = summarize_match(empty_log)

    assert summary.total_games == 0
    assert summary.agent_a_win_rate == 0.0
    assert summary.first_mover_win_rate == 0.0
    assert summary.game_length_mean == 0.0
    assert summary.game_length_histogram == ()
