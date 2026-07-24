"""Pure, torch-free summary statistics over a recorded ``MatchLog``.

No engine replay is needed here -- outcome, seat, and move-distribution stats are
computable directly from the logged actions (``MatchGameLog.actions``) and
declared outcomes. Board reconstruction (when a report needs actual board
positions) is ``gamesim.analysis.replay.replay_match_game``'s job, not this
module's -- see docs/adr/0009-offline-analysis-and-reporting.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gamesim.recording.match_log import MatchGameLog, MatchLog

_FirstMoverResult = Literal["win", "loss", "draw"]


@dataclass(frozen=True)
class MatchSummary:
    """Aggregate statistics for a batch of recorded Connect Four games.

    ``*_distribution``/``game_length_histogram`` fields are ``(key, count)`` pairs
    sorted by key -- plain tuples rather than dicts, so ``MatchSummary`` stays
    hashable/frozen and has a deterministic iteration order for display and tests.
    """

    agent_a: str
    agent_b: str
    total_games: int

    agent_a_wins: int
    agent_b_wins: int
    draws: int
    agent_a_win_rate: float
    agent_b_win_rate: float
    draw_rate: float

    # Does moving first matter? "First mover" is whichever named agent occupied
    # game.seats[0] in that particular game (record_match alternates this).
    first_mover_wins: int
    first_mover_losses: int
    first_mover_draws: int
    first_mover_win_rate: float

    game_length_mean: float
    game_length_min: int
    game_length_max: int
    game_length_histogram: tuple[tuple[int, int], ...]

    opening_move_distribution: tuple[tuple[int, int], ...]
    column_usage_distribution: tuple[tuple[int, int], ...]


def summarize_match(log: MatchLog) -> MatchSummary:
    """Compute :class:`MatchSummary` statistics purely from the logged actions."""
    total_games = len(log.games)
    agent_a_wins = sum(game.outcome == "agent_a" for game in log.games)
    agent_b_wins = sum(game.outcome == "agent_b" for game in log.games)
    draws = sum(game.outcome == "draw" for game in log.games)

    first_mover_wins = 0
    first_mover_losses = 0
    first_mover_draws = 0
    for game in log.games:
        result = _first_mover_result(log, game)
        if result == "win":
            first_mover_wins += 1
        elif result == "loss":
            first_mover_losses += 1
        else:
            first_mover_draws += 1

    lengths = [len(game.actions) for game in log.games]
    length_counts: dict[int, int] = {}
    for length in lengths:
        length_counts[length] = length_counts.get(length, 0) + 1

    opening_counts: dict[int, int] = {}
    column_counts: dict[int, int] = {}
    for game in log.games:
        for move_index, (_agent, column) in enumerate(game.actions):
            column_counts[column] = column_counts.get(column, 0) + 1
            if move_index == 0:
                opening_counts[column] = opening_counts.get(column, 0) + 1

    return MatchSummary(
        agent_a=log.agent_a,
        agent_b=log.agent_b,
        total_games=total_games,
        agent_a_wins=agent_a_wins,
        agent_b_wins=agent_b_wins,
        draws=draws,
        agent_a_win_rate=_rate(agent_a_wins, total_games),
        agent_b_win_rate=_rate(agent_b_wins, total_games),
        draw_rate=_rate(draws, total_games),
        first_mover_wins=first_mover_wins,
        first_mover_losses=first_mover_losses,
        first_mover_draws=first_mover_draws,
        first_mover_win_rate=_rate(first_mover_wins, total_games),
        game_length_mean=(sum(lengths) / len(lengths)) if lengths else 0.0,
        game_length_min=min(lengths) if lengths else 0,
        game_length_max=max(lengths) if lengths else 0,
        game_length_histogram=tuple(sorted(length_counts.items())),
        opening_move_distribution=tuple(sorted(opening_counts.items())),
        column_usage_distribution=tuple(sorted(column_counts.items())),
    )


def _first_mover_result(log: MatchLog, game: MatchGameLog) -> _FirstMoverResult:
    """Whether the seat that moved first (``game.seats[0]``) won, lost, or drew."""
    if game.outcome == "draw":
        return "draw"
    winner_name = log.agent_a if game.outcome == "agent_a" else log.agent_b
    return "win" if winner_name == game.seats[0] else "loss"


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0
