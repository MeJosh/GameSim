"""Torch-free incremental-training progress schema + metrics (Slice 3d).

Extends the Slice-3a/3b analysis layer (see docs/adr/0009-offline-analysis-and-reporting.md)
to a whole *run* of checkpoints rather than a single match: for each labeled "stage"
(a baseline or a trained checkpoint) this module records evaluation matches against
fixed baselines (random + minimax), derives their metrics via
``gamesim.analysis.summary.summarize_match``, and plays a round-robin of head-to-head
matches among labeled stage agents.

Everything here is generic over ``gamesim.core.agent.Agent`` -- it never touches torch
or sb3. The torch-backed training driver (``gamesim.experiments.incremental``) calls
``evaluate_stage``/``head_to_head`` with a live policy wrapped as an ``Agent`` and
snapshot opponents, all behind its own local/isolated imports.

The persisted schema is versioned (``PROGRESS_FORMAT``, currently v2) and round-trips
through ``write_progress_log``/``read_progress_log``; the v1 schema written by the
original ``gamesim.experiments.incremental.write_progress`` (trained-vs-random winrate
only) is a different, older format string and is rejected with a clear error rather
than silently misparsed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gamesim.analysis.summary import MatchSummary, summarize_match
from gamesim.core.agent import Agent
from gamesim.recording.match import record_match
from gamesim.recording.match_log import MatchLog

PROGRESS_FORMAT = "gamesim.incremental-training/v2"

DEFAULT_RANDOM_LABEL = "random"
DEFAULT_MINIMAX_LABEL = "minimax"
DEFAULT_NUM_GAMES = 200


@dataclass(frozen=True)
class BaselineMetrics:
    """One stage agent's evaluation results against a single fixed baseline.

    Derived from a recorded match's :class:`~gamesim.analysis.summary.MatchSummary`
    (``opponent`` is the baseline's logical name, i.e. ``summary.agent_b``).
    """

    opponent: str
    total_games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    game_length_mean: float
    opening_move_distribution: tuple[tuple[int, int], ...]

    @classmethod
    def from_summary(cls, summary: MatchSummary) -> BaselineMetrics:
        return cls(
            opponent=summary.agent_b,
            total_games=summary.total_games,
            wins=summary.agent_a_wins,
            losses=summary.agent_b_wins,
            draws=summary.draws,
            win_rate=summary.agent_a_win_rate,
            game_length_mean=summary.game_length_mean,
            opening_move_distribution=summary.opening_move_distribution,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "opponent": self.opponent,
            "total_games": self.total_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "game_length_mean": self.game_length_mean,
            "opening_move_distribution": [
                [column, count] for column, count in self.opening_move_distribution
            ],
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> BaselineMetrics:
        try:
            opponent = record["opponent"]
            total_games = record["total_games"]
            wins = record["wins"]
            losses = record["losses"]
            draws = record["draws"]
            win_rate = record["win_rate"]
            game_length_mean = record["game_length_mean"]
            opening_move_distribution = record["opening_move_distribution"]
        except KeyError as error:
            raise ValueError(f"baseline metrics missing field: {error}") from error
        if not isinstance(opponent, str):
            raise ValueError("baseline metrics 'opponent' must be a string")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (total_games, wins, losses, draws)
        ):
            raise ValueError("baseline metrics counts must be integers")
        if not isinstance(win_rate, (int, float)) or not isinstance(game_length_mean, (int, float)):
            raise ValueError("baseline metrics rates must be numbers")
        if not isinstance(opening_move_distribution, list):
            raise ValueError("baseline metrics 'opening_move_distribution' must be a list")
        distribution_entries: list[tuple[int, int]] = []
        for pair in opening_move_distribution:
            if (
                not isinstance(pair, Sequence)
                or isinstance(pair, (str, bytes))
                or len(pair) != 2
                or not all(isinstance(value, int) and not isinstance(value, bool) for value in pair)
            ):
                raise ValueError(
                    "baseline metrics 'opening_move_distribution' entries must be "
                    "[column, count] integer pairs"
                )
            distribution_entries.append((int(pair[0]), int(pair[1])))
        distribution = tuple(distribution_entries)
        return cls(
            opponent=opponent,
            total_games=total_games,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=float(win_rate),
            game_length_mean=float(game_length_mean),
            opening_move_distribution=distribution,
        )


@dataclass(frozen=True)
class StageMetrics:
    """One evaluated stage (a baseline snapshot or a trained checkpoint).

    ``match_log_paths`` optionally records where the representative recorded
    ``MatchLog``s for this stage's baseline evaluations were persisted, as paths
    relative to the run directory (e.g. ``{"random": "matches/baseline-vs-
    random.zip"}``) -- see :func:`evaluate_stage`, which returns the in-memory
    ``MatchLog``s but leaves writing them (and thus populating this field) to the
    caller. Defaults to empty and is backward-tolerant: older persisted progress
    logs without this field simply read back with no paths.
    """

    label: str
    cumulative_timesteps: int
    vs_random: BaselineMetrics
    vs_minimax: BaselineMetrics
    match_log_paths: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "cumulative_timesteps": self.cumulative_timesteps,
            "vs_random": self.vs_random.to_dict(),
            "vs_minimax": self.vs_minimax.to_dict(),
            "match_log_paths": dict(self.match_log_paths),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> StageMetrics:
        try:
            label = record["label"]
            cumulative_timesteps = record["cumulative_timesteps"]
            vs_random = record["vs_random"]
            vs_minimax = record["vs_minimax"]
        except KeyError as error:
            raise ValueError(f"stage metrics missing field: {error}") from error
        match_log_paths_record = record.get("match_log_paths", {})
        if not isinstance(label, str):
            raise ValueError("stage metrics 'label' must be a string")
        if not isinstance(cumulative_timesteps, int) or isinstance(cumulative_timesteps, bool):
            raise ValueError("stage metrics 'cumulative_timesteps' must be an int")
        if not isinstance(vs_random, Mapping) or not isinstance(vs_minimax, Mapping):
            raise ValueError("stage metrics 'vs_random'/'vs_minimax' must be objects")
        if not isinstance(match_log_paths_record, Mapping):
            raise ValueError("stage metrics 'match_log_paths' must be an object")
        match_log_paths: dict[str, str] = {}
        for key, value in match_log_paths_record.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("stage metrics 'match_log_paths' entries must be strings")
            match_log_paths[key] = value
        return cls(
            label=label,
            cumulative_timesteps=cumulative_timesteps,
            vs_random=BaselineMetrics.from_dict(vs_random),
            vs_minimax=BaselineMetrics.from_dict(vs_minimax),
            match_log_paths=match_log_paths,
        )


@dataclass(frozen=True)
class HeadToHeadEntry:
    """One directed head-to-head record: how ``row`` fared against ``column``.

    Produced in complementary pairs by :func:`head_to_head` -- the ``(row, column)``
    and ``(column, row)`` entries for the same underlying match always mirror each
    other exactly (``row``'s wins are ``column``'s losses and vice versa, draws
    match), since both are read from the two sides of one recorded match.
    """

    row: str
    column: str
    wins: int
    losses: int
    draws: int
    games: int
    win_rate: float
    loss_rate: float
    draw_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "column": self.column,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "games": self.games,
            "win_rate": self.win_rate,
            "loss_rate": self.loss_rate,
            "draw_rate": self.draw_rate,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> HeadToHeadEntry:
        try:
            row = record["row"]
            column = record["column"]
            wins = record["wins"]
            losses = record["losses"]
            draws = record["draws"]
            games = record["games"]
            win_rate = record["win_rate"]
            loss_rate = record["loss_rate"]
            draw_rate = record["draw_rate"]
        except KeyError as error:
            raise ValueError(f"head-to-head entry missing field: {error}") from error
        if not isinstance(row, str) or not isinstance(column, str):
            raise ValueError("head-to-head entry 'row'/'column' must be strings")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (wins, losses, draws, games)
        ):
            raise ValueError("head-to-head entry counts must be integers")
        return cls(
            row=row,
            column=column,
            wins=wins,
            losses=losses,
            draws=draws,
            games=games,
            win_rate=float(win_rate),
            loss_rate=float(loss_rate),
            draw_rate=float(draw_rate),
        )


@dataclass(frozen=True)
class ProgressLog:
    """A whole run's progress: evaluated stages plus a head-to-head matrix."""

    stages: tuple[StageMetrics, ...]
    head_to_head: tuple[HeadToHeadEntry, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": PROGRESS_FORMAT,
            "stages": [stage.to_dict() for stage in self.stages],
            "head_to_head": [entry.to_dict() for entry in self.head_to_head],
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> ProgressLog:
        format_value = record.get("format")
        if format_value != PROGRESS_FORMAT:
            raise ValueError(
                f"unsupported progress log format: {format_value!r} (expected {PROGRESS_FORMAT!r})"
            )
        stages_record = record.get("stages")
        head_to_head_record = record.get("head_to_head", [])
        if not isinstance(stages_record, list):
            raise ValueError("progress log 'stages' must be a list")
        if not isinstance(head_to_head_record, list):
            raise ValueError("progress log 'head_to_head' must be a list")
        stages = tuple(StageMetrics.from_dict(item) for item in stages_record)
        head_to_head = tuple(HeadToHeadEntry.from_dict(item) for item in head_to_head_record)
        return cls(stages=stages, head_to_head=head_to_head)


def write_progress_log(path: str | Path, progress: ProgressLog) -> Path:
    """Atomically write ``progress`` as the versioned JSON progress index."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(progress.to_dict(), indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def read_progress_log(path: str | Path) -> ProgressLog:
    """Read and validate a versioned JSON progress index, raising ``ValueError``.

    Rejects anything that isn't exactly :data:`PROGRESS_FORMAT` -- including the
    older ``gamesim.incremental-training/v1`` schema -- with a clear message rather
    than guessing at a migration.
    """
    input_path = Path(path)
    try:
        record = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("invalid JSON progress log") from error
    if not isinstance(record, Mapping):
        raise ValueError("progress log must be a JSON object")
    return ProgressLog.from_dict(record)


def evaluate_stage(
    agent: Agent[Any, int],
    *,
    label: str,
    cumulative_timesteps: int,
    random_agent: Agent[Any, int],
    minimax_agent: Agent[Any, int],
    random_label: str = DEFAULT_RANDOM_LABEL,
    minimax_label: str = DEFAULT_MINIMAX_LABEL,
    num_games: int = DEFAULT_NUM_GAMES,
    seed: int = 0,
) -> tuple[StageMetrics, dict[str, MatchLog]]:
    """Evaluate ``agent`` (a stage/checkpoint stand-in) against both baselines.

    Torch-free and generic over ``Agent``: tests exercise it with ``MinimaxAgent``
    and ``RandomAgent`` in place of a trained checkpoint. Records one match against
    ``random_agent`` and one against ``minimax_agent`` (via
    ``gamesim.recording.match.record_match``, which alternates first-mover) and
    derives each side's :class:`BaselineMetrics` via ``summarize_match``.

    Returns the derived :class:`StageMetrics` (with an empty ``match_log_paths``)
    together with the two recorded ``MatchLog``s that produced it, keyed
    ``"random"``/``"minimax"``. This function deliberately never touches the
    filesystem -- it is generic over any ``Agent`` and used directly by tests with
    non-torch stand-ins -- so persisting those logs (e.g. via
    ``gamesim.recording.match_log.write_match_log`` and recording their paths on
    the returned ``StageMetrics``) is left to the caller.
    """
    vs_random_log = record_match(
        agent,
        random_agent,
        agent_a_name=label,
        agent_b_name=random_label,
        num_games=num_games,
        seed=seed,
    )
    vs_minimax_log = record_match(
        agent,
        minimax_agent,
        agent_a_name=label,
        agent_b_name=minimax_label,
        num_games=num_games,
        seed=seed + 1,
    )
    stage = StageMetrics(
        label=label,
        cumulative_timesteps=cumulative_timesteps,
        vs_random=BaselineMetrics.from_summary(summarize_match(vs_random_log)),
        vs_minimax=BaselineMetrics.from_summary(summarize_match(vs_minimax_log)),
    )
    return stage, {"random": vs_random_log, "minimax": vs_minimax_log}


def head_to_head(
    labeled_agents: Sequence[tuple[str, Agent[Any, int]]],
    *,
    num_games: int = DEFAULT_NUM_GAMES,
    seed: int = 0,
) -> tuple[HeadToHeadEntry, ...]:
    """Round-robin every unordered pair of ``labeled_agents`` exactly once.

    Each pair is played as a single recorded match (``record_match`` alternates
    which side moves first internally, so this stays fair without doubling the
    number of games). Both directions are reported: a ``(row, column)`` entry from
    that match's agent-a perspective, and its ``(column, row)`` mirror from the
    agent-b perspective -- so the pair is complementary by construction.
    """
    entries: list[HeadToHeadEntry] = []
    for row_index, (row_label, row_agent) in enumerate(labeled_agents):
        for column_index in range(row_index + 1, len(labeled_agents)):
            column_label, column_agent = labeled_agents[column_index]
            match_seed = seed + row_index * 997 + column_index
            log = record_match(
                row_agent,
                column_agent,
                agent_a_name=row_label,
                agent_b_name=column_label,
                num_games=num_games,
                seed=match_seed,
            )
            summary = summarize_match(log)
            entries.append(
                _head_to_head_entry(
                    row_label,
                    column_label,
                    summary.agent_a_wins,
                    summary.agent_b_wins,
                    summary.draws,
                )
            )
            entries.append(
                _head_to_head_entry(
                    column_label,
                    row_label,
                    summary.agent_b_wins,
                    summary.agent_a_wins,
                    summary.draws,
                )
            )
    return tuple(entries)


def _head_to_head_entry(
    row: str, column: str, wins: int, losses: int, draws: int
) -> HeadToHeadEntry:
    games = wins + losses + draws
    return HeadToHeadEntry(
        row=row,
        column=column,
        wins=wins,
        losses=losses,
        draws=draws,
        games=games,
        win_rate=_rate(wins, games),
        loss_rate=_rate(losses, games),
        draw_rate=_rate(draws, games),
    )


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0
