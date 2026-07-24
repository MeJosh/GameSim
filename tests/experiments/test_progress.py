"""Tests for torch-free progress metrics + schema -- plan Slice 3d, test list 1 & 3.

All agents here are non-torch stand-ins for training checkpoints (``MinimaxAgent`` at
different depths for "strong"/"weak" and ``RandomAgent`` for baselines), per the plan's
"Core principle": the metric computation, schema read/write are torch-free and tested
in-sandbox with these agents rather than a real trained policy.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gamesim.agents.scripted import MinimaxAgent
from gamesim.analysis.summary import summarize_match
from gamesim.core.agent import Agent, RandomAgent
from gamesim.experiments.progress import (
    PROGRESS_FORMAT,
    BaselineMetrics,
    HeadToHeadEntry,
    ProgressLog,
    StageMetrics,
    evaluate_stage,
    head_to_head,
    read_progress_log,
    write_progress_log,
)
from gamesim.recording.match import record_match
from gamesim.recording.match_log import read_match_log, write_match_log

# --- evaluate_stage: metrics derive correctly from summarize_match -----------------


def test_evaluate_stage_derives_baseline_metrics_from_summarize_match() -> None:
    stage, match_logs = evaluate_stage(
        MinimaxAgent(depth=2),
        label="checkpoint",
        cumulative_timesteps=4_096,
        random_agent=RandomAgent(seed=7),
        minimax_agent=MinimaxAgent(depth=1),
        num_games=6,
        seed=3,
    )

    # Independently recorded/summarized with fresh agent instances at the same
    # seeds -- evaluate_stage must reproduce exactly the same MatchSummary fields.
    expected_vs_random = summarize_match(
        record_match(
            MinimaxAgent(depth=2),
            RandomAgent(seed=7),
            agent_a_name="checkpoint",
            agent_b_name="random",
            num_games=6,
            seed=3,
        )
    )
    expected_vs_minimax = summarize_match(
        record_match(
            MinimaxAgent(depth=2),
            MinimaxAgent(depth=1),
            agent_a_name="checkpoint",
            agent_b_name="minimax",
            num_games=6,
            seed=4,  # evaluate_stage uses seed + 1 for the minimax match
        )
    )

    assert stage.label == "checkpoint"
    assert stage.cumulative_timesteps == 4_096

    assert stage.vs_random.opponent == "random"
    assert stage.vs_random.total_games == expected_vs_random.total_games
    assert stage.vs_random.wins == expected_vs_random.agent_a_wins
    assert stage.vs_random.losses == expected_vs_random.agent_b_wins
    assert stage.vs_random.draws == expected_vs_random.draws
    assert stage.vs_random.win_rate == expected_vs_random.agent_a_win_rate
    assert stage.vs_random.game_length_mean == expected_vs_random.game_length_mean
    assert stage.vs_random.opening_move_distribution == expected_vs_random.opening_move_distribution

    assert stage.vs_minimax.opponent == "minimax"
    assert stage.vs_minimax.wins == expected_vs_minimax.agent_a_wins
    assert stage.vs_minimax.losses == expected_vs_minimax.agent_b_wins
    assert stage.vs_minimax.draws == expected_vs_minimax.draws
    assert stage.vs_minimax.game_length_mean == expected_vs_minimax.game_length_mean
    assert (
        stage.vs_minimax.opening_move_distribution == expected_vs_minimax.opening_move_distribution
    )

    # evaluate_stage never writes to disk, but hands back the recorded MatchLogs
    # that produced the stats above -- StageMetrics.match_log_paths stays empty
    # until a caller persists them (see the round-trip test below).
    assert stage.match_log_paths == {}
    assert set(match_logs) == {"random", "minimax"}
    assert match_logs["random"].agent_a == "checkpoint"
    assert match_logs["random"].agent_b == "random"
    assert len(match_logs["random"].games) == 6
    assert match_logs["minimax"].agent_a == "checkpoint"
    assert match_logs["minimax"].agent_b == "minimax"
    assert len(match_logs["minimax"].games) == 6


# --- winrate direction: minimax stand-in dominates the random stand-in -------------


def test_stage_metrics_winrate_direction_minimax_beats_random() -> None:
    strong, _strong_match_logs = evaluate_stage(
        MinimaxAgent(depth=3),
        label="strong",
        cumulative_timesteps=10_000,
        random_agent=RandomAgent(seed=101),
        minimax_agent=MinimaxAgent(depth=1),
        num_games=20,
        seed=1,
    )
    weak, _weak_match_logs = evaluate_stage(
        RandomAgent(seed=5),
        label="weak",
        cumulative_timesteps=0,
        random_agent=RandomAgent(seed=101),
        minimax_agent=MinimaxAgent(depth=1),
        num_games=20,
        seed=1,
    )

    # Pinned exact values (deterministic agents / fixed seeds).
    assert (strong.vs_random.wins, strong.vs_random.losses, strong.vs_random.draws) == (
        20,
        0,
        0,
    )
    assert strong.vs_random.win_rate == 1.0
    assert strong.vs_minimax.win_rate == 1.0
    assert weak.vs_random.win_rate == pytest.approx(0.4)
    assert weak.vs_minimax.win_rate == 0.0

    # Direction: the strong (minimax) stand-in dominates both baselines far more
    # than the weak (random) stand-in does.
    assert strong.vs_random.win_rate > weak.vs_random.win_rate
    assert strong.vs_minimax.win_rate > weak.vs_minimax.win_rate


# --- head_to_head: coherent matrix + strength ordering ------------------------------


def test_head_to_head_matrix_is_coherent_and_shows_strength_ordering() -> None:
    labeled_agents: list[tuple[str, Agent[Any, int]]] = [
        ("weak", RandomAgent(seed=5)),
        ("strong", MinimaxAgent(depth=3)),
        ("baseline-random", RandomAgent(seed=101)),
    ]

    entries = head_to_head(labeled_agents, num_games=20, seed=42)
    by_pair = {(entry.row, entry.column): entry for entry in entries}

    assert set(by_pair) == {
        ("weak", "strong"),
        ("strong", "weak"),
        ("weak", "baseline-random"),
        ("baseline-random", "weak"),
        ("strong", "baseline-random"),
        ("baseline-random", "strong"),
    }

    for entry in entries:
        assert entry.wins + entry.losses + entry.draws == entry.games == 20
        assert entry.win_rate == pytest.approx(entry.wins / entry.games)
        assert entry.loss_rate == pytest.approx(entry.losses / entry.games)
        assert entry.draw_rate == pytest.approx(entry.draws / entry.games)

    # Complementary pairs mirror each other exactly: they're read from opposite
    # sides of the same recorded match.
    for (row, column), entry in by_pair.items():
        mirror = by_pair[(column, row)]
        assert mirror.wins == entry.losses
        assert mirror.losses == entry.wins
        assert mirror.draws == entry.draws
        assert mirror.games == entry.games

    # Pinned exact values from fixed seeds.
    assert (by_pair[("strong", "weak")].wins, by_pair[("strong", "weak")].losses) == (19, 1)
    assert by_pair[("strong", "baseline-random")].win_rate == 1.0

    # Strength ordering: strong beats both random baselines far more often than
    # random beats random -- the "minimax >> random" direction check.
    assert by_pair[("strong", "weak")].win_rate > by_pair[("weak", "baseline-random")].win_rate
    assert (
        by_pair[("strong", "baseline-random")].win_rate
        > by_pair[("weak", "baseline-random")].win_rate
    )


# --- schema round-trip ----------------------------------------------------------------


def _sample_progress_log() -> ProgressLog:
    return ProgressLog(
        stages=(
            StageMetrics(
                label="baseline",
                cumulative_timesteps=0,
                vs_random=BaselineMetrics(
                    opponent="random",
                    total_games=10,
                    wins=4,
                    losses=5,
                    draws=1,
                    win_rate=0.4,
                    game_length_mean=5.5,
                    opening_move_distribution=((2, 4), (3, 6)),
                ),
                vs_minimax=BaselineMetrics(
                    opponent="minimax",
                    total_games=10,
                    wins=0,
                    losses=10,
                    draws=0,
                    win_rate=0.0,
                    game_length_mean=6.0,
                    opening_move_distribution=((3, 10),),
                ),
            ),
            StageMetrics(
                label="step-0002048",
                cumulative_timesteps=2_048,
                vs_random=BaselineMetrics(
                    opponent="random",
                    total_games=20,
                    wins=15,
                    losses=4,
                    draws=1,
                    win_rate=0.75,
                    game_length_mean=7.2,
                    opening_move_distribution=((3, 20),),
                ),
                vs_minimax=BaselineMetrics(
                    opponent="minimax",
                    total_games=20,
                    wins=6,
                    losses=13,
                    draws=1,
                    win_rate=0.3,
                    game_length_mean=8.1,
                    opening_move_distribution=((3, 16), (4, 4)),
                ),
            ),
        ),
        head_to_head=(
            HeadToHeadEntry(
                row="step-0002048",
                column="baseline",
                wins=15,
                losses=4,
                draws=1,
                games=20,
                win_rate=0.75,
                loss_rate=0.2,
                draw_rate=0.05,
            ),
            HeadToHeadEntry(
                row="baseline",
                column="step-0002048",
                wins=4,
                losses=15,
                draws=1,
                games=20,
                win_rate=0.2,
                loss_rate=0.75,
                draw_rate=0.05,
            ),
        ),
    )


def test_progress_log_round_trips_through_write_and_read(tmp_path: Path) -> None:
    progress = _sample_progress_log()

    path = write_progress_log(tmp_path / "progress.json", progress)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["format"] == PROGRESS_FORMAT == "gamesim.incremental-training/v2"
    assert len(raw["stages"]) == 2
    assert len(raw["head_to_head"]) == 2

    round_tripped = read_progress_log(path)
    assert round_tripped == progress


# --- per-stage match logs: evaluate_stage -> write_match_log -> read_match_log,
# and their recorded paths round-trip through the progress log itself -------------


def test_evaluate_stage_match_logs_round_trip_and_paths_persist_through_progress_log(
    tmp_path: Path,
) -> None:
    """Fix 2: evaluate_stage hands back real MatchLogs a caller can persist, and
    their (run-dir-relative) paths survive a full ProgressLog write/read cycle --
    all torch-free, using MinimaxAgent/RandomAgent stand-ins for a checkpoint."""
    stage, match_logs = evaluate_stage(
        MinimaxAgent(depth=2),
        label="checkpoint",
        cumulative_timesteps=2_048,
        random_agent=RandomAgent(seed=11),
        minimax_agent=MinimaxAgent(depth=1),
        num_games=4,
        seed=9,
    )
    assert set(match_logs) == {"random", "minimax"}

    run_dir = tmp_path / "run"
    match_log_paths: dict[str, str] = {}
    for opponent_key, match_log in match_logs.items():
        relative_path = Path("matches") / f"checkpoint-vs-{opponent_key}.zip"
        written_path = write_match_log(run_dir / relative_path, match_log)
        assert written_path == run_dir / relative_path
        match_log_paths[opponent_key] = relative_path.as_posix()

    for opponent_key, match_log in match_logs.items():
        round_tripped_log = read_match_log(run_dir / match_log_paths[opponent_key])
        assert round_tripped_log == match_log
        assert round_tripped_log.agent_a == "checkpoint"
        assert len(round_tripped_log.games) == 4

    stage_with_paths = replace(stage, match_log_paths=match_log_paths)
    progress = ProgressLog(stages=(stage_with_paths,))

    progress_path = write_progress_log(run_dir / "progress.json", progress)
    round_tripped_progress = read_progress_log(progress_path)

    assert round_tripped_progress == progress
    assert round_tripped_progress.stages[0].match_log_paths == match_log_paths
    assert round_tripped_progress.stages[0].match_log_paths == {
        "random": "matches/checkpoint-vs-random.zip",
        "minimax": "matches/checkpoint-vs-minimax.zip",
    }


def test_empty_progress_log_round_trips(tmp_path: Path) -> None:
    progress = ProgressLog(stages=())

    path = write_progress_log(tmp_path / "progress.json", progress)
    round_tripped = read_progress_log(path)

    assert round_tripped == progress
    assert round_tripped.stages == ()
    assert round_tripped.head_to_head == ()


def test_read_progress_log_rejects_the_old_v1_format(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    path.write_text(
        json.dumps(
            {
                "format": "gamesim.incremental-training/v1",
                "stages": [
                    {
                        "label": "baseline",
                        "cumulative_timesteps": 0,
                        "segment_timesteps": 0,
                        "checkpoint": "checkpoints/baseline.zip",
                        "match_log": "matches/baseline-vs-random.zip",
                        "wins": 500,
                        "losses": 400,
                        "draws": 100,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported progress log format"):
        read_progress_log(path)


def _sample_baseline_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "opponent": "random",
        "total_games": 10,
        "wins": 4,
        "losses": 5,
        "draws": 1,
        "win_rate": 0.4,
        "game_length_mean": 5.5,
        "opening_move_distribution": [[3, 1], [4, 2]],
    }
    base.update(overrides)
    return base


def test_read_progress_log_rejects_a_malformed_opening_distribution_pair(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    path.write_text(
        json.dumps(
            {
                "format": PROGRESS_FORMAT,
                "stages": [
                    {
                        "label": "baseline",
                        "cumulative_timesteps": 0,
                        "vs_random": _sample_baseline_dict(opening_move_distribution=[[3, 1], [4]]),
                        "vs_minimax": _sample_baseline_dict(opponent="minimax"),
                    }
                ],
                "head_to_head": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="opening_move_distribution"):
        read_progress_log(path)


def test_stage_metrics_match_log_paths_defaults_to_empty_when_missing(tmp_path: Path) -> None:
    """A stage record with no ``match_log_paths`` key (e.g. an older v2 log written
    before Fix 2) still reads, treated as having no persisted match logs."""
    path = tmp_path / "progress.json"
    path.write_text(
        json.dumps(
            {
                "format": PROGRESS_FORMAT,
                "stages": [
                    {
                        "label": "baseline",
                        "cumulative_timesteps": 0,
                        "vs_random": _sample_baseline_dict(),
                        "vs_minimax": _sample_baseline_dict(opponent="minimax"),
                    }
                ],
                "head_to_head": [],
            }
        ),
        encoding="utf-8",
    )

    progress = read_progress_log(path)

    assert progress.stages[0].match_log_paths == {}


def test_read_progress_log_rejects_a_stage_missing_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    path.write_text(
        json.dumps(
            {
                "format": PROGRESS_FORMAT,
                "stages": [{"label": "baseline", "cumulative_timesteps": 0}],
                "head_to_head": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing field"):
        read_progress_log(path)


def test_read_progress_log_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        read_progress_log(path)


def test_read_progress_log_rejects_a_non_object_top_level_value(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        read_progress_log(path)


# --- torch-free guarantee ------------------------------------------------------------


def test_progress_module_does_not_import_torch() -> None:
    script = "import sys\nimport gamesim.experiments.progress\nassert 'torch' not in sys.modules\n"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
