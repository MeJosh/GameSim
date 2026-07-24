"""Tests for the bounded incremental-training experiment scaffold."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from gamesim.experiments.incremental import (
    DEFAULT_EVALUATION_GAMES,
    SMOKE_TRAINING_SEGMENTS,
    StageResult,
    prepare_run_directory,
    write_progress,
)


def test_smoke_schedule_is_small_and_rollout_aligned() -> None:
    assert SMOKE_TRAINING_SEGMENTS == (2_048, 4_096, 8_192)
    assert sum(SMOKE_TRAINING_SEGMENTS) == 14_336
    assert DEFAULT_EVALUATION_GAMES == 1_000


def test_smoke_script_targets_the_checkout_source_tree() -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "run_incremental_smoke.py"

    script_namespace = runpy.run_path(str(script_path), run_name="smoke_script_test")

    assert script_namespace["_source_directory"]() == script_path.parents[1] / "src"


def test_prepare_run_directory_refuses_to_overwrite_an_existing_run(tmp_path: Path) -> None:
    run_dir = prepare_run_directory(tmp_path / "smoke")

    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "matches").is_dir()
    with pytest.raises(FileExistsError):
        prepare_run_directory(run_dir)


def test_write_progress_replaces_a_complete_json_index(tmp_path: Path) -> None:
    run_dir = prepare_run_directory(tmp_path / "smoke")
    stage = StageResult(
        label="baseline",
        cumulative_timesteps=0,
        segment_timesteps=0,
        checkpoint="checkpoints/baseline.zip",
        match_log="matches/baseline-vs-random.zip",
        wins=500,
        losses=400,
        draws=100,
    )

    progress_path = write_progress(run_dir, [stage])
    progress = json.loads(progress_path.read_text(encoding="utf-8"))

    assert progress["format"] == "gamesim.incremental-training/v1"
    assert progress["stages"] == [
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
    ]
