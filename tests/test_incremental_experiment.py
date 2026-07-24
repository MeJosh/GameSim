"""Tests for the bounded incremental-training experiment scaffold.

This module only exercises the torch-free scaffolding (constants, directory
handling, the smoke script's source-tree wiring) -- ``run_smoke_experiment`` itself
requires sb3-contrib/torch and is not exercised here. Its per-stage evaluation logic
(``evaluate_stage``/``head_to_head``/the versioned progress schema) lives in
``gamesim.experiments.progress`` and is fully tested torch-free in
``tests/experiments/test_progress.py``.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from gamesim.experiments.incremental import (
    DEFAULT_EVALUATION_GAMES,
    DEFAULT_HEAD_TO_HEAD_GAMES,
    DEFAULT_MINIMAX_DEPTH,
    SMOKE_TRAINING_SEGMENTS,
    prepare_run_directory,
)


def test_smoke_schedule_is_small_and_rollout_aligned() -> None:
    assert SMOKE_TRAINING_SEGMENTS == (2_048, 4_096, 8_192)
    assert sum(SMOKE_TRAINING_SEGMENTS) == 14_336
    assert DEFAULT_EVALUATION_GAMES == 1_000
    assert DEFAULT_HEAD_TO_HEAD_GAMES == 200
    assert DEFAULT_MINIMAX_DEPTH == 4


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
