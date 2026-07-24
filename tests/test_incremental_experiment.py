"""Tests for the bounded incremental-training experiment scaffold.

This module only exercises the torch-free scaffolding (constants, directory
handling, argument validation, the scripts' source-tree wiring, and the live
progress display) -- neither ``run_smoke_experiment`` nor
``run_incremental_experiment`` is exercised end-to-end here, since both require
sb3-contrib/torch. Both validate their arguments *before* importing torch, so the
invalid-argument paths are exercised directly. ``_StagedRunDisplay`` only depends on
``rich`` (a core dependency, unlike torch/sb3), so it's exercised directly too --
without starting its live-refresh thread, since these tests only need to check the
task state it manages, not its rendering. Per-stage evaluation logic
(``evaluate_stage``/``head_to_head``/the versioned progress schema) lives in
``gamesim.experiments.progress`` and is fully tested torch-free in
``tests/experiments/test_progress.py``.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

from gamesim.experiments.incremental import (
    DEFAULT_EVALUATION_GAMES,
    DEFAULT_GROWTH_FACTOR,
    DEFAULT_HEAD_TO_HEAD_GAMES,
    DEFAULT_INCREMENTAL_EVALUATION_GAMES,
    DEFAULT_INCREMENTAL_HEAD_TO_HEAD_GAMES,
    DEFAULT_INITIAL_SEGMENT_TIMESTEPS,
    DEFAULT_MINIMAX_DEPTH,
    DEFAULT_NUM_STAGES,
    SMOKE_TRAINING_SEGMENTS,
    _format_duration,
    _format_magnitude,
    _StagedRunDisplay,
    prepare_run_directory,
    run_incremental_experiment,
)


def test_smoke_schedule_is_small_and_rollout_aligned() -> None:
    assert SMOKE_TRAINING_SEGMENTS == (2_048, 4_096, 8_192)
    assert sum(SMOKE_TRAINING_SEGMENTS) == 14_336
    assert DEFAULT_EVALUATION_GAMES == 1_000
    assert DEFAULT_HEAD_TO_HEAD_GAMES == 200
    assert DEFAULT_MINIMAX_DEPTH == 4


def test_incremental_schedule_defaults() -> None:
    assert DEFAULT_NUM_STAGES == 6
    assert DEFAULT_INITIAL_SEGMENT_TIMESTEPS == 4_096
    assert DEFAULT_GROWTH_FACTOR == 2.0
    # Lower than the smoke defaults: head-to-head cost grows with the number of
    # stages, and the vs-minimax evaluation leg (~0.9s/game at depth 4) is a fixed
    # per-stage tax regardless of that stage's training length, so keeping
    # per-match game counts modest leaves most of the run's wall time for the
    # training segments the schedule is meant to grow.
    assert DEFAULT_INCREMENTAL_EVALUATION_GAMES == 100
    assert DEFAULT_INCREMENTAL_HEAD_TO_HEAD_GAMES == 100


def test_smoke_script_targets_the_checkout_source_tree() -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "run_incremental_smoke.py"

    script_namespace = runpy.run_path(str(script_path), run_name="smoke_script_test")

    assert script_namespace["_source_directory"]() == script_path.parents[1] / "src"


def test_incremental_training_script_targets_the_checkout_source_tree() -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "run_incremental_training.py"

    script_namespace = runpy.run_path(str(script_path), run_name="incremental_training_script_test")

    assert script_namespace["_source_directory"]() == script_path.parents[1] / "src"


def test_prepare_run_directory_refuses_to_overwrite_an_existing_run(tmp_path: Path) -> None:
    run_dir = prepare_run_directory(tmp_path / "smoke")

    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "matches").is_dir()
    with pytest.raises(FileExistsError):
        prepare_run_directory(run_dir)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"evaluation_games": 0}, "evaluation_games"),
        ({"head_to_head_games": 0}, "head_to_head_games"),
        ({"num_stages": 0}, "num_stages"),
        ({"initial_segment_timesteps": 0}, "initial_segment_timesteps"),
        ({"growth_factor": 1}, "growth_factor"),
        ({"growth_factor": 0.5}, "growth_factor"),
    ],
)
def test_run_incremental_experiment_validates_arguments_before_any_torch_import(
    tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    """Bad arguments raise ``ValueError`` without requiring sb3-contrib/torch.

    Regression guard for the ordering in ``run_incremental_experiment``: all
    validation must run before ``from sb3_contrib import MaskablePPO`` so this test
    (which runs torch-free, like the rest of this module) actually exercises it --
    if a check moved below the import, this would fail with ``ModuleNotFoundError``
    instead of the expected ``ValueError``.
    """
    with pytest.raises(ValueError, match=message):
        run_incremental_experiment(run_dir=tmp_path / "incremental", **kwargs)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0:00"),
        (59, "0:59"),
        (60, "1:00"),
        (3_599, "59:59"),
        (3_600, "1:00:00"),
        (3_661, "1:01:01"),
        (-5, "0:00"),  # never renders a negative duration
    ],
)
def test_format_duration(seconds: int, expected: str) -> None:
    assert _format_duration(seconds) == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, "0"),
        (999, "999"),
        (1_000, "1k"),
        (10_000, "10k"),
        (4_096, "4.1k"),
        (8_192, "8.2k"),
        (16_384, "16.4k"),
        (32_768, "32.8k"),
        (65_536, "65.5k"),
        (131_072, "131.1k"),
        (1_000_000, "1M"),
        (1_048_576, "1M"),
        (4_000_000, "4M"),
        (2_500_000_000, "2.5B"),
    ],
)
def test_format_magnitude(count: int, expected: str) -> None:
    assert _format_magnitude(count) == expected


def test_staged_run_display_starts_every_row_pending(tmp_path: Path) -> None:
    """The whole schedule is known up front, so every stage gets a row immediately
    -- not just whichever stage happens to be running -- all starting "pending".
    """
    run_dir = tmp_path / "staged-run"
    segment_timesteps = [100, 200, 400]
    display = _StagedRunDisplay(run_dir=run_dir, num_stages=3, segment_timesteps=segment_timesteps)

    assert len(display.stage_tasks) == 3
    run_task = display._task(display.run_task)
    assert run_task.fields["status"] == "summary"
    assert run_task.total == 3
    assert run_task.completed == 0

    baseline_task = display._task(display.baseline_task)
    assert baseline_task.fields["status"] == "pending"
    assert baseline_task.start_time is None  # not started yet -- no ticking clock

    for stage_number, (task_id, steps) in enumerate(
        zip(display.stage_tasks, segment_timesteps, strict=True), start=1
    ):
        task = display._task(task_id)
        assert task.fields["status"] == "pending"
        assert task.total == steps
        assert task.start_time is None
        assert task.description == f"Stage {stage_number} ({_format_magnitude(steps)})"


def test_staged_run_display_tracks_a_stage_through_training_and_evaluating(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "staged-run"
    display = _StagedRunDisplay(run_dir=run_dir, num_stages=2, segment_timesteps=[100, 200])
    first_stage = display.stage_tasks[0]

    display.begin_training(first_stage, 100)
    task = display._task(first_stage)
    assert task.fields["status"] == "training"
    assert task.total == 100
    assert task.completed == 0
    assert task.start_time is not None  # clock now running

    display.update_training(first_stage, 42)
    assert display._task(first_stage).completed == 42

    display.begin_evaluating(first_stage, "evaluating vs random & minimax")
    task = display._task(first_stage)
    assert task.fields["status"] == "evaluating"
    assert task.fields["phase"] == "evaluating vs random & minimax"


def test_staged_run_display_finish_stage_marks_complete_and_refreshes_estimates(
    tmp_path: Path,
) -> None:
    """Finishing a training stage measures its throughput and refreshes every
    still-pending row's estimate from it; finishing the (training-free) baseline
    only refreshes the evaluation-overhead component (no throughput data yet), so
    pending rows still read "estimating…" until a real training stage completes.
    """
    run_dir = tmp_path / "staged-run"
    display = _StagedRunDisplay(run_dir=run_dir, num_stages=2, segment_timesteps=[100, 200])

    display.begin_evaluating(display.baseline_task, "evaluating vs random & minimax")
    display.finish_stage(display.baseline_task, training_timesteps=None, training_seconds=None)
    baseline_task = display._task(display.baseline_task)
    assert baseline_task.fields["status"] == "complete"
    assert baseline_task.completed == baseline_task.total
    run_task = display._task(display.run_task)
    assert run_task.completed == 1

    first_stage, second_stage = display.stage_tasks
    assert display._task(second_stage).fields.get("estimate_seconds") is None  # no data yet

    display.begin_training(first_stage, 100)
    display.update_training(first_stage, 100)
    display.begin_evaluating(first_stage, "evaluating vs random & minimax")
    display.finish_stage(first_stage, training_timesteps=100, training_seconds=2.0)

    first_task = display._task(first_stage)
    assert first_task.fields["status"] == "complete"
    assert first_task.completed == first_task.total
    assert display._task(display.run_task).completed == 2

    # Stage 2 is 200 timesteps at the same measured throughput (100 steps / 2s) ->
    # ~4s of training, plus whatever (here, near-instant) evaluation-only duration
    # baseline measured -- so the estimate should be at least that 4s floor.
    second_estimate = display._task(second_stage).fields["estimate_seconds"]
    assert second_estimate is not None
    assert second_estimate >= 4.0


def test_staged_run_display_print_does_not_raise(tmp_path: Path) -> None:
    display = _StagedRunDisplay(
        run_dir=tmp_path / "staged-run", num_stages=1, segment_timesteps=[100]
    )
    display.print("stage 1: vs random 80.0% (8-1-1)")
