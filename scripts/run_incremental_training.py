"""Run a staged incremental Connect Four training experiment.

Starts a baseline (untrained policy vs. random/minimax), then trains for exactly
``--num-stages`` doubling-length segments -- each one snapshotted and evaluated vs.
random, vs. minimax, and head-to-head against every earlier snapshot. The whole
schedule is fixed up front (no time budget or other early-stopping), so with the
live progress display (on by default) every stage's row -- including ones that
haven't started -- is visible from the beginning, with an estimated duration for
each once training speed is known. See
``gamesim.experiments.incremental.run_incremental_experiment`` for details.

Needs the ``rl`` extra (``make install-rl``) and is meant to run on your own machine,
not the dev sandbox -- see plans/phase-02-drl-selfplay.md, "Sandbox vs. local".

Usage:
    python scripts/run_incremental_training.py --run-dir runs/incremental-001
    python scripts/run_incremental_training.py --run-dir runs/incremental-001 \\
        --num-stages 8 --initial-segment-timesteps 8192 --growth-factor 1.5

Once it finishes, open the results with:
    make progress-report RUN_DIR=runs/incremental-001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _source_directory() -> Path:
    """Return the local source tree when the script runs from a checkout."""
    return Path(__file__).resolve().parents[1] / "src"


source_directory = _source_directory()
if str(source_directory) not in sys.path:
    sys.path.insert(0, str(source_directory))

from gamesim.experiments.incremental import (
    DEFAULT_GROWTH_FACTOR,
    DEFAULT_INCREMENTAL_EVALUATION_GAMES,
    DEFAULT_INCREMENTAL_HEAD_TO_HEAD_GAMES,
    DEFAULT_INITIAL_SEGMENT_TIMESTEPS,
    DEFAULT_NUM_STAGES,
    run_incremental_experiment,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a baseline plus a fixed number of doubling-length incremental PPO training stages."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="New output directory; the script refuses to overwrite an existing one.",
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        default=DEFAULT_NUM_STAGES,
        help="Exactly how many training stages to run after the baseline.",
    )
    parser.add_argument(
        "--initial-segment-timesteps",
        type=int,
        default=DEFAULT_INITIAL_SEGMENT_TIMESTEPS,
        help="Timesteps trained in the first (shortest) stage.",
    )
    parser.add_argument(
        "--growth-factor",
        type=float,
        default=DEFAULT_GROWTH_FACTOR,
        help="Multiplier applied to the segment length after every stage.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Training and evaluation seed.")
    parser.add_argument(
        "--evaluation-games",
        type=int,
        default=DEFAULT_INCREMENTAL_EVALUATION_GAMES,
        help="Recorded games vs. random and vs. minimax after each stage.",
    )
    parser.add_argument(
        "--head-to-head-games",
        type=int,
        default=DEFAULT_INCREMENTAL_HEAD_TO_HEAD_GAMES,
        help="Recorded games for each stage-vs-earlier-stage match.",
    )
    parser.add_argument("--random-seed", type=int, default=123, help="Random opponent seed.")
    parser.add_argument(
        "--minimax-depth", type=int, default=4, help="Search depth for the minimax baseline."
    )
    parser.add_argument("--device", default="cpu", help="PPO device, e.g. cpu or cuda.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the live progress display and fall back to plain print output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_incremental_experiment(
        run_dir=args.run_dir,
        num_stages=args.num_stages,
        initial_segment_timesteps=args.initial_segment_timesteps,
        growth_factor=args.growth_factor,
        seed=args.seed,
        evaluation_games=args.evaluation_games,
        head_to_head_games=args.head_to_head_games,
        random_seed=args.random_seed,
        minimax_depth=args.minimax_depth,
        device=args.device,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
