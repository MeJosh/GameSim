"""Run a small, bounded incremental Connect Four training experiment.

Usage (after installing the ``rl`` extra):
    python scripts/run_incremental_smoke.py --run-dir runs/incremental-smoke-001
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _source_directory() -> Path:
    """Return the local source tree when the script runs from a checkout."""
    return Path(__file__).resolve().parents[1] / "src"


source_directory = _source_directory()
if str(source_directory) not in sys.path:
    sys.path.insert(0, str(source_directory))

from gamesim.experiments.incremental import DEFAULT_EVALUATION_GAMES, run_smoke_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a baseline and three small incremental PPO training stages."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="New output directory; the script refuses to overwrite an existing one.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Training and evaluation seed.")
    parser.add_argument(
        "--evaluation-games",
        type=int,
        default=DEFAULT_EVALUATION_GAMES,
        help="Recorded trained-vs-random games after each stage.",
    )
    parser.add_argument("--random-seed", type=int, default=123, help="Random opponent seed.")
    parser.add_argument("--device", default="cpu", help="PPO device, e.g. cpu or cuda.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_smoke_experiment(
        run_dir=args.run_dir,
        seed=args.seed,
        evaluation_games=args.evaluation_games,
        random_seed=args.random_seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
