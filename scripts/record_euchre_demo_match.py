"""Record a demo Euchre match log (RandomAgent vs RandomAgent, torch-free).

There's no trained Euchre policy yet (DRL/encoder work is a follow-up to Phase 4 --
see plans/phase-04-euchre.md), so this is the fastest way to get an
``EuchreMatchLog`` to feed ``gamesim.viz.report_euchre`` and actually see the
visualizer render something real. No RL extras needed -- just numpy.

Usage:
    python scripts/record_euchre_demo_match.py --output logs/euchre_demo_match.zip
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

from gamesim.core.agent import RandomAgent
from gamesim.games.euchre import EuchreRules
from gamesim.recording import record_euchre_match, write_euchre_match_log


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a demo Euchre match log (RandomAgent vs RandomAgent)."
    )
    parser.add_argument("--num-hands", type=int, default=50, help="Hands to record (default 50).")
    parser.add_argument("--seed", type=int, default=0, help="Match seed (default 0).")
    parser.add_argument(
        "--stick-the-dealer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable stick-the-dealer (default on -- see games.euchre.state.EuchreRules).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/euchre_demo_match.zip"),
        help="Destination match-log ZIP path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    log = record_euchre_match(
        RandomAgent(seed=args.seed * 2 + 1),
        RandomAgent(seed=args.seed * 2 + 2),
        team_a_name="RandomA",
        team_b_name="RandomB",
        num_hands=args.num_hands,
        seed=args.seed,
        rules=EuchreRules(stick_the_dealer=args.stick_the_dealer),
    )
    output_path = write_euchre_match_log(args.output, log)
    print(f"Recorded {len(log.games)} hands to {output_path}")


if __name__ == "__main__":
    main()
