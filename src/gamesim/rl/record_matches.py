"""CLI to record replayable Connect Four matches for the local browser explorer."""

from __future__ import annotations

import argparse
from pathlib import Path

from gamesim.core.agent import RandomAgent
from gamesim.games.connect_four import ConnectFourEncoder
from gamesim.games.connect_four.engine import ConnectFourObservation
from gamesim.recording import record_match, write_match_log


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record trained-vs-random Connect Four games for replay in the local web UI."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/connect_four_maskable_ppo.zip"),
        help="Path to the trained MaskablePPO checkpoint.",
    )
    parser.add_argument("--games", type=int, default=100, help="Number of games to record.")
    parser.add_argument("--seed", type=int, default=0, help="Match seed.")
    parser.add_argument("--random-seed", type=int, default=123, help="Random agent seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/connect_four_trained_vs_random.zip"),
        help="Destination ZIP match archive.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load the policy, run the requested match, and write its replay log."""
    args = _parse_args(argv)
    if args.games < 1:
        raise ValueError("--games must be at least 1")

    from gamesim.rl.train import MaskablePolicyAgent

    trained_agent: MaskablePolicyAgent[ConnectFourObservation] = MaskablePolicyAgent.load(
        args.checkpoint,
        ConnectFourEncoder(),
    )
    random_agent = RandomAgent[ConnectFourObservation](seed=args.random_seed)
    log = record_match(
        trained_agent,
        random_agent,
        agent_a_name="trained",
        agent_b_name="random",
        num_games=args.games,
        seed=args.seed,
    )
    output_path = write_match_log(args.output, log)
    wins = sum(game.outcome == "agent_a" for game in log.games)
    losses = sum(game.outcome == "agent_b" for game in log.games)
    draws = sum(game.outcome == "draw" for game in log.games)
    print(f"Recorded {len(log.games)} games to {output_path} ({wins}-{losses}-{draws})")


if __name__ == "__main__":
    main()
