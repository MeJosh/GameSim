"""CLI to record replayable Connect Four matches between selectable agents.

Each side is chosen independently via an *agent spec* string:

- ``random`` -- ``RandomAgent``.
- ``minimax`` or ``minimax:<depth>`` -- ``MinimaxAgent`` (default depth 4).
- ``trained:<checkpoint-path>`` -- a saved ``MaskablePPO`` policy, loaded via
  ``gamesim.rl.train.MaskablePolicyAgent``.

This lets any matchup be recorded: minimax-v-random, minimax-v-minimax,
model-v-model, model-v-random, etc.

Trained loading stays torch-isolated: ``_load_trained_agent`` imports
``gamesim.rl.train`` (itself torch-free at import time -- see its module
docstring) and only its ``.load()`` call reaches into ``sb3_contrib``/torch. So
importing this module, or calling ``build_agent`` with a non-``trained`` spec,
never requires torch/sb3-contrib to be installed (see
plans/phase-03-visualization.md, Slice 3a, and plans/phase-02-drl-selfplay.md,
"Sandbox vs. local").
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gamesim.agents.scripted import MinimaxAgent
from gamesim.core.agent import Agent, RandomAgent
from gamesim.games.connect_four import ConnectFourEncoder
from gamesim.games.connect_four.engine import ConnectFourObservation
from gamesim.recording import record_match, write_match_log

ConnectFourAgent = Agent[ConnectFourObservation, int]

_DEFAULT_MINIMAX_DEPTH = 4


def _load_trained_agent(checkpoint_path: Path) -> ConnectFourAgent:
    """Load a saved MaskablePPO checkpoint as an ``Agent``. Torch-dependent.

    Kept as a standalone, monkeypatchable function (rather than inlined in
    ``build_agent``) so tests can exercise the ``trained:`` spec path without
    torch/sb3-contrib installed.
    """
    from gamesim.rl.train import MaskablePolicyAgent  # local import: torch-dependent

    return MaskablePolicyAgent.load(checkpoint_path, ConnectFourEncoder())


def build_agent(spec: str, *, seed: int | None = None) -> ConnectFourAgent:
    """Build an agent from an agent-spec string (see module docstring).

    ``seed`` is only consulted for ``random`` specs; ``minimax`` is deterministic
    and ``trained`` is already fixed by its checkpoint.
    """
    kind, _, rest = spec.partition(":")
    if kind == "random":
        return RandomAgent[ConnectFourObservation](seed=seed)
    if kind == "minimax":
        if rest:
            try:
                depth = int(rest)
            except ValueError as exc:
                raise ValueError(f"invalid minimax depth: {rest!r}") from exc
        else:
            depth = _DEFAULT_MINIMAX_DEPTH
        return MinimaxAgent(depth=depth)
    if kind == "trained":
        if not rest:
            raise ValueError("trained agent spec must include a checkpoint path: trained:<path>")
        return _load_trained_agent(Path(rest))
    raise ValueError(
        f"unknown agent spec: {spec!r} (expected random, minimax[:depth], or trained:<path>)"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a Connect Four match between two selectable agents for replay."
    )
    parser.add_argument(
        "--agent-a",
        default="trained:checkpoints/connect_four_maskable_ppo.zip",
        help="Agent spec for agent A: random, minimax[:depth], or trained:<checkpoint-path>.",
    )
    parser.add_argument(
        "--agent-b",
        default="random",
        help="Agent spec for agent B: random, minimax[:depth], or trained:<checkpoint-path>.",
    )
    parser.add_argument(
        "--agent-a-name",
        default=None,
        help="Label for agent A in the recorded log (defaults to its --agent-a spec).",
    )
    parser.add_argument(
        "--agent-b-name",
        default=None,
        help="Label for agent B in the recorded log (defaults to its --agent-b spec).",
    )
    parser.add_argument(
        "--agent-a-seed",
        type=int,
        default=123,
        help="Seed for agent A, consulted only if --agent-a is 'random'.",
    )
    parser.add_argument(
        "--agent-b-seed",
        type=int,
        default=456,
        help="Seed for agent B, consulted only if --agent-b is 'random'.",
    )
    parser.add_argument("--games", type=int, default=100, help="Number of games to record.")
    parser.add_argument("--seed", type=int, default=0, help="Match seed (per-game seeding).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/connect_four_match.zip"),
        help="Destination ZIP match archive.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build the requested agents, run the match, and write its replay log."""
    args = _parse_args(argv)
    if args.games < 1:
        raise ValueError("--games must be at least 1")

    agent_a = build_agent(args.agent_a, seed=args.agent_a_seed)
    agent_b = build_agent(args.agent_b, seed=args.agent_b_seed)
    agent_a_name = args.agent_a_name or args.agent_a
    agent_b_name = args.agent_b_name or args.agent_b

    log = record_match(
        agent_a,
        agent_b,
        agent_a_name=agent_a_name,
        agent_b_name=agent_b_name,
        num_games=args.games,
        seed=args.seed,
    )
    output_path = write_match_log(args.output, log)
    wins = sum(game.outcome == "agent_a" for game in log.games)
    losses = sum(game.outcome == "agent_b" for game in log.games)
    draws = sum(game.outcome == "draw" for game in log.games)
    print(
        f"Recorded {len(log.games)} games ({agent_a_name} vs {agent_b_name}) to "
        f"{output_path} ({wins}-{losses}-{draws})"
    )


if __name__ == "__main__":
    main()
