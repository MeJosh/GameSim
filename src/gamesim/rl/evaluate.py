"""Evaluation harness: play N games between two agents, alternating who moves first.

Built on the existing ``core.runner.run_game``, so it works with any ``Engine`` +
two ``Agent`` pairing (not just Connect Four's), and inherits the engine's
determinism -- a fixed ``seed`` here makes an entire evaluation run reproducible.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from gamesim.agents.scripted import MinimaxAgent
from gamesim.core.agent import Agent, RandomAgent
from gamesim.core.engine import Engine
from gamesim.core.runner import run_game
from gamesim.core.types import ActionT, AgentId, Observation
from gamesim.games.connect_four import ConnectFourEncoder, ConnectFourEngine
from gamesim.games.connect_four.engine import ConnectFourObservation


@dataclass(frozen=True)
class EvaluationResult:
    """Outcome of an ``evaluate()`` run, from ``agent_a``'s point of view.

    ``wins_a``/``wins_b``/``draws`` always sum to ``games``.
    """

    games: int
    wins_a: int
    wins_b: int
    draws: int

    @property
    def win_rate_a(self) -> float:
        return self.wins_a / self.games if self.games else 0.0

    @property
    def win_rate_b(self) -> float:
        return self.wins_b / self.games if self.games else 0.0


class EvaluationProgress(Protocol):
    """Receives per-game evaluation progress updates."""

    def start(self, total_games: int) -> None:
        """Start rendering progress for an evaluation run."""

    def update(self, completed_games: int, result: EvaluationResult) -> None:
        """Render an updated aggregate result after one or more games."""

    def stop(self) -> None:
        """Finish rendering progress for an evaluation run."""


class RichEvaluationProgress:
    """Rich-powered progress display for evaluation CLI runs."""

    def __init__(self, *, console: Console | None = None, label: str = "Evaluating") -> None:
        self._console = console or Console()
        self._label = label
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None

    def start(self, total_games: int) -> None:
        progress = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn(
                "wins={task.fields[wins_a]} "
                "losses={task.fields[wins_b]} "
                "draws={task.fields[draws]} "
                "win={task.fields[win_rate]}"
            ),
            console=self._console,
        )
        self._progress = progress
        self._task_id = progress.add_task(
            self._label,
            total=total_games,
            wins_a=0,
            wins_b=0,
            draws=0,
            win_rate="0.0%",
        )
        progress.start()

    def update(self, completed_games: int, result: EvaluationResult) -> None:
        if self._progress is None or self._task_id is None:
            return
        self._progress.update(
            self._task_id,
            completed=completed_games,
            wins_a=result.wins_a,
            wins_b=result.wins_b,
            draws=result.draws,
            win_rate=f"{result.win_rate_a * 100:.1f}%",
        )

    def stop(self) -> None:
        if self._progress is not None:
            self._progress.stop()
        self._progress = None
        self._task_id = None


def evaluate(
    engine: Engine[Observation, ActionT],
    agent_a: Agent[Observation, ActionT],
    agent_b: Agent[Observation, ActionT],
    *,
    num_games: int,
    seed: int | None = None,
    progress: EvaluationProgress | None = None,
) -> EvaluationResult:
    """Play ``num_games`` between ``agent_a`` and ``agent_b`` via ``run_game``.

    Which agent occupies seat ``AgentId(0)`` (moves first) alternates every game --
    ``agent_a`` goes first on even-indexed games (0, 2, 4, ...), ``agent_b`` on
    odd-indexed ones -- so neither agent accrues a first-move advantage over the
    run. Per-game seeds are drawn from a ``numpy`` generator seeded with ``seed``,
    so the whole run (outcomes, and every individual game) is reproducible.
    """
    rng = np.random.default_rng(seed)
    wins_a = 0
    wins_b = 0
    draws = 0

    if progress is not None:
        progress.start(num_games)

    try:
        for game_index in range(num_games):
            game_seed = int(rng.integers(0, 2**31 - 1))
            a_moves_first = game_index % 2 == 0
            if a_moves_first:
                seat_a, seat_b = AgentId(0), AgentId(1)
            else:
                seat_a, seat_b = AgentId(1), AgentId(0)

            agents = {seat_a: agent_a, seat_b: agent_b}
            rewards = run_game(engine, agents, seed=game_seed)

            reward_a = rewards[seat_a]
            reward_b = rewards[seat_b]
            if reward_a > reward_b:
                wins_a += 1
            elif reward_b > reward_a:
                wins_b += 1
            else:
                draws += 1

            if progress is not None:
                partial_result = EvaluationResult(
                    games=game_index + 1,
                    wins_a=wins_a,
                    wins_b=wins_b,
                    draws=draws,
                )
                progress.update(game_index + 1, partial_result)
    finally:
        if progress is not None:
            progress.stop()

    return EvaluationResult(games=num_games, wins_a=wins_a, wins_b=wins_b, draws=draws)


OpponentName = Literal["random", "minimax"]


def _make_opponent(
    opponent: OpponentName, *, seed: int | None, minimax_depth: int
) -> Agent[ConnectFourObservation, int]:
    if opponent == "random":
        return RandomAgent[ConnectFourObservation](seed=seed)
    if opponent == "minimax":
        return MinimaxAgent(depth=minimax_depth)
    raise ValueError(f"Unknown opponent: {opponent}")


def _format_result(opponent: str, result: EvaluationResult) -> str:
    win_pct = result.win_rate_a * 100
    opponent_pct = result.win_rate_b * 100
    return (
        f"vs {opponent}: "
        f"{result.wins_a}-{result.wins_b}-{result.draws} "
        f"({win_pct:.1f}% policy win rate, {opponent_pct:.1f}% opponent win rate)"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Connect Four MaskablePPO checkpoint against baselines."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/connect_four_maskable_ppo.zip"),
        help="Path to a trained MaskablePPO checkpoint.",
    )
    parser.add_argument(
        "--opponent",
        choices=["random", "minimax", "all"],
        default="all",
        help="Baseline opponent to evaluate against.",
    )
    parser.add_argument("--games", type=int, default=100, help="Number of games per opponent.")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation seed.")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=123,
        help="Seed for the random baseline agent.",
    )
    parser.add_argument(
        "--minimax-depth",
        type=int,
        default=3,
        help="Search depth for the minimax baseline.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="'cpu', 'cuda', or 'auto'.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar and live evaluation stats.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    from gamesim.rl.train import MaskablePolicyAgent

    encoder = ConnectFourEncoder()
    policy_agent: MaskablePolicyAgent[ConnectFourObservation] = MaskablePolicyAgent.load(
        args.checkpoint,
        encoder,
        device=args.device,
    )

    opponents: list[OpponentName]
    if args.opponent == "all":
        opponents = ["random", "minimax"]
    else:
        opponents = [args.opponent]

    for index, opponent_name in enumerate(opponents):
        opponent = _make_opponent(
            opponent_name,
            seed=args.random_seed,
            minimax_depth=args.minimax_depth,
        )
        result = evaluate(
            ConnectFourEngine(),
            policy_agent,
            opponent,
            num_games=args.games,
            seed=args.seed + index,
            progress=None
            if args.no_progress
            else RichEvaluationProgress(label=f"Evaluating vs {opponent_name}"),
        )
        print(_format_result(opponent_name, result))


if __name__ == "__main__":
    main()


__all__ = ["EvaluationResult", "evaluate", "main"]
