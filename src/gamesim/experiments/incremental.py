"""Bounded incremental-training experiment support (TORCH-DEPENDENT driver).

The experiment keeps one live PPO model across short learning segments. It is kept
outside the normal training CLI while the workflow is still exploratory.

Per checkpoint ("stage"), this now records a richer evaluation via the torch-free
``gamesim.experiments.progress`` layer (see docs/adr/0009-offline-analysis-and-
reporting.md and plans/phase-03-visualization.md Slice 3d): winrate/game-length/
opening-move stats vs both a random and a minimax baseline (``evaluate_stage``), plus
a head-to-head record against every earlier stage (``head_to_head``). ``evaluate_stage``
also returns the two recorded ``MatchLog``s behind those stats; this driver writes them
to ``<run_dir>/matches/`` via ``write_match_log`` and records their relative paths on
each stage's ``match_log_paths``, so a checkpoint's actual games -- not just aggregate
stats -- stay openable later (e.g. via ``gamesim.viz.report``). The resulting
``ProgressLog`` is persisted as the versioned ``progress.json`` (``PROGRESS_FORMAT``)
via ``write_progress_log``.

All sb3/torch imports stay local to ``run_smoke_experiment``, so importing this
module -- and everything it wires together from ``gamesim.experiments.progress`` --
never requires torch to be installed. Only *calling* ``run_smoke_experiment`` does.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from gamesim.agents.scripted import MinimaxAgent
from gamesim.core.agent import Agent, RandomAgent
from gamesim.experiments.progress import (
    HeadToHeadEntry,
    ProgressLog,
    StageMetrics,
    evaluate_stage,
    head_to_head,
    write_progress_log,
)
from gamesim.games.connect_four import ConnectFourEncoder, ConnectFourEngine
from gamesim.games.connect_four.engine import ConnectFourObservation
from gamesim.games.connect_four.state import NUM_COLUMNS
from gamesim.recording.match_log import write_match_log

SMOKE_TRAINING_SEGMENTS: tuple[int, ...] = (2_048, 4_096, 8_192)
DEFAULT_EVALUATION_GAMES = 1_000
DEFAULT_HEAD_TO_HEAD_GAMES = 200
DEFAULT_MINIMAX_DEPTH = 4


def prepare_run_directory(run_dir: str | Path) -> Path:
    """Create an empty experiment directory without ever overwriting another run."""
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    (output_dir / "matches").mkdir()
    return output_dir


def run_smoke_experiment(
    *,
    run_dir: str | Path,
    seed: int = 0,
    evaluation_games: int = DEFAULT_EVALUATION_GAMES,
    head_to_head_games: int = DEFAULT_HEAD_TO_HEAD_GAMES,
    random_seed: int = 123,
    minimax_depth: int = DEFAULT_MINIMAX_DEPTH,
    device: str = "cpu",
) -> ProgressLog:
    """Run a baseline plus three short, cumulative PPO training segments.

    Requires the ``rl`` extra. All model imports stay local so planning and tests
    remain usable without torch or sb3-contrib installed. Returns the full
    :class:`~gamesim.experiments.progress.ProgressLog` for the run (also persisted
    as ``<run_dir>/progress.json`` after every stage).
    """
    if evaluation_games < 1:
        raise ValueError("evaluation_games must be at least 1")

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    from gamesim.rl.selfplay_env import SelfPlayEnv
    from gamesim.rl.train import MaskablePolicyAgent, MaskablePPOSnapshotOpponent

    output_dir = prepare_run_directory(run_dir)
    encoder = ConnectFourEncoder()
    base_env: SelfPlayEnv[ConnectFourObservation] = SelfPlayEnv(
        ConnectFourEngine(),
        encoder,
        num_actions=NUM_COLUMNS,
        seed=seed,
    )
    env = ActionMasker(base_env, lambda current_env: current_env.action_masks())
    model = MaskablePPO("MlpPolicy", env, seed=seed, device=device, verbose=0)

    stages: list[StageMetrics] = []
    head_to_head_entries: list[HeadToHeadEntry] = []
    earlier_agents: list[tuple[str, Agent[ConnectFourObservation, int]]] = []

    def record_stage(label: str, cumulative_timesteps: int) -> StageMetrics:
        """Checkpoint the live model, evaluate it, and persist the updated progress log.

        Also writes the representative recorded ``MatchLog`` for each baseline
        evaluation (vs random, vs minimax) to ``<run_dir>/matches/`` so the checkpoint's
        actual games -- not just aggregate stats -- can later be opened in the
        standalone HTML match report (``gamesim.viz.report``); their run-dir-relative
        paths are recorded on the persisted ``StageMetrics.match_log_paths``.
        """
        checkpoint_path = output_dir / "checkpoints" / f"{label}.zip"
        model.save(checkpoint_path)
        agent: MaskablePolicyAgent[ConnectFourObservation] = MaskablePolicyAgent(model, encoder)

        stage, match_logs = evaluate_stage(
            agent,
            label=label,
            cumulative_timesteps=cumulative_timesteps,
            random_agent=RandomAgent[ConnectFourObservation](seed=random_seed),
            minimax_agent=MinimaxAgent(depth=minimax_depth),
            num_games=evaluation_games,
            seed=seed,
        )
        match_log_paths: dict[str, str] = {}
        for opponent_key, match_log in match_logs.items():
            relative_path = Path("matches") / f"{label}-vs-{opponent_key}.zip"
            write_match_log(output_dir / relative_path, match_log)
            match_log_paths[opponent_key] = relative_path.as_posix()
        stage = replace(stage, match_log_paths=match_log_paths)
        stages.append(stage)

        for earlier_label, earlier_agent in earlier_agents:
            head_to_head_entries.extend(
                head_to_head(
                    [(label, agent), (earlier_label, earlier_agent)],
                    num_games=head_to_head_games,
                    seed=seed,
                )
            )
        earlier_agents.append((label, agent))

        write_progress_log(
            output_dir / "progress.json",
            ProgressLog(stages=tuple(stages), head_to_head=tuple(head_to_head_entries)),
        )
        print(
            f"{label}: vs random {stage.vs_random.win_rate:.1%} "
            f"({stage.vs_random.wins}-{stage.vs_random.losses}-{stage.vs_random.draws}), "
            f"vs minimax {stage.vs_minimax.win_rate:.1%} "
            f"({stage.vs_minimax.wins}-{stage.vs_minimax.losses}-{stage.vs_minimax.draws}) "
            f"over {evaluation_games:,} games each"
        )
        return stage

    record_stage("baseline", 0)

    cumulative_timesteps = 0
    for segment_timesteps in SMOKE_TRAINING_SEGMENTS:
        print(f"Training next {segment_timesteps:,} timesteps...")
        model.learn(total_timesteps=segment_timesteps, reset_num_timesteps=False)
        cumulative_timesteps += segment_timesteps
        record_stage(f"step-{cumulative_timesteps:07d}", cumulative_timesteps)
        checkpoint_path = output_dir / "checkpoints" / f"step-{cumulative_timesteps:07d}.zip"
        base_env.set_opponent(MaskablePPOSnapshotOpponent.load(checkpoint_path))

    return ProgressLog(stages=tuple(stages), head_to_head=tuple(head_to_head_entries))
