"""Bounded incremental-training experiment support.

The experiment keeps one live PPO model across short learning segments. It is kept
outside the normal training CLI while the workflow is still exploratory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gamesim.core.agent import RandomAgent
from gamesim.games.connect_four import ConnectFourEncoder, ConnectFourEngine
from gamesim.games.connect_four.engine import ConnectFourObservation
from gamesim.games.connect_four.state import NUM_COLUMNS
from gamesim.recording import record_match, write_match_log

SMOKE_TRAINING_SEGMENTS: tuple[int, ...] = (2_048, 4_096, 8_192)
DEFAULT_EVALUATION_GAMES = 1_000


@dataclass(frozen=True)
class StageResult:
    """One baseline or trained checkpoint evaluation in an incremental run."""

    label: str
    cumulative_timesteps: int
    segment_timesteps: int
    checkpoint: str
    match_log: str
    wins: int
    losses: int
    draws: int


def prepare_run_directory(run_dir: str | Path) -> Path:
    """Create an empty experiment directory without ever overwriting another run."""
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    (output_dir / "matches").mkdir()
    return output_dir


def write_progress(run_dir: str | Path, stages: list[StageResult]) -> Path:
    """Atomically update the small progress index consumed by later visualizations."""
    output_path = Path(run_dir) / "progress.json"
    temporary_path = output_path.with_suffix(".tmp")
    payload: dict[str, Any] = {
        "format": "gamesim.incremental-training/v1",
        "stages": [asdict(stage) for stage in stages],
    }
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def run_smoke_experiment(
    *,
    run_dir: str | Path,
    seed: int = 0,
    evaluation_games: int = DEFAULT_EVALUATION_GAMES,
    random_seed: int = 123,
    device: str = "cpu",
) -> list[StageResult]:
    """Run a baseline plus three short, cumulative PPO training segments.

    Requires the ``rl`` extra. All model imports stay local so planning and tests
    remain usable without torch or sb3-contrib installed.
    """
    if evaluation_games < 1:
        raise ValueError("evaluation_games must be at least 1")

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    from gamesim.rl.selfplay_env import SelfPlayEnv
    from gamesim.rl.train import MaskablePPOSnapshotOpponent

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
    stages: list[StageResult] = []

    _record_stage(
        model=model,
        encoder=encoder,
        output_dir=output_dir,
        stages=stages,
        label="baseline",
        cumulative_timesteps=0,
        segment_timesteps=0,
        evaluation_games=evaluation_games,
        evaluation_seed=seed,
        random_seed=random_seed,
    )

    cumulative_timesteps = 0
    for segment_timesteps in SMOKE_TRAINING_SEGMENTS:
        print(f"Training next {segment_timesteps:,} timesteps...")
        model.learn(total_timesteps=segment_timesteps, reset_num_timesteps=False)
        cumulative_timesteps += segment_timesteps
        stage = _record_stage(
            model=model,
            encoder=encoder,
            output_dir=output_dir,
            stages=stages,
            label=f"step-{cumulative_timesteps:07d}",
            cumulative_timesteps=cumulative_timesteps,
            segment_timesteps=segment_timesteps,
            evaluation_games=evaluation_games,
            evaluation_seed=seed,
            random_seed=random_seed,
        )
        base_env.set_opponent(MaskablePPOSnapshotOpponent.load(output_dir / stage.checkpoint))

    return stages


def _record_stage(
    *,
    model: Any,
    encoder: ConnectFourEncoder,
    output_dir: Path,
    stages: list[StageResult],
    label: str,
    cumulative_timesteps: int,
    segment_timesteps: int,
    evaluation_games: int,
    evaluation_seed: int,
    random_seed: int,
) -> StageResult:
    """Checkpoint the live model, evaluate it, and persist one durable stage record."""
    from gamesim.rl.train import MaskablePolicyAgent

    checkpoint = Path("checkpoints") / f"{label}.zip"
    match_log = Path("matches") / f"{label}-vs-random.zip"
    checkpoint_path = output_dir / checkpoint
    model.save(checkpoint_path)
    policy_agent: MaskablePolicyAgent[ConnectFourObservation] = MaskablePolicyAgent(model, encoder)
    recorded_match = record_match(
        policy_agent,
        RandomAgent[ConnectFourObservation](seed=random_seed),
        agent_a_name="trained",
        agent_b_name="random",
        num_games=evaluation_games,
        seed=evaluation_seed,
    )
    write_match_log(output_dir / match_log, recorded_match)
    stage = StageResult(
        label=label,
        cumulative_timesteps=cumulative_timesteps,
        segment_timesteps=segment_timesteps,
        checkpoint=str(checkpoint),
        match_log=str(match_log),
        wins=sum(game.outcome == "agent_a" for game in recorded_match.games),
        losses=sum(game.outcome == "agent_b" for game in recorded_match.games),
        draws=sum(game.outcome == "draw" for game in recorded_match.games),
    )
    stages.append(stage)
    write_progress(output_dir, stages)
    print(f"{label}: {stage.wins}-{stage.losses}-{stage.draws} over {evaluation_games:,} games")
    return stage
