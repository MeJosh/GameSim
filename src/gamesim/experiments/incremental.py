"""Bounded incremental-training experiment support (TORCH-DEPENDENT driver).

The experiment keeps one live PPO model across short learning segments. It is kept
outside the normal training CLI while the workflow is still exploratory.

Per checkpoint ("stage"), this records a richer evaluation via the torch-free
``gamesim.experiments.progress`` layer (see docs/adr/0009-offline-analysis-and-
reporting.md and plans/phase-03-visualization.md Slice 3d): winrate/game-length/
opening-move stats vs both a random and a minimax baseline (``evaluate_stage``), plus
a head-to-head record against every earlier stage (``head_to_head``). ``evaluate_stage``
also returns the two recorded ``MatchLog``s behind those stats; both drivers below write
them to ``<run_dir>/matches/`` via ``write_match_log`` and record their relative paths on
each stage's ``match_log_paths``, so a checkpoint's actual games -- not just aggregate
stats -- stay openable later (e.g. via ``gamesim.viz.report``). The resulting
``ProgressLog`` is persisted as the versioned ``progress.json`` (``PROGRESS_FORMAT``)
via ``write_progress_log`` after every stage.

Two entry points share the same per-stage evaluation/checkpoint machinery
(``_record_stage``) but pick training segment lengths differently:

- :func:`run_smoke_experiment` -- a fixed, tiny three-stage schedule
  (``SMOKE_TRAINING_SEGMENTS``) meant to verify the pipeline end-to-end fast. This is
  what ``scripts/run_incremental_smoke.py`` calls.
- :func:`run_incremental_experiment` -- runs exactly ``num_stages`` stages, each
  ``growth_factor`` times longer than the last starting from
  ``initial_segment_timesteps``. The whole schedule is fixed up front (no time budget
  or other early-stopping), so with ``show_progress=True`` (the CLI defaults to this)
  every stage -- including ones that haven't started yet -- gets its own live row: a
  real progress bar with an elapsed timer while it trains, a pulsing bar while it
  evaluates (``evaluate_stage``/``head_to_head`` have no per-game progress hook), and
  once training speed is known, an estimated duration for every stage still pending.
  See ``_StagedRunDisplay``, styled after ``gamesim.rl.train``'s training progress bar.
  This is what ``scripts/run_incremental_training.py`` calls.

Per ``plans/phase-02-drl-selfplay.md`` ("Sandbox vs. local"), neither entry point is
runnable in the dev sandbox: PyTorch can't be installed there (the CPU wheel index is
proxy-blocked and the default PyPI wheel is too large), and the sandbox also caps
individual command run time well below any real training run. Both belong on your own
machine, after ``make install-rl``.

All sb3/torch imports stay local to each ``run_*`` function, so importing this module
-- and everything it wires together from ``gamesim.experiments.progress`` -- never
requires torch to be installed. Only *calling* ``run_smoke_experiment`` or
``run_incremental_experiment`` does. Both validate their arguments before any torch
import, so bad-argument checks stay testable torch-free too. ``rich`` (used for the
optional live display) *is* a core dependency, unlike torch/sb3, so importing it at
module scope is fine.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from rich.console import Console, RenderableType
from rich.progress import (
    Progress,
    ProgressColumn,
    TaskID,
)
from rich.progress import (
    Task as RichTask,
)
from rich.progress_bar import ProgressBar
from rich.text import Text

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

# run_incremental_experiment defaults. Evaluation/head-to-head game counts are lower
# than the smoke defaults above on purpose, for two compounding reasons:
# - head-to-head cost grows with the number of stages (every new stage plays one
#   match against every earlier stage);
# - the vs-minimax leg of every stage's evaluation is the real cost driver, not
#   games-vs-random or head-to-head: MinimaxAgent's depth-4 alpha-beta search
#   (DEFAULT_MINIMAX_DEPTH) is ~0.9s/game (benchmarked -- see progress notes), *fixed
#   per stage regardless of how short that stage's training segment was*. At the
#   smoke defaults' 1,000 games that's ~15 minutes of evaluation alone, repeated at
#   every stage -- fine for the smoke script's one-off local sanity check, but it
#   would swallow most of a run's wall time here before any stage's *training* time
#   got to grow. Keeping this default modest leaves most of the time for training.
DEFAULT_INITIAL_SEGMENT_TIMESTEPS = 4_096
DEFAULT_GROWTH_FACTOR = 2.0
DEFAULT_NUM_STAGES = 6
DEFAULT_INCREMENTAL_EVALUATION_GAMES = 100
DEFAULT_INCREMENTAL_HEAD_TO_HEAD_GAMES = 100


def prepare_run_directory(run_dir: str | Path) -> Path:
    """Create an empty experiment directory without ever overwriting another run."""
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    (output_dir / "matches").mkdir()
    return output_dir


def _format_duration(seconds: float) -> str:
    """Render a non-negative duration as ``H:MM:SS`` (or ``M:SS`` under an hour)."""
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _format_magnitude(count: int) -> str:
    """Render a timestep count with a k/M/B suffix, e.g. ``10_000 -> "10k"``.

    One decimal place, trailing ``.0`` stripped -- so round numbers stay terse
    ("10k", "4M") while the doubling schedule's less-round segment lengths still
    read as one meaningful digit ("4.1k" for 4,096) instead of a false-precision
    exact count.
    """
    magnitude = abs(count)
    for threshold, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if magnitude >= threshold:
            scaled = count / threshold
            return f"{scaled:.1f}".rstrip("0").rstrip(".") + suffix
    return str(count)


# Status values a _StagedRunDisplay task can be in, and how each renders. "summary"
# is the one always-present overview row; every other status belongs to a baseline/
# stage row. Driven entirely by explicit calls (begin_training/begin_evaluating/
# finish_stage) -- never inferred from rich's own completed/total bookkeeping -- so a
# stage's bar/color/text always reflects what this module knows actually happened.
_STATUS_ICON = {
    "pending": "○",
    "training": "●",
    "evaluating": "●",
    "complete": "✓",
    "summary": "▸",
}
_STATUS_STYLE = {
    "pending": "dim",
    "training": "orange3",
    "evaluating": "orange3",
    "complete": "green3",
    "summary": "bold cyan",
}


class _StageLabelColumn(ProgressColumn):
    """A status icon plus the task's description, colored by ``task.fields["status"]``."""

    def render(self, task: RichTask) -> Text:
        status = task.fields.get("status", "pending")
        icon = _STATUS_ICON.get(status, "○")
        style = _STATUS_STYLE.get(status, "dim")
        return Text(f"{icon} {task.description}", style=style)


class _StageBarColumn(ProgressColumn):
    """A bar whose fill/color/pulse is driven by ``status``, not rich's defaults.

    "pending" rows get no bar at all (nothing to show yet -- see
    ``_StageDetailColumn`` for their estimate instead). "training" fills a real
    determinate bar as timesteps complete. "evaluating" pulses -- there is no
    per-game progress hook to fill a determinate bar with. "complete" is a solid
    finished bar. Explicit ``pulse=`` control matters here: naively driving color
    off "did this task reach its total" (rich's own default) would flip a stage's
    bar to "finished" the instant its *training* timesteps hit 100%, even though
    evaluation for that stage hasn't run yet.
    """

    def __init__(self, bar_width: int | None = 28) -> None:
        super().__init__()
        self._bar_width = bar_width

    def render(self, task: RichTask) -> RenderableType:
        status = task.fields.get("status", "pending")
        if status == "pending":
            return Text("")
        is_summary = status == "summary"
        complete_style = "cyan" if is_summary else "orange3"
        finished_style = "cyan" if is_summary else "green3"
        return ProgressBar(
            total=task.total or 1,
            completed=task.completed,
            width=self._bar_width,
            pulse=(status == "evaluating"),
            complete_style=complete_style,
            finished_style=finished_style,
            pulse_style=complete_style,
        )


class _StageDetailColumn(ProgressColumn):
    """The status-dependent text after the bar: live steps/elapsed, an estimate,
    or a final duration -- all read fresh from ``task.elapsed``/``task.fields`` on
    every render, so (unlike a value that's only updated when something calls
    ``Progress.update``) this is always current, even mid-way through a long,
    hook-free evaluation phase.
    """

    def render(self, task: RichTask) -> Text:
        status = task.fields.get("status", "pending")
        elapsed = task.elapsed or 0.0
        style = _STATUS_STYLE.get(status, "dim")

        if status == "summary":
            run_dir = task.fields.get("run_dir", "")
            done = int(task.completed)
            total = int(task.total or 0)
            return Text(
                f"{run_dir} · {done}/{total} stages complete · {_format_duration(elapsed)} elapsed",
                style=style,
            )
        if status == "pending":
            estimate = task.fields.get("estimate_seconds")
            if estimate is None:
                return Text("estimating…", style=style + " italic")
            return Text(f"~{_format_duration(estimate)} est.", style=style)
        if status == "complete":
            return Text(f"done in {_format_duration(elapsed)}", style=style)
        if status == "training":
            completed = int(task.completed)
            total = int(task.total or 0)
            return Text(
                f"training · {completed:,}/{total:,} steps ({task.percentage:.1f}%) "
                f"· {_format_duration(elapsed)} elapsed",
                style=style,
            )
        # "evaluating"
        phase = task.fields.get("phase", "evaluating")
        return Text(f"{phase} · {_format_duration(elapsed)} elapsed", style=style)


class _StagedRunDisplay:
    """Live Rich display for :func:`run_incremental_experiment`: one row per stage.

    Because ``num_stages`` fixes the whole schedule up front, every stage's row --
    baseline plus all ``num_stages`` training stages, whether or not they've started
    -- is created immediately, so the full plan is visible from the first frame
    (a summary row on top tracks overall stage-completion progress and total
    elapsed time). A row moves through ``pending`` -> ``training`` -> ``evaluating``
    -> ``complete`` via :meth:`begin_training`/:meth:`begin_evaluating`/
    :meth:`finish_stage`; ``_StageLabelColumn``/``_StageBarColumn``/
    ``_StageDetailColumn`` render purely off each task's ``status`` field, so color
    (dim -> orange -> green) and bar behavior (none -> filling -> pulsing -> solid)
    follow directly from that one field.

    :meth:`finish_stage` also refines two running estimates -- timesteps/second from
    completed *training* time, and per-stage evaluation overhead from completed
    *total* stage time minus its training time -- and recomputes every still-pending
    row's displayed estimate from them, so estimates get more accurate as the run
    progresses (and read "estimating…" for any stage before the first one finishes,
    since there's no data yet). The evaluation-overhead estimate is a flat
    extrapolation from the most recently completed stage; it under-counts later
    stages somewhat, since head-to-head cost grows by one more opponent every stage.

    All elapsed-time text reads ``Task.elapsed`` fresh on every render (rich tracks
    that off the wall clock from each task's ``start_time``/``stop_time``), so it's
    always live without this class needing to poll or tick anything itself.
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        num_stages: int,
        segment_timesteps: Sequence[int],
        console: Console | None = None,
    ) -> None:
        self._segment_timesteps = list(segment_timesteps)
        self._completed_stages = 0
        self._steps_per_second: float | None = None
        self._eval_seconds: float | None = None

        self._progress = Progress(
            _StageLabelColumn(),
            _StageBarColumn(),
            _StageDetailColumn(),
            console=console,
            refresh_per_second=20,
        )
        self.run_task: TaskID = self._progress.add_task(
            "Run",
            total=num_stages,
            completed=0,
            status="summary",
            run_dir=str(run_dir),
        )
        self.baseline_task: TaskID = self._progress.add_task(
            "Baseline", total=1, completed=0, status="pending", start=False
        )
        self.stage_tasks: list[TaskID] = [
            self._progress.add_task(
                f"Stage {stage_number} ({_format_magnitude(steps)})",
                total=steps,
                completed=0,
                status="pending",
                start=False,
            )
            for stage_number, steps in enumerate(self._segment_timesteps, start=1)
        ]

    def _task(self, task_id: TaskID) -> RichTask:
        for task in self._progress.tasks:
            if task.id == task_id:
                return task
        raise KeyError(task_id)

    def start(self) -> None:
        self._progress.start()

    def stop(self) -> None:
        self._progress.stop()

    def begin_training(self, task_id: TaskID, total_timesteps: int) -> None:
        self._progress.start_task(task_id)
        self._progress.update(task_id, total=total_timesteps, completed=0, status="training")

    def update_training(self, task_id: TaskID, completed: int) -> None:
        self._progress.update(task_id, completed=completed)

    def begin_evaluating(self, task_id: TaskID, phase: str) -> None:
        self._progress.start_task(task_id)  # no-op if already started
        self._progress.update(task_id, status="evaluating", phase=phase)

    def finish_stage(
        self,
        task_id: TaskID,
        *,
        training_timesteps: int | None,
        training_seconds: float | None,
    ) -> None:
        task = self._task(task_id)
        self._progress.update(task_id, completed=task.total or 0, status="complete")
        self._progress.stop_task(task_id)

        self._completed_stages += 1
        self._progress.update(self.run_task, completed=self._completed_stages)

        if training_timesteps and training_seconds and training_seconds > 0:
            self._steps_per_second = training_timesteps / training_seconds
        stage_seconds = task.elapsed or 0.0
        self._eval_seconds = max(stage_seconds - (training_seconds or 0.0), 0.0)
        self._refresh_pending_estimates()

    def _refresh_pending_estimates(self) -> None:
        for task_id, timesteps in zip(self.stage_tasks, self._segment_timesteps, strict=True):
            task = self._task(task_id)
            if task.fields.get("status") != "pending":
                continue
            estimate: float | None = None
            if self._steps_per_second:
                estimate = timesteps / self._steps_per_second + (self._eval_seconds or 0.0)
            self._progress.update(task_id, estimate_seconds=estimate)

    def print(self, text: str) -> None:
        self._progress.console.print(text)


def _record_stage(
    *,
    output_dir: Path,
    model: Any,
    encoder: ConnectFourEncoder,
    label: str,
    cumulative_timesteps: int,
    random_agent: Agent[ConnectFourObservation, int],
    minimax_agent: Agent[ConnectFourObservation, int],
    evaluation_games: int,
    head_to_head_games: int,
    seed: int,
    stages: list[StageMetrics],
    head_to_head_entries: list[HeadToHeadEntry],
    earlier_agents: list[tuple[str, Agent[ConnectFourObservation, int]]],
    on_phase: Callable[[str], None] | None = None,
    printer: Callable[[str], None] | None = None,
) -> StageMetrics:
    """Checkpoint the live model, evaluate it, and persist the updated progress log.

    Shared by :func:`run_smoke_experiment` and :func:`run_incremental_experiment` so
    both drivers checkpoint/evaluate/record identically. ``model`` is a live
    ``MaskablePPO`` instance, typed ``Any`` here so this helper -- like the rest of the
    module -- never needs torch/sb3 imports at module scope. Mutates ``stages``,
    ``head_to_head_entries``, and ``earlier_agents`` in place (and also returns the new
    stage) so callers can keep a simple flat list of locals across repeated calls.

    ``on_phase``, if given, is called with a short human-readable description right
    before each blocking evaluation step (e.g. ``"evaluating vs random & minimax"``,
    ``"head-to-head vs step-0004096"``) -- used by :func:`run_incremental_experiment`
    to keep its live display's current-activity text accurate. ``printer``, if given,
    replaces the builtin ``print`` for this stage's one-line results summary (used to
    route that line through a live display's console instead, so it doesn't corrupt
    the live-rendered progress bars); defaults to ``print``.

    Also writes the representative recorded ``MatchLog`` for each baseline evaluation
    (vs random, vs minimax) to ``<output_dir>/matches/`` so the checkpoint's actual
    games -- not just aggregate stats -- can later be opened in the standalone HTML
    match report (``gamesim.viz.report``); their run-dir-relative paths are recorded on
    the persisted ``StageMetrics.match_log_paths``.
    """
    from gamesim.rl.train import MaskablePolicyAgent

    printer = printer or print

    checkpoint_path = output_dir / "checkpoints" / f"{label}.zip"
    model.save(checkpoint_path)
    agent: MaskablePolicyAgent[ConnectFourObservation] = MaskablePolicyAgent(model, encoder)

    if on_phase is not None:
        on_phase("evaluating vs random & minimax")
    stage, match_logs = evaluate_stage(
        agent,
        label=label,
        cumulative_timesteps=cumulative_timesteps,
        random_agent=random_agent,
        minimax_agent=minimax_agent,
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
        if on_phase is not None:
            on_phase(f"head-to-head vs {earlier_label}")
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
    printer(
        f"{label}: vs random {stage.vs_random.win_rate:.1%} "
        f"({stage.vs_random.wins}-{stage.vs_random.losses}-{stage.vs_random.draws}), "
        f"vs minimax {stage.vs_minimax.win_rate:.1%} "
        f"({stage.vs_minimax.wins}-{stage.vs_minimax.losses}-{stage.vs_minimax.draws}) "
        f"over {evaluation_games:,} games each"
    )
    return stage


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

    stages: list[StageMetrics] = []
    head_to_head_entries: list[HeadToHeadEntry] = []
    earlier_agents: list[tuple[str, Agent[ConnectFourObservation, int]]] = []
    random_agent = RandomAgent[ConnectFourObservation](seed=random_seed)
    minimax_agent = MinimaxAgent(depth=minimax_depth)

    def record_stage(label: str, cumulative_timesteps: int) -> StageMetrics:
        return _record_stage(
            output_dir=output_dir,
            model=model,
            encoder=encoder,
            label=label,
            cumulative_timesteps=cumulative_timesteps,
            random_agent=random_agent,
            minimax_agent=minimax_agent,
            evaluation_games=evaluation_games,
            head_to_head_games=head_to_head_games,
            seed=seed,
            stages=stages,
            head_to_head_entries=head_to_head_entries,
            earlier_agents=earlier_agents,
        )

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


def run_incremental_experiment(
    *,
    run_dir: str | Path,
    num_stages: int = DEFAULT_NUM_STAGES,
    initial_segment_timesteps: int = DEFAULT_INITIAL_SEGMENT_TIMESTEPS,
    growth_factor: float = DEFAULT_GROWTH_FACTOR,
    seed: int = 0,
    evaluation_games: int = DEFAULT_INCREMENTAL_EVALUATION_GAMES,
    head_to_head_games: int = DEFAULT_INCREMENTAL_HEAD_TO_HEAD_GAMES,
    random_seed: int = 123,
    minimax_depth: int = DEFAULT_MINIMAX_DEPTH,
    device: str = "cpu",
    show_progress: bool = False,
) -> ProgressLog:
    """Run a baseline plus exactly ``num_stages`` doubling-length PPO training stages.

    Each stage trains ``growth_factor`` times longer than the last, starting from
    ``initial_segment_timesteps`` -- so early stages are cheap sanity checkpoints and
    later stages do most of the training. The whole schedule (``num_stages`` stages of
    known length) is fixed before training starts: there is no time budget or other
    early-stopping, so how long the run actually takes depends entirely on
    ``num_stages``/``growth_factor`` and your machine's speed. Requires the ``rl``
    extra, and per ``plans/phase-02-drl-selfplay.md`` ("Sandbox vs. local") is meant to
    run locally, not in the dev sandbox. Returns the full
    :class:`~gamesim.experiments.progress.ProgressLog` for the run (also persisted as
    ``<run_dir>/progress.json`` after every stage, so an interrupted run still leaves
    every completed stage on disk).

    ``show_progress`` turns on a live Rich display (:class:`_StagedRunDisplay`): one
    row per stage -- including stages that haven't started -- with a live bar and
    elapsed timer for whichever stage is training or evaluating, a frozen green bar
    and total duration for completed stages, and (once training speed is known) an
    estimated duration for stages still pending. When ``False`` (the default --
    library callers/tests get plain, deterministic ``print`` output), behavior is
    unchanged from before this option existed.
    """
    if evaluation_games < 1:
        raise ValueError("evaluation_games must be at least 1")
    if head_to_head_games < 1:
        raise ValueError("head_to_head_games must be at least 1")
    if num_stages < 1:
        raise ValueError("num_stages must be at least 1")
    if initial_segment_timesteps < 1:
        raise ValueError("initial_segment_timesteps must be at least 1")
    if growth_factor <= 1:
        raise ValueError("growth_factor must be greater than 1")

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.callbacks import BaseCallback

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

    stages: list[StageMetrics] = []
    head_to_head_entries: list[HeadToHeadEntry] = []
    earlier_agents: list[tuple[str, Agent[ConnectFourObservation, int]]] = []
    random_agent = RandomAgent[ConnectFourObservation](seed=random_seed)
    minimax_agent = MinimaxAgent(depth=minimax_depth)

    segment_schedule = [
        int(initial_segment_timesteps * growth_factor**stage_offset)
        for stage_offset in range(num_stages)
    ]

    display = (
        _StagedRunDisplay(
            run_dir=output_dir, num_stages=num_stages, segment_timesteps=segment_schedule
        )
        if show_progress
        else None
    )

    def announce(text: str) -> None:
        (display.print if display is not None else print)(text)

    def record_stage(
        label: str,
        cumulative_timesteps: int,
        *,
        stage_number: int,
        task_id: TaskID | None,
    ) -> StageMetrics:
        def on_phase(phase_text: str) -> None:
            if display is not None and task_id is not None:
                display.begin_evaluating(task_id, phase_text)

        return _record_stage(
            output_dir=output_dir,
            model=model,
            encoder=encoder,
            label=label,
            cumulative_timesteps=cumulative_timesteps,
            random_agent=random_agent,
            minimax_agent=minimax_agent,
            evaluation_games=evaluation_games,
            head_to_head_games=head_to_head_games,
            seed=seed,
            stages=stages,
            head_to_head_entries=head_to_head_entries,
            earlier_agents=earlier_agents,
            on_phase=on_phase if display is not None else None,
            printer=display.print if display is not None else None,
        )

    class _StageProgressCallback(BaseCallback):  # type: ignore[misc]
        """Forwards each SB3 env step to the live display's per-stage bar."""

        def __init__(self, task_id: TaskID, start_timesteps: int) -> None:
            super().__init__(verbose=0)
            self._task_id = task_id
            self._start_timesteps = start_timesteps

        def _on_step(self) -> bool:
            if display is not None:
                display.update_training(self._task_id, self.num_timesteps - self._start_timesteps)
            return True

    if display is not None:
        display.start()
    try:
        baseline_task = display.baseline_task if display is not None else None
        baseline_start = time.monotonic()
        record_stage("baseline", 0, stage_number=0, task_id=baseline_task)
        baseline_elapsed = time.monotonic() - baseline_start
        if display is not None and baseline_task is not None:
            display.finish_stage(baseline_task, training_timesteps=None, training_seconds=None)
        announce(f"baseline: evaluation took {baseline_elapsed:.1f}s")

        cumulative_timesteps = 0
        for stage_index in range(1, num_stages + 1):
            segment_timesteps = segment_schedule[stage_index - 1]
            task_id = display.stage_tasks[stage_index - 1] if display is not None else None

            stage_start = time.monotonic()
            if display is not None and task_id is not None:
                display.begin_training(task_id, segment_timesteps)
            else:
                print(
                    f"Stage {stage_index}/{num_stages}: training next "
                    f"{segment_timesteps:,} timesteps..."
                )

            train_start = time.monotonic()
            callback = (
                _StageProgressCallback(task_id, model.num_timesteps)
                if display is not None and task_id is not None
                else None
            )
            model.learn(
                total_timesteps=segment_timesteps, reset_num_timesteps=False, callback=callback
            )
            training_seconds = time.monotonic() - train_start

            cumulative_timesteps += segment_timesteps
            record_stage(
                f"step-{cumulative_timesteps:07d}",
                cumulative_timesteps,
                stage_number=stage_index,
                task_id=task_id,
            )
            checkpoint_path = output_dir / "checkpoints" / f"step-{cumulative_timesteps:07d}.zip"
            base_env.set_opponent(MaskablePPOSnapshotOpponent.load(checkpoint_path))

            if display is not None and task_id is not None:
                display.finish_stage(
                    task_id, training_timesteps=segment_timesteps, training_seconds=training_seconds
                )
            stage_elapsed = time.monotonic() - stage_start
            announce(
                f"Stage {stage_index}/{num_stages} took {stage_elapsed:.1f}s total "
                f"({segment_timesteps / training_seconds:.0f} steps/s training)"
            )
    finally:
        if display is not None:
            display.stop()

    return ProgressLog(stages=tuple(stages), head_to_head=tuple(head_to_head_entries))
