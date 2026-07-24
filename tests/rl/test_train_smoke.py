"""Training smoke test -- Phase 2 plan, Slice 2b test list item 3.

TORCH-DEPENDENT. This whole module must SKIP cleanly (not error) when
sb3-contrib/torch aren't installed -- see plans/phase-02-drl-selfplay.md,
"Sandbox vs. local". ``pytest.importorskip`` below is the very first thing this
module does, before importing ``gamesim.rl.train`` (which would otherwise need
sb3-contrib at import time via its local imports being exercised) so collection
never errors in the sandbox: it skips.

This is deliberately a **weak wiring check**, not a strength benchmark: a handful of
timesteps trains for correctness of the pipeline (env -> MaskablePPO -> checkpoint
-> reload -> play), not a competent policy. Run explicitly with
``pytest -m slow`` (or ``make test-slow``); it is NOT part of the default
``pytest -q`` run (see ``pyproject.toml``'s ``slow`` marker).
"""

from __future__ import annotations

import pytest

pytest.importorskip("sb3_contrib")

from pathlib import Path  # noqa: E402

from gamesim.core.agent import RandomAgent  # noqa: E402
from gamesim.games.connect_four import ConnectFourEncoder, ConnectFourEngine  # noqa: E402
from gamesim.games.connect_four.engine import ConnectFourObservation  # noqa: E402
from gamesim.rl.evaluate import evaluate  # noqa: E402
from gamesim.rl.train import MaskablePolicyAgent, train  # noqa: E402

pytestmark = pytest.mark.slow


def test_short_selfplay_run_saves_reloads_and_beats_random(tmp_path: Path) -> None:
    checkpoint_path = train(
        total_timesteps=512,
        seed=0,
        refresh_every=256,
        checkpoint_dir=tmp_path,
        checkpoint_name="smoke",
        device="cpu",
        verbose=0,
    )
    assert checkpoint_path.exists()

    encoder = ConnectFourEncoder()
    policy_agent: MaskablePolicyAgent[ConnectFourObservation] = MaskablePolicyAgent.load(
        checkpoint_path, encoder, device="cpu"
    )

    engine = ConnectFourEngine()
    random_agent: RandomAgent[ConnectFourObservation] = RandomAgent(seed=123)

    # Weak, fast wiring bar (not a strength benchmark -- see module docstring):
    # a policy trained for 512 timesteps only needs to clear >50% against random
    # over a modest number of games, which also incidentally proves it never
    # attempted an illegal move (evaluate()/run_game would raise via the engine's
    # own validation if it had).
    result = evaluate(engine, policy_agent, random_agent, num_games=20, seed=7)
    assert result.win_rate_a > 0.5
