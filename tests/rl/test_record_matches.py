"""Tests for generalized match recording -- plan Slice 3a, test 7.

Only non-torch agent specs (``random``, ``minimax[:depth]``) are exercised end to
end; the ``trained:`` path is covered with the model loader monkeypatched so this
module never needs torch/sb3-contrib installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gamesim.agents.scripted import MinimaxAgent
from gamesim.core.agent import RandomAgent
from gamesim.core.types import ActionMask, AgentId
from gamesim.games.connect_four.engine import ConnectFourObservation
from gamesim.games.connect_four.state import NUM_COLUMNS, NUM_ROWS
from gamesim.recording.match_log import read_match_log
from gamesim.rl import record_matches


def _empty_observation() -> ConnectFourObservation:
    return ConnectFourObservation(
        board=np.zeros((NUM_ROWS, NUM_COLUMNS), dtype=np.int8),
        perspective_agent=AgentId(0),
        legal_actions=np.ones(NUM_COLUMNS, dtype=np.bool_),
    )


# --- build_agent ---------------------------------------------------------------


def test_build_agent_parses_random_spec() -> None:
    agent = record_matches.build_agent("random", seed=3)

    assert isinstance(agent, RandomAgent)
    mask: ActionMask = np.ones(NUM_COLUMNS, dtype=np.bool_)
    action = agent.act(_empty_observation(), mask)
    assert 0 <= action < NUM_COLUMNS


def test_build_agent_parses_minimax_default_depth() -> None:
    agent = record_matches.build_agent("minimax")

    assert isinstance(agent, MinimaxAgent)
    assert agent._depth == 4  # pinning the documented default


def test_build_agent_parses_minimax_with_explicit_depth() -> None:
    agent = record_matches.build_agent("minimax:3")

    assert isinstance(agent, MinimaxAgent)
    assert agent._depth == 3


def test_build_agent_rejects_unknown_spec() -> None:
    with pytest.raises(ValueError, match="unknown agent spec"):
        record_matches.build_agent("nonsense")


def test_build_agent_rejects_non_integer_minimax_depth() -> None:
    with pytest.raises(ValueError, match="invalid minimax depth: 'abc'"):
        record_matches.build_agent("minimax:abc")


def test_build_agent_trained_spec_requires_a_checkpoint_path() -> None:
    with pytest.raises(ValueError, match="checkpoint path"):
        record_matches.build_agent("trained:")


def test_build_agent_trained_spec_uses_the_monkeypatched_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_paths: list[Path] = []

    class _FakeAgent:
        def act(self, observation: ConnectFourObservation, mask: ActionMask) -> int:
            return int(mask.nonzero()[0][0])

    def fake_loader(path: Path) -> _FakeAgent:
        seen_paths.append(path)
        return _FakeAgent()

    monkeypatch.setattr(record_matches, "_load_trained_agent", fake_loader)

    agent = record_matches.build_agent("trained:checkpoints/foo.zip")

    assert isinstance(agent, _FakeAgent)
    assert seen_paths == [Path("checkpoints/foo.zip")]


# --- CLI argument parsing ---------------------------------------------------------


def test_parse_args_accepts_the_makefiles_record_matches_flags() -> None:
    """`_parse_args` must still accept exactly the argv `make record-matches` builds.

    Mirrors Makefile's `record-matches` recipe:
        $(VENV_PY) -m gamesim.rl.record_matches --agent-a $(AGENT_A) \\
            --agent-b $(AGENT_B) --games $(GAMES) --seed $(SEED) --output $(MATCH_LOG)
    with AGENT_A defaulting to trained:$(CHECKPOINT) and AGENT_B to random. Only
    argument parsing is exercised here (not `build_agent`/`main`) so this stays
    torch-free even though the spec names a `trained:` checkpoint.
    """
    args = record_matches._parse_args(
        [
            "--agent-a",
            "trained:checkpoints/connect_four_maskable_ppo.zip",
            "--agent-b",
            "random",
            "--games",
            "100",
            "--seed",
            "0",
            "--output",
            "logs/connect_four_trained_vs_random.zip",
        ]
    )

    assert args.agent_a == "trained:checkpoints/connect_four_maskable_ppo.zip"
    assert args.agent_b == "random"
    assert args.games == 100
    assert args.seed == 0
    assert str(args.output) == "logs/connect_four_trained_vs_random.zip"


# --- end-to-end CLI --------------------------------------------------------------


def test_record_matches_cli_writes_a_reproducible_log_for_minimax_v_random(
    tmp_path: Path,
) -> None:
    output = tmp_path / "minimax_v_random.zip"

    record_matches.main(
        [
            "--agent-a",
            "minimax:2",
            "--agent-b",
            "random",
            "--games",
            "3",
            "--seed",
            "11",
            "--agent-b-seed",
            "5",
            "--output",
            str(output),
        ]
    )

    log = read_match_log(output)
    assert log.agent_a == "minimax:2"
    assert log.agent_b == "random"
    assert len(log.games) == 3
    for game in log.games:
        assert game.outcome in {"agent_a", "agent_b", "draw"}
        assert len(game.actions) > 0

    # Re-running with the same arguments reproduces identical games.
    output_again = tmp_path / "minimax_v_random_again.zip"
    record_matches.main(
        [
            "--agent-a",
            "minimax:2",
            "--agent-b",
            "random",
            "--games",
            "3",
            "--seed",
            "11",
            "--agent-b-seed",
            "5",
            "--output",
            str(output_again),
        ]
    )
    log_again = read_match_log(output_again)
    assert [g.actions for g in log.games] == [g.actions for g in log_again.games]
    assert [g.outcome for g in log.games] == [g.outcome for g in log_again.games]


def test_record_matches_cli_writes_a_valid_log_for_minimax_v_minimax(tmp_path: Path) -> None:
    output = tmp_path / "minimax_v_minimax.zip"

    record_matches.main(
        [
            "--agent-a",
            "minimax:1",
            "--agent-b",
            "minimax:2",
            "--agent-a-name",
            "shallow",
            "--agent-b-name",
            "deep",
            "--games",
            "2",
            "--seed",
            "0",
            "--output",
            str(output),
        ]
    )

    log = read_match_log(output)
    assert log.agent_a == "shallow"
    assert log.agent_b == "deep"
    assert len(log.games) == 2
    outcomes = {game.outcome for game in log.games}
    assert outcomes <= {"agent_a", "agent_b", "draw"}
