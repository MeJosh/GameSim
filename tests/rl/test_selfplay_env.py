"""``SelfPlayEnv`` tests -- Phase 2 plan, Slice 2b test list item 1.

TORCH-FREE: this whole module must run without sb3-contrib/torch installed (see
plans/phase-02-drl-selfplay.md, "Sandbox vs. local"). See plans/phase-02-drl-
selfplay.md for the full spec these pin down.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import SupportsFloat

import numpy as np
import pytest
from gymnasium.spaces import Discrete

from gamesim.core.types import ActionMask, AgentId
from gamesim.games.connect_four import ConnectFourEncoder, ConnectFourEngine
from gamesim.games.connect_four.engine import ConnectFourObservation
from gamesim.games.connect_four.state import NUM_COLUMNS
from gamesim.rl.selfplay_env import OpponentPolicy, SelfPlayEnv, make_random_opponent


def new_env(
    *, opponent: OpponentPolicy | None = None, seed: int | None = 0
) -> SelfPlayEnv[ConnectFourObservation]:
    return SelfPlayEnv(
        ConnectFourEngine(),
        ConnectFourEncoder(),
        num_actions=NUM_COLUMNS,
        opponent=opponent,
        seed=seed,
    )


class _SequencePlayer:
    """Replays a fixed list of columns one ply at a time.

    Used both as the opponent policy *and* by the test to choose the learner's own
    actions, so a single hardcoded move sequence (independent of which seat ends up
    being "the learner") drives an entire scripted game -- see the module-level
    comment on ``test_terminal_reward_matches_a_parallel_raw_engine_replay`` for why
    this is seat-agnostic.
    """

    def __init__(self, columns: Sequence[int]) -> None:
        self._columns = list(columns)
        self._index = 0

    def next_column(self) -> int:
        column = self._columns[self._index]
        self._index += 1
        return column

    def as_opponent(self) -> OpponentPolicy:
        def _policy(observation: np.ndarray, mask: ActionMask) -> int:
            del observation
            column = self.next_column()
            assert mask[column], f"scripted opponent move {column} was illegal"
            return column

        return _policy


# --- action_masks() correctness -------------------------------------------------------


def test_action_masks_matches_encoder_at_reset_and_after_a_move() -> None:
    env = new_env(seed=0)
    obs, info = env.reset(seed=0)

    assert obs.shape == (3, 6, 7)
    assert obs.dtype == np.float32
    mask = env.action_masks()
    assert mask.shape == (NUM_COLUMNS,)
    assert mask.dtype == np.bool_
    assert np.array_equal(mask, info["action_mask"])
    # Some columns may already be occupied if the opponent moved first (learner
    # assigned seat 1), but the mask must never be all-false on a live env.
    assert np.any(mask)

    legal = np.flatnonzero(mask)
    obs2, reward, terminated, truncated, info2 = env.step(int(legal[0]))
    assert env.action_masks().shape == (NUM_COLUMNS,)
    assert np.array_equal(env.action_masks(), info2["action_mask"])
    assert isinstance(terminated, bool)
    assert truncated is False


# --- Only legal actions are ever accepted ----------------------------------------------


def test_learner_illegal_action_raises() -> None:
    env = new_env(seed=1)
    env.reset(seed=1)

    with pytest.raises(ValueError):
        env.step(NUM_COLUMNS)  # out of range

    mask = env.action_masks()
    illegal = np.flatnonzero(~mask)
    if illegal.size:
        with pytest.raises(ValueError):
            env.step(int(illegal[0]))


def test_opponent_illegal_action_raises() -> None:
    # A misbehaving opponent that always proposes an out-of-range action. Exactly
    # one of the two players is "the opponent" every ply after the learner's first
    # move (Connect Four has only two seats), so this is triggered by a single
    # legal learner step regardless of which seat ends up being the learner.
    bad_opponent: OpponentPolicy = lambda observation, mask: NUM_COLUMNS  # noqa: E731
    env = new_env(opponent=bad_opponent, seed=2)
    obs, info = env.reset(seed=2)

    legal = np.flatnonzero(env.action_masks())
    with pytest.raises(ValueError):
        env.step(int(legal[0]))


def test_default_opponent_never_plays_a_full_column() -> None:
    # Random default opponent, run for many plies; if it ever played an illegal
    # move the env would have raised already, so simply completing many episodes
    # without raising is the assertion.
    env = new_env(seed=3)
    for episode in range(15):
        obs, info = env.reset(seed=episode)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated) and steps < 42:
            mask = env.action_masks()
            legal = np.flatnonzero(mask)
            action = int(legal[0])
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
        assert steps < 42 or terminated  # never silently hangs


# --- Opponent auto-stepping from both seats --------------------------------------------


def test_opponent_moves_first_when_learner_is_seat_one() -> None:
    # Deterministic opponent (always plays column 6) makes it easy to detect
    # whether it moved before the learner's first turn.
    found_seat_zero = False
    found_seat_one = False
    for seed in range(30):
        opponent = _SequencePlayer([6]).as_opponent()
        env = new_env(opponent=opponent, seed=seed)
        obs, info = env.reset(seed=seed)
        opponent_moved_first = bool(np.any(obs[1] == 1.0))
        if opponent_moved_first:
            found_seat_one = True
            assert obs[1, 0, 6] == 1.0  # opponent's disc landed in column 6
        else:
            found_seat_zero = True
            assert np.all(obs[0] == 0.0) and np.all(obs[1] == 0.0)  # no discs placed yet
    # Over enough seeds, both seat assignments must occur (seat randomization).
    assert found_seat_zero
    assert found_seat_one


def test_reset_seat_randomization_is_balanced_and_deterministic_under_seed() -> None:
    def seat_sequence(top_seed: int, resets: int) -> list[bool]:
        env = new_env(opponent=make_random_opponent(0), seed=top_seed)
        env.reset(seed=top_seed)
        seats = [env._learner_agent == AgentId(0)]
        for _ in range(resets - 1):
            obs, info = env.reset()  # no explicit seed: RNG continues advancing
            seats.append(env._learner_agent == AgentId(0))
        return seats

    seats_a = seat_sequence(top_seed=123, resets=40)
    seats_b = seat_sequence(top_seed=123, resets=40)
    assert seats_a == seats_b, "same top-level seed must reproduce the same seat sequence"

    # Balanced: with 40 draws, both seats must show up (heavily improbable not to).
    assert any(seats_a)
    assert not all(seats_a)


# --- Episode rewards match the engine, from the learner's perspective ------------------


def _raw_engine_rewards_for(columns: Sequence[int]) -> dict[AgentId, float]:
    """Replay ``columns`` (ply order, agent alternating starting at agent 0) on a
    bare ``ConnectFourEngine`` and return the final per-agent rewards."""
    engine = ConnectFourEngine()
    engine.reset(seed=0)
    for column in columns:
        engine.step(engine.current_agent(), column)
    assert engine.is_terminal(), "fixture sequence did not reach a terminal state"
    return dict(engine.rewards())


# Agent 0 wins with a horizontal four on the bottom row (mirrors
# tests/games/test_connect_four.py::test_horizontal_win_is_terminal_with_correct_rewards).
_AGENT_0_WINS = [0, 4, 1, 4, 2, 5, 3]

# Agent 1 wins with a horizontal four on the bottom row: agent 0 makes filler moves
# in columns 5/6 while agent 1 builds columns 0-3.
_AGENT_1_WINS = [6, 0, 6, 1, 5, 2, 5, 3]

# A verified 42-move full-board draw (see
# tests/games/test_connect_four.py::test_full_board_no_line_is_draw`).
_DRAW = [
    3, 5, 0, 4, 3, 6, 0, 0, 3, 4, 0, 6, 4, 3, 0, 2, 4, 4, 5, 5, 5,
    0, 6, 1, 3, 4, 1, 1, 3, 1, 6, 2, 5, 1, 1, 6, 2, 6, 2, 2, 5, 2,
]  # fmt: skip


# Top-level seeds probed against SelfPlayEnv.reset() (with ConnectFourEngine, which
# ignores its own seed for gameplay -- see ConnectFourState.new_game -- so seat
# assignment is the only seed-sensitive thing here) to pin down which learner seat
# each one produces. ConnectFourEngine's turn order and win/draw logic never depend
# on the seed, so these seat assignments are stable regardless of which column
# sequence or opponent is plugged in for a given test run.
_SEED_FOR_LEARNER_SEAT_0 = 2  # reset(seed=2) assigns the learner to agent 0
_SEED_FOR_LEARNER_SEAT_1 = 0  # reset(seed=0) assigns the learner to agent 1


@pytest.mark.parametrize(
    ("columns", "seed"),
    [
        (_AGENT_0_WINS, _SEED_FOR_LEARNER_SEAT_0),
        (_AGENT_0_WINS, _SEED_FOR_LEARNER_SEAT_1),
        (_AGENT_1_WINS, _SEED_FOR_LEARNER_SEAT_0),
        (_AGENT_1_WINS, _SEED_FOR_LEARNER_SEAT_1),
        (_DRAW, _SEED_FOR_LEARNER_SEAT_0),
        (_DRAW, _SEED_FOR_LEARNER_SEAT_1),
    ],
    ids=[
        "agent0_wins-learner_seat_0",
        "agent0_wins-learner_seat_1",
        "agent1_wins-learner_seat_0",
        "agent1_wins-learner_seat_1",
        "draw-learner_seat_0",
        "draw-learner_seat_1",
    ],
)
def test_terminal_reward_matches_a_parallel_raw_engine_replay(
    columns: Sequence[int], seed: int
) -> None:
    # The exact same column sequence is fed to both a bare engine (ply order,
    # agent-alternating starting at agent 0) and to a SelfPlayEnv where BOTH the
    # opponent policy and the test's own choice of the learner's action pull from
    # one shared, ply-ordered cursor over the same list. Since Connect Four strictly
    # alternates and the shared cursor only ever advances by exactly one ply at a
    # time (whether that ply's mover turns out to be "the opponent", auto-played
    # inside step()/reset(), or "the learner", chosen by the test right before
    # calling step()), the two replays place identical discs in identical order --
    # this holds regardless of which seat ``reset()`` happened to assign the
    # learner. So the env's terminal reward must equal the independently-computed
    # raw-engine reward for whichever agent id ended up being the learner. ``seed``
    # is one of the two pinned values above, so both possible learner seats are
    # exercised (with known win/loss/draw outcomes) across the full parametrization
    # instead of always landing on whichever seat ``seed=0`` happens to assign.
    expected_rewards = _raw_engine_rewards_for(columns)

    player = _SequencePlayer(columns)
    env = new_env(opponent=player.as_opponent(), seed=seed)
    obs, info = env.reset(seed=seed)

    # Guard against the pinned seed/seat mapping above silently drifting (e.g. if
    # SelfPlayEnv's RNG usage ever changes) -- without this, a drifted seed would
    # still pass (the assertion below is seat-agnostic) but would silently stop
    # covering the seat the test id claims to cover.
    expected_seat = AgentId(0) if seed == _SEED_FOR_LEARNER_SEAT_0 else AgentId(1)
    assert env._learner_agent == expected_seat

    terminated = False
    reward: SupportsFloat = 0.0
    while not terminated:
        action = player.next_column()
        assert env.action_masks()[action]
        obs, reward, terminated, truncated, info = env.step(action)

    assert float(reward) == expected_rewards[env._learner_agent]


def test_ongoing_step_reward_is_zero() -> None:
    env = new_env(seed=5)
    env.reset(seed=5)
    mask = env.action_masks()
    legal = np.flatnonzero(mask)
    # A single ply can never end Connect Four (a win needs 4 discs already placed --
    # 7 plies minimum -- and a draw needs a full 42-cell board), so this step is
    # always non-terminal and the reward is unconditionally 0.0.
    obs, reward, terminated, truncated, info = env.step(int(legal[0]))
    assert terminated is False
    assert float(reward) == 0.0


# --- Sanity on spaces / construction -----------------------------------------------


def test_observation_and_action_spaces() -> None:
    env = new_env(seed=0)
    assert env.observation_space.shape == (3, 6, 7)
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == NUM_COLUMNS


def test_rejects_engines_with_agent_count_other_than_two() -> None:
    class _ThreeAgentEngine(ConnectFourEngine):
        def agents(self) -> tuple[AgentId, AgentId, AgentId]:
            return (AgentId(0), AgentId(1), AgentId(2))

    with pytest.raises(ValueError):
        SelfPlayEnv(_ThreeAgentEngine(), ConnectFourEncoder(), num_actions=NUM_COLUMNS, seed=0)
