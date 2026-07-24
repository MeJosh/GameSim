"""PettingZoo AEC adapter tests -- Phase 2 plan, Slice 2a test list items 5-7.

See plans/phase-02-drl-selfplay.md for the full spec these pin down.
"""

from __future__ import annotations

import numpy as np
from gymnasium.spaces import Discrete
from pettingzoo.test import api_test  # type: ignore[import-untyped]

from gamesim.games.connect_four import ConnectFourEngine
from gamesim.games.connect_four.encoder import ConnectFourEncoder
from gamesim.games.connect_four.engine import ConnectFourObservation
from gamesim.games.connect_four.state import NUM_COLUMNS, NUM_ROWS
from gamesim.rl.pettingzoo_env import GameSimAECEnv


def new_env() -> GameSimAECEnv[ConnectFourObservation, int]:
    return GameSimAECEnv(ConnectFourEngine(), ConnectFourEncoder(), num_actions=NUM_COLUMNS)


# --- 5. Full PettingZoo AEC API conformance -------------------------------------------


def test_env_passes_pettingzoo_api_test() -> None:
    # Kept small (num_cycles) so this stays fast; it plays several full Connect Four
    # games end to end via env.agent_iter(), sampling legal actions from the mask,
    # and asserts the full AEC contract (agent cycling, reward accumulation,
    # dead-agent stepping, space membership, etc).
    env = new_env()
    api_test(env, num_cycles=200, verbose_progress=False)


# --- 5 & 6. observe() shape / action_mask / agent cycling / terminations -------------


def test_observe_returns_the_queried_agents_own_perspective_not_the_active_agents() -> None:
    # Review finding: GameSimAECEnv.observe(agent) must return the observation FOR
    # the agent named in its argument (the PettingZoo AEC observe(agent) contract --
    # see pettingzoo.utils.env.AECEnv.observe), not whoever is currently on turn.
    env = new_env()
    env.reset(seed=0)

    # agent_0 drops a disc in column 0; it is now agent_1's turn.
    env.step(0)
    assert env.agent_selection == "agent_1"

    # Querying the *inactive* agent_0's observation must show agent_0's own disc
    # (placed at board row 0, col 0) on agent_0's "mine" plane (index 0) -- not
    # agent_1's perspective mislabeled as agent_0's.
    obs_for_agent_0 = env.observe("agent_0")["observation"]
    assert obs_for_agent_0[0, 0, 0] == 1.0, (
        "observe('agent_0') must show agent_0's own disc on its 'mine' plane, "
        "even though it's currently agent_1's turn."
    )
    assert obs_for_agent_0[1, 0, 0] == 0.0

    # The active agent's own observation is unchanged: agent_1's "mine" plane does
    # NOT show agent_0's disc.
    obs_for_agent_1 = env.observe("agent_1")["observation"]
    assert obs_for_agent_1[0, 0, 0] == 0.0
    assert obs_for_agent_1[1, 0, 0] == 1.0

    # The inactive agent has no legal actions right now; the active agent does.
    assert np.all(env.observe("agent_0")["action_mask"] == 0)
    assert np.any(env.observe("agent_1")["action_mask"] == 1)


def test_reset_and_observe_return_observation_and_mask() -> None:
    env = new_env()
    env.reset(seed=0)

    assert env.agents == ["agent_0", "agent_1"]
    assert env.agent_selection == "agent_0"
    assert env.action_space(env.agent_selection) == Discrete(NUM_COLUMNS)

    result = env.observe("agent_0")
    assert set(result.keys()) == {"observation", "action_mask"}
    assert result["observation"].shape == (3, NUM_ROWS, NUM_COLUMNS)
    assert result["observation"].dtype == np.float32
    assert result["action_mask"].shape == (NUM_COLUMNS,)
    assert result["action_mask"].dtype == np.int8
    assert np.all(result["action_mask"] == 1)  # all columns legal at start


def test_action_mask_never_permits_a_full_column() -> None:
    env = new_env()
    env.reset(seed=0)

    # Fill column 0 by having whichever agent is on turn keep dropping into it.
    for _ in range(NUM_ROWS):
        agent = env.agent_selection
        env.step(0)
        if env.terminations[agent]:
            break

    observation, _, terminated, _, _ = env.last()
    if not terminated:
        assert observation["action_mask"][0] == 0


def test_agents_cycle_and_terminations_and_rewards_over_scripted_games() -> None:
    env = new_env()
    env.reset(seed=1)

    # Horizontal win for agent_0: columns 0,1,2,3 on the bottom row, agent_1 plays
    # elsewhere (mirrors the engine-level horizontal win test).
    scripted_columns = [0, 4, 1, 4, 2, 5, 3]
    seen_agents = []
    for column in scripted_columns:
        seen_agents.append(env.agent_selection)
        env.step(column)

    # Agents alternated starting from agent_0.
    assert seen_agents == [
        "agent_0",
        "agent_1",
        "agent_0",
        "agent_1",
        "agent_0",
        "agent_1",
        "agent_0",
    ]
    assert env.terminations["agent_0"] is True
    assert env.terminations["agent_1"] is True
    assert env._cumulative_rewards["agent_0"] == 1.0
    assert env._cumulative_rewards["agent_1"] == -1.0


def test_agent_iter_drains_all_agents_after_terminal() -> None:
    env = new_env()
    env.reset(seed=2)
    scripted_columns = [0, 4, 1, 4, 2, 5, 3]
    for column in scripted_columns:
        env.step(column)

    # Both agents must be steppable (with action=None) to drain per the PettingZoo
    # dead-agent-stepping convention, after which env.agents is empty.
    for _ in range(2):
        assert env.agent_selection in ("agent_0", "agent_1")
        env.step(None)
    assert env.agents == []


# --- 7. Stepping mirrors the engine's own state transition ---------------------------


def test_step_mirrors_engine_state_transition() -> None:
    from gamesim.core.types import AgentId
    from gamesim.games.connect_four.state import PLAYER_TOKENS

    engine = ConnectFourEngine()
    engine.reset(seed=42)
    env = GameSimAECEnv(ConnectFourEngine(), ConnectFourEncoder(), num_actions=NUM_COLUMNS)
    env.reset(seed=42)

    columns = [3, 2, 4, 1, 5]
    for i, column in enumerate(columns):
        current = engine.current_agent()
        engine.step(current, column)
        env.step(column)

        # The engine's board is agent-invariant (perfect information); compare it
        # against the *acting* agent's canonical-form encoding reconstructed back
        # into raw tokens, since that's the perspective the wrapper's observe()
        # produces for whichever agent is currently on turn.
        expected_board = engine.observation(AgentId(0)).board
        next_agent_index = int(env.agent_selection.split("_")[1])
        actual_obs = env.observe(env.agent_selection)["observation"]

        reconstructed = np.zeros_like(expected_board)
        reconstructed[actual_obs[0] == 1.0] = PLAYER_TOKENS[next_agent_index]
        reconstructed[actual_obs[1] == 1.0] = PLAYER_TOKENS[1 - next_agent_index]
        assert np.array_equal(reconstructed, expected_board), f"mismatch after move {i}"
