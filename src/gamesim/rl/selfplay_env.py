"""A single-agent Gymnasium env for MaskablePPO self-play (TORCH-FREE).

``sb3-contrib``'s ``MaskablePPO`` is a single-agent algorithm, so we can't hand it
the two-player ``GameSimAECEnv`` directly. ``SelfPlayEnv`` instead presents the game
from one seat -- "the learner" -- and drives the *other* seat internally via an
injected ``opponent`` policy callable, auto-playing opponent moves until it's the
learner's turn again (or the game ends). This is the standard trick for training a
two-player game with a single-agent RL algorithm.

This module deliberately imports only ``gymnasium``, ``numpy``, ``gamesim.core``, and
an ``Encoder`` -- **no torch, no sb3-contrib** -- so it is fully importable and
testable in any environment, including one where the heavy DRL stack isn't
installed. ``gamesim.rl.train`` (torch-dependent) builds on top of this.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from gamesim.core.engine import Engine
from gamesim.core.types import ActionMask, AgentId, Observation

from .encoder import Encoder

# What an opponent policy is: given the encoded observation (network-facing tensor)
# and the legal-action mask *for whoever is currently on turn*, return a legal
# action. Frozen-snapshot MaskablePPO opponents (``train.py``) and the default
# random opponent below both satisfy this signature, so ``SelfPlayEnv`` never needs
# to know which kind it has.
OpponentPolicy = Callable[[npt.NDArray[np.float32], ActionMask], int]


def make_random_opponent(seed: int | None = None) -> OpponentPolicy:
    """Build a uniform-random-over-legal-actions opponent policy.

    This is ``SelfPlayEnv``'s default opponent, and a sensible first opponent for
    training before any policy snapshot exists.
    """
    rng = np.random.default_rng(seed)

    def _policy(observation: npt.NDArray[np.float32], mask: ActionMask) -> int:
        del observation  # unused: random play doesn't look at the board
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise ValueError("make_random_opponent policy called with no legal actions")
        return int(rng.choice(legal))

    return _policy


class SelfPlayEnv(gym.Env[npt.NDArray[np.float32], int], Generic[Observation]):
    """Single-agent Gymnasium view of a 2-player ``Engine``, for MaskablePPO.

    Parameters
    ----------
    engine:
        The authoritative game engine (e.g. ``ConnectFourEngine``). Must report
        exactly two agents (self-play with more than two seats isn't supported by
        this simple wrapper).
    encoder:
        Converts engine observations to network-facing tensors and exposes the
        legal-action mask (see ``gamesim.rl.encoder.Encoder``).
    num_actions:
        Size of the (``Discrete``) action space.
    opponent:
        Callable used to choose the non-learner seat's moves. Defaults to a
        uniform-random policy seeded from ``seed``. ``train.py`` swaps this out
        (via the public ``opponent`` attribute) for a frozen policy snapshot as
        self-play progresses -- see its module docstring for the refresh mechanism.
    seed:
        Seeds both the env's own RNG (seat randomization, engine resets when no
        per-call seed is given) and the default random opponent.

    Design notes
    ------------
    - **Seat randomization**: every ``reset()`` picks which of the engine's two
      agents is "the learner" uniformly at random (via ``self.np_random``, which
      Gymnasium seeds from ``reset(seed=...)``), so the policy sees both seats
      roughly equally over training -- this is what makes it a fair self-play
      opponent of itself rather than always the first- or second-mover.
    - **Reward perspective**: rewards returned by ``step`` are always from the
      *learner's* point of view (+1 win / -1 loss / 0 draw-or-ongoing), regardless
      of which engine seat the learner occupies in a given episode.
    - **Legality**: both the learner's action (validated in ``step``) and the
      opponent's action (validated in the internal auto-play loop) are checked
      against the engine's own mask before being applied; an illegal move from
      either side raises rather than being silently corrected, so bugs in the
      opponent policy or the calling RL library surface immediately.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        engine: Engine[Observation, int],
        encoder: Encoder[Observation],
        *,
        num_actions: int,
        opponent: OpponentPolicy | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._engine = engine
        self._encoder = encoder
        self._num_actions = num_actions
        self.opponent: OpponentPolicy = (
            opponent if opponent is not None else make_random_opponent(seed)
        )

        # Discover the observation shape once up front (also validates the engine
        # reports exactly two agents, which this single-opponent wrapper requires).
        self._engine.reset(seed=seed)
        engine_agents = list(self._engine.agents())
        if len(engine_agents) != 2:
            raise ValueError(
                f"SelfPlayEnv requires a 2-agent engine, got {len(engine_agents)} agents"
            )
        self._agents: tuple[AgentId, AgentId] = (engine_agents[0], engine_agents[1])
        sample_observation = self._engine.observation(engine_agents[0])
        obs_shape = self._encoder.encode(sample_observation).shape

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(num_actions)

        self._learner_agent: AgentId = self._agents[0]
        self._current_mask: ActionMask = np.zeros(num_actions, dtype=np.bool_)

    def set_opponent(self, opponent: OpponentPolicy) -> None:
        """Swap the opponent policy in place.

        Exists (alongside the plain ``self.opponent`` attribute) as an explicit
        method so it can be invoked through ``VecEnv.env_method`` -- sb3 typically
        wraps a single env in a ``DummyVecEnv``, and ``env_method`` calls a *named
        method* on the underlying env(s), not an attribute assignment. This is how
        ``gamesim.rl.train``'s self-play snapshot callback refreshes the opponent
        mid-training without reaching into VecEnv internals.
        """
        self.opponent = opponent

    # -- sb3-contrib's ActionMasker / MaskablePPO reads this accessor directly ------

    def action_masks(self) -> npt.NDArray[np.bool_]:
        """Legal-action mask for the learner in the *current* state.

        This is the accessor ``sb3_contrib.common.wrappers.ActionMasker`` (and
        ``MaskablePPO`` when handed a bare env) looks for by convention -- see
        sb3-contrib's action-masking docs. Kept up to date by ``reset``/``step``.
        """
        return self._current_mask

    # -- Gymnasium API ---------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed, options=options)

        engine_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._engine.reset(seed=engine_seed)

        # Balanced self-play: randomize which seat the learner occupies this episode.
        seat_index = int(self.np_random.integers(0, len(self._agents)))
        self._learner_agent = self._agents[seat_index]

        self._play_opponent_until_learner_turn_or_terminal()

        encoded, mask = self._observe_learner()
        self._current_mask = mask
        return encoded, {"action_mask": mask}

    def step(
        self, action: int
    ) -> tuple[npt.NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        if not (0 <= action < self._num_actions) or not self._current_mask[action]:
            raise ValueError(
                f"illegal action for learner (agent {self._learner_agent}): {action}"
            )

        result = self._engine.step(self._learner_agent, action)
        terminated = result.terminal

        if not terminated:
            self._play_opponent_until_learner_turn_or_terminal()
            terminated = self._engine.is_terminal()

        encoded, mask = self._observe_learner()
        self._current_mask = mask

        if terminated:
            rewards = self._engine.rewards()
            reward: float = float(rewards.get(self._learner_agent, 0.0))
        else:
            reward = 0.0

        info: dict[str, Any] = {"action_mask": mask}
        return encoded, reward, terminated, False, info

    # -- internals ---------------------------------------------------------------

    def _observe_learner(self) -> tuple[npt.NDArray[np.float32], ActionMask]:
        observation = self._engine.observation(self._learner_agent)
        encoded = self._encoder.encode(observation)
        mask = self._encoder.action_mask(observation)
        return encoded, mask

    def _play_opponent_until_learner_turn_or_terminal(self) -> None:
        while not self._engine.is_terminal() and self._engine.current_agent() != (
            self._learner_agent
        ):
            self._opponent_step()

    def _opponent_step(self) -> None:
        current = self._engine.current_agent()
        observation = self._engine.observation(current)
        encoded = self._encoder.encode(observation)
        mask = self._encoder.action_mask(observation)
        action = self.opponent(encoded, mask)
        if not (0 <= action < self._num_actions) or not mask[action]:
            raise ValueError(f"opponent proposed illegal action {action} for agent {current}")
        self._engine.step(current, action)


__all__ = ["OpponentPolicy", "SelfPlayEnv", "make_random_opponent"]
