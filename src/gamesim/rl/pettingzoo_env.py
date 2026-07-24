"""A PettingZoo AEC wrapper over a generic ``gamesim.core.engine.Engine``.

Presents any ``Engine`` + ``Encoder`` pair as a standard PettingZoo
Agent-Environment-Cycle (AEC) multi-agent env, so the engine can be driven by
standard DRL tooling (MaskablePPO in Slice 2b, or any other PettingZoo-compatible
library) without the engine itself ever importing anything DRL-related.

Kept deliberately game-agnostic: turn order, rewards, and terminal handling are all
delegated to the injected ``Engine`` (the single source of truth per
docs/architecture.md); board -> tensor conversion and the action mask are delegated
to the injected ``Encoder``. No Connect-Four-specific logic lives here, so a second
game (Phase 4) can reuse this wrapper by supplying its own engine and encoder.

Turn-order note: ``agent_selection`` tracks ``engine.current_agent()`` directly while
the game is live (rather than an independent round-robin cycler), so this stays
correct for any turn structure the engine implements -- not just strict alternation.
Once the engine reports terminal, this follows the standard PettingZoo "dead agent
stepping" convention (see ``pettingzoo.utils.env.AECEnv._was_dead_step``): every
agent is marked terminated at once (Connect Four ends for both players
simultaneously) and must be stepped once more with ``action=None`` to be removed
from ``self.agents``, cycling through ``possible_agents`` order to visit them all.
"""

from __future__ import annotations

from typing import Any, Generic

import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from pettingzoo import AECEnv  # type: ignore[import-untyped]

from gamesim.core.engine import Engine
from gamesim.core.types import ActionT, AgentId, Observation
from gamesim.rl.encoder import Encoder

# What observe()/last() hand back to the caller: the encoder's tensor plus the
# DRL-facing action mask (int8, per gymnasium's Discrete.sample(mask) contract).
PettingZooObservation = dict[str, npt.NDArray[Any]]


class GameSimAECEnv(AECEnv, Generic[Observation, ActionT]):  # type: ignore[misc]
    """PettingZoo AEC env wrapping an ``Engine[Observation, ActionT]`` + ``Encoder``.

    ``AECEnv`` ships without type stubs (no ``py.typed`` marker), so mypy treats it
    as ``Any`` -- inheriting from it makes this class's base dynamically typed, a
    known/accepted trade-off documented in the class docstring rather than papered
    over with broad ignores elsewhere in this module. The explicit
    ``Generic[Observation, ActionT]`` is added alongside it purely so this class's
    *own* methods (``__init__``, ``step``) share one consistent binding of those two
    type variables -- without it, each method using them would be treated as its own
    independently-generic function.
    """

    metadata: dict[str, Any] = {
        "render_modes": [],
        "name": "gamesim_aec_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        engine: Engine[Observation, ActionT],
        encoder: Encoder[Observation],
        *,
        num_actions: int,
    ) -> None:
        super().__init__()
        self._engine = engine
        self._encoder = encoder
        self._num_actions = num_actions

        # Discover the fixed agent set once up front. Engines must report `agents()`
        # consistently across resets (see core.engine.Engine); Connect Four's does
        # not even require a prior reset(), but we reset here defensively so this
        # also works for engines that do.
        self._engine.reset()
        engine_agent_ids = list(self._engine.agents())
        self._agent_names: list[str] = [self._agent_name(a) for a in engine_agent_ids]
        self._name_to_id: dict[str, AgentId] = dict(
            zip(self._agent_names, engine_agent_ids, strict=True)
        )

        self.possible_agents: list[str] = list(self._agent_names)
        sample_observation = self._engine.observation(engine_agent_ids[0])
        obs_shape = self._encoder.encode(sample_observation).shape

        self.action_spaces: dict[str, spaces.Space[Any]] = {
            name: spaces.Discrete(num_actions) for name in self.possible_agents
        }
        self.observation_spaces: dict[str, spaces.Space[Any]] = {
            name: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(num_actions,), dtype=np.int8
                    ),
                }
            )
            for name in self.possible_agents
        }

        self.agents: list[str] = []
        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self.agent_selection: str = ""

    @staticmethod
    def _agent_name(agent_id: AgentId) -> str:
        return f"agent_{int(agent_id)}"

    def observation_space(self, agent: str) -> spaces.Space[Any]:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space[Any]:
        return self.action_spaces[agent]

    def observe(self, agent: str) -> PettingZooObservation:
        agent_id = self._name_to_id[agent]
        observation = self._engine.observation(agent_id)
        mask = self._encoder.action_mask(observation).astype(np.int8)
        encoded = self._encoder.encode(observation)
        return {"observation": encoded, "action_mask": mask}

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        self._engine.reset(seed=seed)
        self.agents = list(self.possible_agents)
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {name: {} for name in self.agents}
        self.agent_selection = self._agent_name(self._engine.current_agent())

    def step(self, action: ActionT | None) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        assert action is not None, "step() called with action=None for a live agent"
        agent_id = self._name_to_id[agent]
        self._clear_rewards()
        result = self._engine.step(agent_id, action)

        if result.terminal:
            for name in self.agents:
                self.rewards[name] = float(result.rewards.get(self._name_to_id[name], 0.0))
            self.terminations = dict.fromkeys(self.agents, True)
            self._accumulate_rewards()
            self.agent_selection = self._next_possible_agent(agent)
        else:
            self._accumulate_rewards()
            self.agent_selection = self._agent_name(self._engine.current_agent())

    def _next_possible_agent(self, agent: str) -> str:
        index = self.possible_agents.index(agent)
        return self.possible_agents[(index + 1) % len(self.possible_agents)]
