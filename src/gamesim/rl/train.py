"""Self-play training entrypoint for Connect Four (TORCH-DEPENDENT).

Trains ``sb3_contrib.MaskablePPO`` against itself on top of the torch-free
``SelfPlayEnv`` (``gamesim.rl.selfplay_env``). This module is the *only* place in
``gamesim.rl`` that needs torch/sb3-contrib installed -- nothing else in the package
imports it (in particular, ``gamesim/rl/__init__.py`` does not), so importing
``gamesim.rl`` or any of its other submodules never requires the DRL stack.

Every sb3/sb3-contrib import in this module is **local** (inside the function that
needs it), and the one type-only import is guarded by ``TYPE_CHECKING`` -- so even
*importing this module itself* doesn't require torch to be installed; only
*calling* ``train()`` (or the classmethods that load a saved model) does. This is
what lets ``tests/rl/test_train_smoke.py`` guard its whole module with
``pytest.importorskip("sb3_contrib")`` and skip cleanly rather than error when the
DRL stack is absent (see plans/phase-02-drl-selfplay.md, "Sandbox vs. local").

Self-play snapshot mechanism (kept intentionally simple)
----------------------------------------------------------
``MaskablePPO`` is single-agent, so ``SelfPlayEnv`` supplies the opponent's moves via
a plain ``OpponentPolicy`` callable (see ``selfplay_env.py``). Training starts that
opponent out as uniform-random (``SelfPlayEnv``'s default). ``SelfPlaySnapshotCallback``
(defined inside ``train()``, since its base class is sb3's ``BaseCallback``) then
periodically -- every ``refresh_every`` timesteps -- **freezes a copy of the
in-training policy** and swaps it in as the new opponent:

1. Save the live model to a snapshot checkpoint on disk (cheap -- it's a small
   network).
2. Reload that checkpoint into a *brand-new* ``MaskablePPO`` instance
   (``device="cpu"``), wrapped as a ``MaskablePPOSnapshotOpponent``.
3. Push it into the training env with ``SelfPlayEnv.set_opponent`` via
   ``VecEnv.env_method`` (works whether sb3 wrapped the env in a ``DummyVecEnv`` or
   handed it back unwrapped).

Reloading from disk (rather than an in-memory ``deepcopy`` of the policy) costs a
little I/O per refresh but is trivially correct: the opponent is a genuinely
separate model instance that the learner's continuing gradient steps can never
mutate out from under it. Simpler than a rolling opponent *pool* (out of scope for
this slice -- see plans/phase-02-drl-selfplay.md).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic

import numpy as np
import numpy.typing as npt

from gamesim.core.types import ActionMask, Observation
from gamesim.games.connect_four import ConnectFourEncoder, ConnectFourEngine
from gamesim.games.connect_four.engine import ConnectFourObservation
from gamesim.games.connect_four.state import NUM_COLUMNS

from .encoder import Encoder
from .selfplay_env import SelfPlayEnv

if TYPE_CHECKING:
    # Type-checking only -- never executed. mypy resolves this via the
    # [[tool.mypy.overrides]] entry for "sb3_contrib.*" (ignore_missing_imports),
    # so this line does not require sb3-contrib to be installed for `make
    # typecheck` to pass; it only supplies a name for the annotations below.
    from sb3_contrib import MaskablePPO

DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_CHECKPOINT_NAME = "connect_four_maskable_ppo"


def _predict_masked_action(
    model: MaskablePPO, encoded_observation: npt.NDArray[np.float32], mask: ActionMask
) -> int:
    """Shared prediction helper for both adapter classes below."""
    action, _state = model.predict(encoded_observation, action_masks=mask, deterministic=True)
    return int(action)


class MaskablePPOSnapshotOpponent:
    """Adapts a frozen ``MaskablePPO`` model to ``SelfPlayEnv``'s ``OpponentPolicy``.

    Callable as ``opponent(encoded_observation, mask) -> action``, matching
    ``gamesim.rl.selfplay_env.OpponentPolicy`` exactly, so an instance of this class
    can be assigned straight to ``SelfPlayEnv.opponent`` / passed to
    ``SelfPlayEnv.set_opponent``. "Frozen" means this wraps a *separate* model
    instance (typically reloaded from a checkpoint -- see ``.load()``) rather than
    the live model currently being optimized, so using it as an opponent never
    mutates, and is never mutated by, ongoing training.
    """

    def __init__(self, model: MaskablePPO) -> None:
        self._model = model

    def __call__(self, observation: npt.NDArray[np.float32], mask: ActionMask) -> int:
        return _predict_masked_action(self._model, observation, mask)

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> MaskablePPOSnapshotOpponent:
        """Load a MaskablePPO checkpoint from disk as a frozen opponent policy."""
        from sb3_contrib import MaskablePPO  # local import: torch-dependent

        model = MaskablePPO.load(str(path), device=device)
        return cls(model)


class MaskablePolicyAgent(Generic[Observation]):
    """Adapts a ``MaskablePPO`` model to the ``core.agent.Agent`` protocol.

    Bridges the observation-encoding boundary (a raw engine ``Observation`` -> the
    encoder's tensor) so a trained policy can be dropped straight into the same
    machinery as ``MinimaxAgent``/``RandomAgent`` -- in particular
    ``gamesim.rl.evaluate.evaluate`` -- for baseline comparisons.
    """

    def __init__(self, model: MaskablePPO, encoder: Encoder[Observation]) -> None:
        self._model = model
        self._encoder = encoder

    def act(self, observation: Observation, mask: ActionMask) -> int:
        encoded = self._encoder.encode(observation)
        return _predict_masked_action(self._model, encoded, mask)

    @classmethod
    def load(
        cls, path: str | Path, encoder: Encoder[Observation], *, device: str = "cpu"
    ) -> MaskablePolicyAgent[Observation]:
        """Load a MaskablePPO checkpoint from disk as an evaluatable ``Agent``."""
        from sb3_contrib import MaskablePPO  # local import: torch-dependent

        model = MaskablePPO.load(str(path), device=device)
        return cls(model, encoder)


def train(
    *,
    total_timesteps: int,
    seed: int = 0,
    refresh_every: int = 10_000,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
    device: str = "cpu",
    verbose: int = 1,
) -> Path:
    """Train a Connect Four ``MaskablePPO`` agent via self-play; returns the saved
    checkpoint path (``<checkpoint_dir>/<checkpoint_name>.zip``).

    Reproducible from ``seed`` (seeds the env's seat-randomization/default-opponent
    RNG and the model itself); requires the ``rl`` extras to be installed
    (``make install-rl``) -- see this module's docstring for the self-play snapshot
    mechanism and ``plans/phase-02-drl-selfplay.md`` for the "sandbox vs. local"
    rationale (this is expected to be run locally, not in the dev sandbox).
    """
    # Local, torch-dependent imports -- see the module docstring for why these must
    # not move to module scope.
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.callbacks import BaseCallback

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = checkpoint_dir / f"{checkpoint_name}_selfplay_snapshot.zip"

    def make_env() -> Any:
        base_env: SelfPlayEnv[ConnectFourObservation] = SelfPlayEnv(
            ConnectFourEngine(),
            ConnectFourEncoder(),
            num_actions=NUM_COLUMNS,
            seed=seed,
        )
        # sb3-contrib's convention: MaskablePPO looks for an `action_masks()`
        # method, either directly on the env or via ActionMasker's `mask_fn`.
        # SelfPlayEnv already exposes `action_masks()` itself; ActionMasker is
        # still the documented/most-compatible way to surface it through sb3's
        # VecEnv machinery, so we wrap with it explicitly.
        return ActionMasker(base_env, lambda env: env.action_masks())

    env = make_env()

    class SelfPlaySnapshotCallback(BaseCallback):  # type: ignore[misc]
        """See this module's docstring ("Self-play snapshot mechanism")."""

        def __init__(self, refresh_every: int, snapshot_path: Path, verbose: int = 0) -> None:
            super().__init__(verbose)
            self._refresh_every = refresh_every
            self._snapshot_path = snapshot_path
            self._last_refresh = 0

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last_refresh >= self._refresh_every:
                self._refresh_opponent()
                self._last_refresh = self.num_timesteps
            return True

        def _refresh_opponent(self) -> None:
            self.model.save(self._snapshot_path)
            opponent = MaskablePPOSnapshotOpponent.load(self._snapshot_path, device="cpu")
            self.training_env.env_method("set_opponent", opponent)

    model = MaskablePPO("MlpPolicy", env, seed=seed, device=device, verbose=verbose)
    callback = SelfPlaySnapshotCallback(refresh_every=refresh_every, snapshot_path=snapshot_path)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    checkpoint_path = checkpoint_dir / f"{checkpoint_name}.zip"
    model.save(checkpoint_path)
    return checkpoint_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Connect Four MaskablePPO agent via self-play. Requires the "
            "'rl' extras (`make install-rl`); not expected to run in the dev sandbox."
        )
    )
    parser.add_argument(
        "--timesteps", type=int, default=100_000, help="Total training timesteps."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (env + model).")
    parser.add_argument(
        "--refresh-every",
        type=int,
        default=10_000,
        help="Timesteps between self-play opponent snapshot refreshes.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR), help="Output directory."
    )
    parser.add_argument(
        "--checkpoint-name", type=str, default=DEFAULT_CHECKPOINT_NAME, help="Checkpoint file stem."
    )
    parser.add_argument("--device", type=str, default="cpu", help="'cpu', 'cuda', or 'auto'.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    checkpoint_path = train(
        total_timesteps=args.timesteps,
        seed=args.seed,
        refresh_every=args.refresh_every,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "MaskablePPOSnapshotOpponent",
    "MaskablePolicyAgent",
    "main",
    "train",
]
