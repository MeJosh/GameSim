# Phase 2 — DRL integration + self-play (detailed plan)

**Goal:** Train a Connect Four agent by self-play using `sb3-contrib` MaskablePPO, and
evaluate it against baselines — without leaking tensors/DRL concerns into the engine.

Phase 2 is split into two slices so the valuable, fully-testable foundation lands first
and doesn't depend on the heavy DRL stack:

- **Slice 2a — Foundation (no torch).** Encoder, PettingZoo adapter, a scripted
  baseline (minimax), and an evaluation harness. Pure Python/NumPy, TDD red → green.
- **Slice 2b — Training (torch + sb3-contrib).** MaskablePPO self-play, a fast training
  smoke test, and evaluation vs. baselines.

## Sandbox vs. local (important)

The dev sandbox has short per-command time limits and no GPU, so a **full** self-play
training run to convergence is expected to be done on the user's machine
(`make install-rl` then run the training entrypoint). In-sandbox we verify **wiring and
correctness** via a short smoke test (a handful of training steps that must run
end-to-end and produce a usable policy), not convergence. Slice 2a is fully verifiable
in-sandbox.

---

## Slice 2a — Foundation (no torch)

### Deliverables
- `src/gamesim/games/connect_four/encoder.py` — implements the `Encoder` protocol
  (`src/gamesim/rl/encoder.py`).
  - `encode(observation) -> NDArray[float32]`: board → fixed-shape planes.
    Proposed shape `(3, 6, 7)`: plane 0 = current agent's discs, plane 1 = opponent's
    discs, plane 2 = all-ones/-zeros turn indicator (or constant); document the exact
    choice. Must be from the **acting agent's perspective** (canonical form), so the
    same network serves both seats — this is what makes self-play clean.
  - `action_mask(observation) -> ActionMask`: length-7 legal-move mask (delegates to the
    engine's masking; the encoder just exposes it in the DRL-facing shape).
- `src/gamesim/rl/pettingzoo_env.py` — a PettingZoo **AEC** wrapper over the generic
  `Engine` (parameterized by an engine + encoder), presenting a standard multi-agent env:
  `reset`, `observe(agent)` returning `{"observation": tensor, "action_mask": mask}`,
  `step(action)`, `last()`, agent iteration, `rewards`/`terminations`/`truncations`,
  `action_space` (Discrete(7)), `observation_space`. Keep it engine-agnostic where
  practical; Connect-Four specifics live in the encoder.
- `src/gamesim/agents/scripted.py` (new `agents` subpackage, or under `games/connect_four`)
  — a **minimax** agent for Connect Four (alpha-beta, small fixed depth) as an evaluation
  opponent, plus the already-existing `RandomAgent`. Implements the `Agent` protocol so it
  drops into `run_game`.
- `src/gamesim/rl/evaluate.py` — an **evaluation harness**: play N games between two
  agents (alternating who moves first), return win/loss/draw counts and win-rate, with a
  seed for reproducibility. Built on the existing `Runner`.

### Test list (2a) — red → green
1. Encoder output shape and dtype are fixed and correct for an empty board.
2. Encoder is perspective-correct: from agent 0's turn vs agent 1's turn, "my discs"
   plane tracks the acting agent (canonical form).
3. Encoder round-trip sanity: a known board encodes to the expected planes.
4. `action_mask` from the encoder equals the engine's `legal_actions` (length 7,
   aligned), including a full-column case.
5. PettingZoo env passes the API contract for a couple of scripted games (agents cycle
   correctly, terminations set, rewards match the engine's +1/−1/0).
6. `observe` returns both `observation` and `action_mask`, and the mask never permits a
   full column.
7. Stepping the env with a legal action mirrors the engine's state transition.
8. Minimax agent never plays an illegal move; it takes an immediate winning move when
   one exists; it blocks an immediate opponent win.
9. Evaluation harness: minimax beats `RandomAgent` well above 50% over N seeded games;
   results are reproducible for a fixed seed; first-move alternation is applied.
10. Evaluation harness returns coherent counts (wins + losses + draws == N).

### Notes / decisions (2a)
- Keep the engine untouched. The encoder is the *only* place that knows board→tensor.
- The PettingZoo wrapper should depend only on `core` + an injected encoder, so a second
  game reuses it in Phase 4.
- Whether `agents` is a top-level package or per-game is an open call — prefer a
  top-level `gamesim.agents` for reusable scripted/policy agents, game-specific logic
  (minimax move-gen for Connect Four) can live alongside the game and be wrapped.

---

## Slice 2b — Training (torch + sb3-contrib)

### Deliverables
- `src/gamesim/rl/selfplay.py` — the self-play training setup. MaskablePPO is
  single-agent, so present the 2-player game as a single-agent Gymnasium env from the
  learner's perspective, with the **opponent drawn from a frozen snapshot** of the
  current policy (periodically refreshed). Action masking via sb3-contrib's masking
  (`MaskablePPO` + an `action_masks` accessor sourced from the encoder/engine).
- `src/gamesim/rl/train.py` (or a `scripts/` entrypoint) — a CLI to run training with
  configurable steps/seed and save a checkpoint; reproducible from a seed.
- A **fast training smoke test** (marked `slow`/opt-in): train for a small number of
  steps and assert the pipeline runs end-to-end and the resulting policy plays only legal
  moves and beats a random agent by *some* margin over a modest number of games (a weak,
  fast bar — this checks wiring, not strength).

### Test list (2b)
1. The single-agent self-play Gym env exposes a correct `action_masks` and only accepts
   legal actions; episodes terminate with engine-consistent rewards.
2. Opponent-snapshot mechanism loads/uses a frozen policy without mutating the learner.
3. (`slow`) Short training run completes, saves a checkpoint, reloads it, and the loaded
   policy plays only legal moves and beats random > ~55% over N games (weak wiring bar).

### Definition of done (Phase 2)
- 2a: all 2a tests green; ruff + mypy --strict clean; minimax > random confirmed.
- 2b: wiring verified by the smoke test in-sandbox; a documented `make install-rl` +
  training command that the user can run locally for a full run; a short progress note on
  results/next steps.
- No DRL imports in `core` or in the engine. `rl` extras remain optional.

### Out of scope
League/opponent-pool play beyond a single snapshot, hyperparameter tuning, and any second
game (Phase 4). Reward shaping — keep the terminal-only signal.

### Open questions to resolve during Phase 2
- Exact observation plane design (2 vs 3 planes; include a legal-move plane?).
- Snapshot refresh cadence and whether to keep a small opponent pool even now.
- Whether to standardize on PettingZoo + a SuperSuit/Gym conversion or hand-roll the
  single-agent self-play wrapper (hand-rolled keeps dependencies smaller and the learning
  clearer — lean this way unless it gets unwieldy).
