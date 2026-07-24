# Phase 1 — Engine core + Connect Four (detailed plan)

**Goal:** Implement the `core` interfaces and a working Connect Four engine, built
red → green, such that random agents play legal full games through the `Runner`, and any
game can be logged and replayed to an identical state.

**Definition of done**
- Two `RandomAgent`s play a complete, legal Connect Four game to a correct terminal
  outcome (win / draw), only ever choosing legal moves.
- `legal_actions` / masking is correct (full columns excluded; all-legal at start).
- The engine is deterministic: same seed + same actions ⇒ identical state.
- A recorded game (`JsonlRecorder`) replays through the engine to a byte-identical final
  state; `NullRecorder` adds no events.
- `pytest` green; `ruff` clean; `mypy --strict` clean on `core` + the game.

## Working method (red → green)

For each item below: **(1)** write the failing test, run it, watch it fail for the right
reason; **(2)** write the minimal code to pass; **(3)** refactor with tests green. Commit
per green step. Keep commits small and messages describing the behavior pinned.

## Test list (the specification)

Connect Four = 7 columns × 6 rows; drop a disc into a non-full column; win = 4 in a row
(horizontal, vertical, diagonal); draw = board full, no winner. Two agents alternate.

### A. Core types & construction
1. A fresh engine reports both agents via `agents()` and `current_agent()` is agent 0.
2. `is_terminal()` is `False` at start; `rewards()` is 0 for all agents at start.
3. A fresh board is empty (all cells unoccupied) in agent 0's observation.

### B. Legal actions / masking
4. At start, all 7 columns are legal (mask all-true).
5. After 6 discs in one column, that column becomes illegal (mask false there).
6. Mask length equals the action space size and aligns with column indices.

### C. Applying actions & validation
7. `step` drops a disc into the chosen column at the lowest empty row.
8. `step` advances `current_agent()` to the other agent.
9. `step` with a full column raises (illegal action rejected loudly).
10. `step` by the agent not on turn raises.

### D. Terminal conditions & rewards
11. Horizontal 4-in-a-row ⇒ terminal, mover rewarded (+1 / −1 convention).
12. Vertical 4-in-a-row ⇒ terminal, correct winner.
13. Both diagonal directions ⇒ terminal, correct winner.
14. Full board with no line ⇒ terminal, draw (rewards 0/0).
15. No legal actions once terminal; `step` after terminal raises.

### E. Observation boundary
16. Each agent's observation is well-formed and (for Connect Four) shows the full board
    consistently. (Placeholder for the hidden-info boundary that MTG will exercise.)

### F. Determinism & the Runner
17. `Runner` drives two `RandomAgent`s to a terminal state in a legal sequence.
18. Same seed ⇒ two `Runner` games produce identical action sequences and final state.
19. A `RandomAgent` never selects a masked-illegal action over many seeded games.

### G. Logging & replay
20. `NullRecorder` produces no output and does not alter play.
21. `JsonlRecorder` writes one event per emitted engine event, in order.
22. Replaying a recorded `{seed, actions}` log reconstructs an identical final state.
23. Loading a truncated log replays a valid mid-game state (supports step-through later).

## Build order (maps to the test list)

1. `core/types.py` — `AgentId`, action typing, `ActionMask`, `Observation`, `StepResult`,
   `Event` types. (Tests A, B scaffolding.)
2. `core/engine.py` — the `Engine` Protocol/ABC (already stubbed in scaffold; flesh out).
3. `games/connect_four/` — `state.py` (board), `engine.py` (rules), `actions.py`
   (`DropDisc`). Drive with tests A–E.
4. `core/agent.py` — `Agent` protocol + `RandomAgent` (seeded). Tests F.
5. `core/runner.py` — the game loop wiring engine + agents + recorder. Tests F.
6. `logging/recorder.py` — `Recorder` protocol, `NullRecorder`, `JsonlRecorder`. Tests G.
7. `core/replay.py` — reconstruct a game from a log. Tests G (22–23).

## Notes / decisions to make during Phase 1

- **Reward convention:** terminal +1 winner / −1 loser / 0 draw, zero elsewhere. Revisit
  if reward shaping is wanted in Phase 2 (prefer not to — keep the signal clean).
- **Action encoding:** integer column index `0..6`; mask is a length-7 boolean array.
  This is the encoder's contract in Phase 2.
- **Event schema:** start with `{type, agent, action, seed, ...}`; keep it minimal but
  sufficient for exact replay. Version the schema field from day one.
- **Where does turn order live?** In the engine via `current_agent(state)`, not the
  Runner — keeps N-agent generalization in one place ([ADR 0002](../docs/adr/0002-n-agent-interface.md)).

## Out of scope for Phase 1

DRL/training, encoders, the PettingZoo adapter, visualization, ECS. Those are Phases 2–4.
Keep Connect Four's engine plain (no ECS) — it's the clarity baseline.
