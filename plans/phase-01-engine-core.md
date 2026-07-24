# Phase 1 — Engine core + Connect Four (detailed plan)

**Goal:** Implement the `core` interfaces and a working Connect Four engine, built
red → green, such that random agents play legal full games through the `Runner`, and any
game can be logged and replayed to an identical state.

**Status:** ✅ **Complete** — implemented, independently reviewed (approve-with-nits,
no blocking bugs), and nits addressed. 34 tests green; ruff clean; mypy --strict clean
over both `src` and `tests`. See "As-built notes" at the end for decisions and the
review outcome.

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

---

## As-built notes (implementation decisions & deviations)

Recorded after the Phase 1 implementation pass so the plan matches reality.

**Files delivered**
- `games/connect_four/`: `actions.py` (`Action = int`, column index 0..6), `state.py`
  (`ConnectFourState`, board constants), `engine.py` (`ConnectFourEngine`,
  `ConnectFourObservation`).
- `core/runner.py`: `run_game(engine, agents, *, seed=None, recorder=None)` fleshed out.
- `core/replay.py`: `GameLog` + `replay_game(engine, log, up_to=...)`.
- Tests: `tests/games/test_connect_four.py` (groups A–E, 17 tests),
  `tests/core/test_runner.py` (group F + recorder parts of G, 6 tests),
  `tests/core/test_replay.py` (group G, 3 tests). Plus the 6 pre-existing smoke tests
  → 32 total.

**Decisions worth remembering**
- **Board orientation:** row 0 is the *bottom* (where discs land); a column is full when
  its top row (`NUM_ROWS-1`) is occupied.
- **Terminal representation:** `ConnectFourState` carries an explicit `terminal: bool`
  distinct from `winner`, so a terminal *draw* (`winner=None, terminal=True`) is
  distinguishable from an in-progress game.
- **Engine is the sole event source:** `GameEnded` is emitted by the engine inside
  `step()` when the game becomes terminal; the Runner only adds the opening
  `GameStarted`. Matches architecture.md (§4) — renderers/recorders consume one stream.
- **Seed handling:** `run_game` with `seed=None` generates a seed up front and records it
  in `GameStarted`, so even "unspecified seed" games are reproducible from the log alone
  ([ADR 0006](../docs/adr/0006-deterministic-event-logging.md)).
- **`core` stays dependency-clean:** `replay.py` defines its own minimal `GameLog` with
  two constructors — `from_events` (in-memory `Event` objects) and `from_records` (plain
  dicts matching `Event.to_dict()`, e.g. JSON-decoded `.jsonl` lines) — so `core` can
  replay real recorder output without importing `recording`. The Runner types `recorder`
  as `object` and calls `.record(...)` under a scoped `# type: ignore[attr-defined]`.
- **Draw test fixture:** a genuine 42-move drawn board is hard to hand-derive, so the
  draw test uses a real column sequence produced by `RandomAgent` self-play against the
  engine (hardcoded as `_DRAW_COLUMNS`). Deterministic and reproducible.

**Follow-ups noted for later**
- The observation boundary (test 16) is currently a full-board view (correct for a
  perfect-information game). The hidden-information machinery it foreshadows is exercised
  in Phase 5 (MTG), not here.

**Review outcome & post-review changes (2026-07-23)**
An independent Sonnet sub agent reviewed the implementation: verdict
**approve-with-nits, no blocking bugs**. It positively verified all four win directions
at corners/edges, masking in every state, validation (out-of-turn, full-column,
out-of-range, post-terminal), determinism (byte-identical boards under same seed),
replay (full + truncated), the reward convention, and the `core` dependency rule. Nits
were then addressed:
- Added tests for the **draw-terminal** path (`step` raises + all-false mask at a draw)
  and for **out-of-range** column rejection (distinct from full-column). Test count
  32 → **34**.
- `current_agent()` post-terminal behavior **documented as unspecified** (callers must
  check `is_terminal()` first); no turn-advance logic changed, to preserve determinism.
- Added a **`py.typed`** marker (+ `package-data`) making the package PEP 561-compliant,
  and extended `mypy --strict` to cover `tests` as well as `src` (26 files, clean).
- Removed a redundant test assertion.
