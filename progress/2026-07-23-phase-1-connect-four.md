# 2026-07-23 — Phase 1: engine core + Connect Four

## What we did
Implemented Phase 1 via a Sonnet sub agent, red → green against
[the Phase 1 plan](../plans/phase-01-engine-core.md). Delivered a working Connect Four
engine, the game `Runner`, and deterministic replay — all satisfying the plan's 23-test
specification.

## Outcome (independently verified)
- **32 tests pass** (26 new + 6 smoke), **ruff clean**, **mypy --strict clean**.
- All 23 tests in the plan's list are covered (mapping recorded in the sub agent's
  report and reflected in the test files).
- Verified by the orchestrator by re-running `pytest`, `ruff`, and `mypy` directly, not
  just trusting the sub agent's report.

## How it was built
Orchestrated, not hand-written: a **Sonnet general-purpose sub agent** did the
implementation TDD-style; the orchestrator handled verification, documentation, and the
commit. A separate review agent follows this write-up.

## Key decisions (full detail in the plan's "As-built notes")
- Board row 0 = bottom; explicit `terminal` flag separate from `winner`.
- Engine is the sole event source (`GameEnded` emitted inside `step()`); Runner only
  adds `GameStarted`.
- `run_game` generates and records a seed when none is given, so every game replays
  exactly from its log.
- `core/replay.py` keeps `core` dependency-clean with its own `GameLog`
  (`from_events` / `from_records`) — replays real `.jsonl` logs without importing
  `recording`.

## Learnings / surprises
- Hand-deriving a full-board *draw* is error-prone; generating one via real self-play and
  hardcoding the move sequence was the reliable path — a small reminder that the engine
  itself is the best oracle for fixtures.
- The determinism + event-sourcing design paid off immediately: replay tests were cheap
  and gave strong end-to-end coverage, exactly as [ADR 0006](../docs/adr/0006-deterministic-event-logging.md)
  anticipated.

## Review (Sonnet sub agent)
Verdict: **approve-with-nits, no blocking bugs.** The reviewer independently probed all
four win directions at corners/edges, masking, validation (out-of-turn / full-column /
out-of-range / post-terminal), determinism (byte-identical boards), replay (full +
truncated), the reward convention, and the `core` dependency rule — all correct.

Nits were then addressed by a follow-up sub agent:
- Added tests for the draw-terminal path and out-of-range column rejection (32 → **34**).
- Documented `current_agent()` as unspecified post-terminal (no logic change, to keep
  determinism intact).
- Added a `py.typed` marker and extended `mypy --strict` to cover `tests` too (26 files).
- Removed a redundant assertion.

Final: **34 tests green, ruff clean, mypy --strict clean over src + tests**, all
verified directly by the orchestrator.

## Next
Phase 2 — DRL integration + self-play (PettingZoo adapter + Connect Four encoder, then
MaskablePPO self-play). See [roadmap](../plans/roadmap.md).
