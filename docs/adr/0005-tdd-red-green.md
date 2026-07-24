# 0005 — Test-driven development (red → green)

**Status:** Accepted — 2026-07-23

## Context

The project is run as an iterative experiment and the author wants good software
practices. Engine correctness is critical: agents and DRL results are only trustworthy
if the rules are enforced correctly, and subtle rule bugs are easy to introduce and
hard to spot in aggregate training metrics.

## Decision

Practice **TDD with a red → green workflow**: for each engine rule / behavior, write a
failing test first, then the minimal implementation to pass, then refactor. `pytest` is
the test runner. Determinism (seeded RNG) makes engine behavior exactly testable,
including full-game replays.

## Consequences

- **+** Rules are pinned by executable specs; regressions surface immediately.
- **+** Encourages small, well-specified interfaces (easier to test = better design).
- **+** Deterministic replay tests give cheap, powerful end-to-end coverage.
- **−** Slower to write the first version of each feature. Worth it for a correctness-
  critical simulator.
