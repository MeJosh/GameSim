# 0006 — Deterministic, event-sourced logging

**Status:** Accepted — 2026-07-23

## Context

The author needs to be able to turn logging on/off, debug games, and assess training
progress by replaying specific games — without paying logging cost during bulk training.

## Decision

Make the engine **deterministic** (a single seeded RNG owns all randomness) and
**event-sourced**: the engine emits an ordered stream of immutable events. A `Recorder`
consumes them — `NullRecorder` (default, zero overhead) or `JsonlRecorder` (appends to
`.jsonl`). A recorded log is `{seed, agents, ordered actions/events}`; replaying it
through the engine reproduces the game **exactly**.

## Consequences

- **+** Reproducible experiments and a debugging "time machine" for free.
- **+** Replay is the data source for visualization (see architecture §3).
- **+** Logging is opt-in and cheap when off.
- **−** Every source of randomness *must* go through the engine RNG, and events must
  capture enough to reconstruct state. A discipline the engine contract enforces.
