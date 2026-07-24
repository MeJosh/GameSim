# 0008 - Recorded matches are generated outside the web client

**Status:** Accepted - 2026-07-23

## Context

We need to assess a trained policy over many games and inspect particular outcomes
visually. Running batches inside the web process would entangle training dependencies,
long-running compute, and browser state with a tool that should remain read-only.

## Decision

Add a CLI that records a versioned ZIP match archive. Its `manifest.json` indexes
logical agent labels and game summaries, while each game has its own JSON record with
seed/seats/actions/outcome. The web client uploads that artifact to a local API, which
validates each game by replaying it through the Connect Four engine. The browser
requests a specific game and move; it never derives board state itself.

## Consequences

- **+** Batch generation is scriptable, reproducible, and independent of the web UI.
- **+** The explorer remains a thin visual consumer of engine-adjudicated state.
- **+** Logs are portable artifacts that can later feed reports or regression tests.
- **-** Match logs currently target Connect Four and are held in memory after upload.
