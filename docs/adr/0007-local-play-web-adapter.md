# 0007 - Local play as a detachable web adapter

**Status:** Accepted - 2026-07-23

## Context

We need a quick way to play Connect Four against both a random baseline and trained
policies. That user-facing loop is useful for assessing training, but browser and
HTTP concerns must not dilute the engine-first focus of GameSim.

## Decision

Implement the first client as an optional ``gamesim.web`` package. Its in-memory
service owns short-lived sessions and translates HTTP requests into the existing
``ConnectFourEngine`` and ``Agent`` interfaces. The browser only displays returned
state and submits a column. Every human and opponent move is validated and applied
by the engine; the client contains no rule logic. The trained policy is loaded only
when selected, using the existing ``MaskablePolicyAgent`` adapter.

## Consequences

- **+** The engine remains the sole rules authority and can still run headlessly.
- **+** Random and trained opponents are selected through the existing agent boundary.
- **+** The UI can be removed, replaced, or later moved to another process without
  changing training or game rules.
- **-** Sessions are intentionally local and in-memory: restarting the server loses
  active games. Persistence, replay browsing, and remote multi-user play are later
  concerns.
