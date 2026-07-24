# 0002 — N-agent interface from day one

**Status:** Accepted — 2026-07-23

## Context

The near-term games are two-player, but the author wants the option to move to N
players later (and MTG is commonly multiplayer). The question was whether committing to
two players now would force a rewrite later.

## Decision

Never bake the number "2" into the core. The engine addresses players by `AgentId` and
exposes `agents()`, `current_agent()`, and per-agent `observation`/`legal_actions`/
`rewards`. Turn order is a data-driven policy (`current_agent(state)`), not hardcoded
alternation. This is the PettingZoo AEC model; two-player is simply the N=2 case.

## Consequences

- **+** Extending to N players is naming/turn-policy work, not an engine rewrite.
- **+** Matches the standard multi-agent API, easing the DRL adapter.
- **−** Slightly more ceremony than a hardcoded two-player engine (indexing by id vs.
  two fields). Cheap and worth it.
- **Note:** The *learning* side of N-player general-sum games (credit assignment, team
  play, equilibrium selection) is genuinely harder. That is a training-strategy problem
  to tackle when an N-player game is actually built; it does not affect the engine
  design.
