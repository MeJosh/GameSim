# 0003 — Action masking is first-class

**Status:** Accepted — 2026-07-23

## Context

Board and card games have action spaces that are large and mostly illegal at any given
moment (Connect Four: only non-full columns; MTG: an enormous space of which a tiny
fraction is legal each priority window). DRL agents that must *learn* legality waste
enormous samples and train poorly.

## Decision

Make the **legal-action mask** a first-class part of the engine contract.
`legal_actions(agent)` always returns which actions are currently legal, and the DRL
adapter passes this mask straight to the learner so illegal actions are never sampled.
`step` still validates and rejects illegal actions loudly as a safety net.

## Consequences

- **+** Dramatically improves sample efficiency and stability for DRL.
- **+** Gives scripted/human agents the same legality info for free.
- **+** Essential groundwork for MTG's action space.
- **−** Every game engine must implement masking correctly; it is not optional. This is
  the right constraint to enforce.
- **Implication:** Drives the choice of a mask-aware algorithm — see
  [0004](0004-maskable-ppo-via-library.md).
