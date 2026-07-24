# 0004 — Start DRL with sb3-contrib MaskablePPO

**Status:** Accepted — 2026-07-23

## Context

The author is learning DRL and has no strong preference on algorithm/library. The
framework needs action masking (see [0003](0003-action-masking-first-class.md)) and a
first algorithm that is robust and easy to get working without deep tuning.

## Decision

Use a **proven library with action-mask support**: `sb3-contrib`'s **MaskablePPO**
(Stable-Baselines3 family). Train two-player games via **self-play**. Keep the `Agent`
interface and the encoder/adapter layers clean so the learner can later be swapped for
CleanRL or a hand-rolled implementation once the author wants to study internals.

## Consequences

- **+** PPO with masking is the reliable workhorse for this problem class; minimal
  algorithm-fighting while everything else is built.
- **+** Well-documented, batteries-included; fast route to a first trained agent.
- **+** Clean interfaces preserve the option to go hand-rolled later (a stated learning
  goal) — effectively "library now, swap later."
- **−** Some internals are hidden behind the library; deep understanding is deferred.
  Acceptable, and revisitable.
