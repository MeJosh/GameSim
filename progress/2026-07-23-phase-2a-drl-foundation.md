# 2026-07-23 — Phase 2a: DRL foundation (no torch)

## What we did
Built the pure-Python DRL foundation via orchestrated sub agents (implement → review →
fix), red → green against [the Phase 2 plan](../plans/phase-02-drl-selfplay.md) Slice 2a.

Delivered: a Connect Four **encoder** (`(3,6,7)` canonical planes), a generic
**PettingZoo AEC wrapper** (`GameSimAECEnv`) over any `Engine` + `Encoder`, a
**minimax** baseline agent (alpha-beta, `agents/scripted.py`), and an **evaluation
harness** (`rl/evaluate.py`, seat-alternating, reproducible).

## Outcome (independently verified)
- **55 tests green**, ruff clean, mypy --strict clean over src + tests.
- The full `pettingzoo.test.api_test` AEC-conformance check passes.
- Minimax beats random well above 50% (evaluation harness).

## Review & fix
Reviewer verdict: **approve-with-nits**. It independently checked the alpha-beta search
against an unpruned negamax, the seat-swap win attribution in `evaluate()` (a classic
bug spot — correct here), and PettingZoo conformance. It found one real (dormant) bug:
`observation(agent)` ignored its argument, so querying a non-active agent's view was
silently wrong.

Fixed by enriching `ConnectFourObservation`: `current_agent` → `perspective_agent` plus
a carried `legal_actions` mask. Now `observation(agent)` honors its argument, and the
encoder's `action_mask` equals `engine.legal_actions` in every state (terminal included)
by construction. Minimax docstring notes exact win/block holds for `depth >= 2`.

## Learnings / surprises
- The observation type being too thin was the shared root of two separate findings — a
  reminder that the per-agent observation boundary (a core architectural promise) needs
  to be genuinely exercised, not assumed. Fixing it now keeps Phase 5's hidden-info work
  honest.
- Keeping the AEC wrapper fully generic (engine + injected encoder, no game specifics)
  paid off: it passed PettingZoo's conformance suite and is ready to reuse in Phase 4.

## Next
Slice 2b — MaskablePPO self-play training + fast smoke test (torch/sb3-contrib). Full
convergence runs happen on the user's machine (no GPU in the sandbox).
