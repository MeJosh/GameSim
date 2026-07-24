# 2026-07-23 — Phase 2b: self-play training (MaskablePPO)

## What we did
Built the self-play training layer via orchestrated sub agents (implement → review →
fix), against [the Phase 2 plan](../plans/phase-02-drl-selfplay.md) Slice 2b.

Delivered a torch-free single-agent **`SelfPlayEnv`** (Gymnasium) that presents the
2-player engine from the learner's seat with an injected opponent callable, and a
torch-dependent **`train.py`** that runs `sb3-contrib` MaskablePPO self-play with a
frozen-snapshot opponent refreshed periodically. Plus `make train` / `make test-slow`.

## The torch constraint (and how we worked around it)
PyTorch can't be installed in the dev sandbox (the CPU wheel index is proxy-blocked and
the PyPI wheel is too large; no GPU). So the actual training can't run here. Key design
move: keep the **environment torch-free** so its wiring is fully testable in-sandbox, and
isolate all torch/sb3 imports inside `train.py`'s functions. The training smoke test is
`@pytest.mark.slow` + `importorskip("sb3_contrib")`, so it skips cleanly here and runs
locally after `make install-rl`.

## Outcome (torch-free parts independently verified)
- **70 tests green + 1 skipped** (the training smoke test); ruff + mypy --strict clean.
- `import gamesim.rl.train` pulls in neither torch nor sb3 (isolation confirmed).

## Review & fix
Reviewer verdict: **approve-with-nits, no blocking bugs.** It independently verified the
single most important thing — the reward sign from the learner's perspective across **all
six seat/outcome combinations** — plus the snapshot opponent's isolation (save/reload, no
aliasing), the mask wiring through `ActionMasker`/`MaskablePPO`, and seed threading. Nits
fixed: the reward test now pins both learner seats for every outcome; a dead guard removed.

## To fully close Phase 2 (local, on your machine)
```
make install-rl
make train                    # TIMESTEPS=100000 SEED=0 by default
make test-slow                # runs the training smoke test
```
Then confirm the exit criteria: trained agent beats `RandomAgent` ≫50% and is competitive
vs. `MinimaxAgent` (use `gamesim.rl.evaluate.evaluate`). Checkpoints land in
`checkpoints/` (gitignored).

## Learnings / surprises
- Framing the 2-player game as a single-agent env with an *injected opponent callable*
  was the unlock: it made the whole self-play loop testable without torch, so the only
  unverified-in-sandbox piece is the PPO optimizer itself.
- The reward-across-seats logic needed no seat-specific code — because the engine keys
  rewards by `AgentId`, the learner's perspective falls out of a dict lookup. Worth
  pinning both seats in tests anyway (now done), since it's the highest-risk spot.

## Next
Phase 3 — visualization (live + replay), pure Python and fully verifiable in-sandbox.
