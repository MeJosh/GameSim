# Roadmap

The framework is built in phases, each ending in something demonstrable. Connect Four
is the proving ground; **simplified Magic: The Gathering** is the long-term target that
keeps the design honest. Every phase validates one hard capability MTG will need.

Legend: ☐ not started · ◐ in progress · ☑ done

## Phase 0 — Planning & scaffold  ◐
Docs, ADRs, roadmap, project skeleton, and the core interface stubs. TDD harness runs.
**Exit:** `pytest` runs green on a scaffold; interfaces agreed. → detailed plan not needed.

## Phase 1 — Engine core + Connect Four (TDD)  ☑
Implement the `core` interfaces and a Connect Four engine, driven red → green. Random
agents play full games through the `Runner`. Deterministic logging + exact replay work.
**Validates:** authoritative rules, action masking, per-agent observation boundary,
determinism, event logging.
**Exit:** two `RandomAgent`s play a legal game to a correct terminal result; a recorded
game replays to an identical state; masking verified. → [phase-01-engine-core.md](phase-01-engine-core.md)
**Done 2026-07-23:** 34 tests green, ruff + mypy --strict clean; independently reviewed
(no blocking bugs). Built by sub agents, orchestrated.

## Phase 2 — DRL integration + self-play  ☑ (code-complete; training runs locally)
PettingZoo AEC adapter + a Connect Four encoder (state↔tensor, mask passthrough). Train
with `sb3-contrib` MaskablePPO via self-play. Evaluate vs. a random baseline and a
simple minimax. Split into **2a foundation** (encoder, adapter, minimax, eval harness —
no torch) and **2b training** (MaskablePPO self-play + smoke test). → [phase-02-drl-selfplay.md](phase-02-drl-selfplay.md)
**Validates:** the engine↔DRL boundary, masking end-to-end, self-play loop, evaluation
harness.
**Exit:** a trained agent beats random ≫50% and is competitive vs. shallow minimax;
training is reproducible from a seed.
**Done 2026-07-23:** 70 tests green, ruff + mypy --strict clean; both slices
independently reviewed (no blocking bugs). The self-play env is torch-free and fully
tested here; the actual training loop is code-complete and runs on the user's machine
(`make install-rl` + `make train`) — the sandbox can't install PyTorch (no GPU; index
blocked). **Remaining to fully close Phase 2's exit criteria:** run training locally and
confirm the trained agent beats random ≫50% / is competitive vs. minimax.

## Phase 3 — Visualization, interaction & progress measurement  ☑ (torch-free layers done; model runs local)
Interact with a trained model (play vs random/minimax/trained), log a sample of games
(e.g. model-v-model), step through them + see a summary (portable standalone HTML report
+ the browser explorer), and measure training progress across incremental checkpoints
(winrate / game length / opening strategy / head-to-head). Builds on the MVP web UI +
match-logging + incremental-experiment work. → [phase-03-visualization.md](phase-03-visualization.md)
**Validates:** the viz hook + renderer, engine-replayed analysis, the trained-model
interaction loop, and progress measurement.
**Exit:** play a game vs an opponent; record a match, open a self-contained HTML report
and step through it with a summary; (stretch) a progress report across checkpoints.
Analysis/report layers are torch-free and tested in-sandbox; model-backed runs are local.
**Done 2026-07-23:** all three deliverables (play, model-v-model logging + HTML report +
summary + explorer, incremental progress report) complete. 146 tests green + 1 skipped;
ruff + format + mypy --strict clean; built via orchestrated sub agents (implement → review
→ fix per slice). **Remaining (local, needs torch):** run a training-backed session to
exercise the trained web opponent, `record-matches` with a checkpoint, and the incremental
progress run end to end.

## Phase 4 — Prove game-agnosticism + optional ECS  ☐
Add a second simple game (candidate: Nim or Tic-Tac-Toe) reusing core/agents/logging/DRL
with only game-specific code. Introduce a lightweight ECS layer where it earns its keep.
Refactor any Connect-Four-specific assumptions that leaked into `core`.
**Validates:** the framework is genuinely game-agnostic; ECS pattern established.
**Exit:** second game trains through the same pipeline with no `core` changes.

## Phase 5 — MTG groundwork  ☐
The hard target, incrementally. Cards-as-data + effect primitives; the stack & priority
system as a data-driven turn/phase machine; hidden information (hands, shuffled
libraries) via the observation boundary; a couple of hand-built simplified decks.
**Validates:** everything above, under real pressure.
**Exit:** a simplified MTG match runs between random agents with correct rules; a first
training run is attempted. (Expected to spawn its own sub-roadmap.)

---

## Cross-cutting practices (every phase)

- **TDD red → green** for all engine rules ([ADR 0005](../docs/adr/0005-tdd-red-green.md)).
- **Progress write-ups** in [`progress/`](../progress/) at the end of each phase: what
  was built, what was learned, what surprised us, what to change.
- **Keep `core` dependency-free** — nothing in `core` imports agents, viz, or torch.
- **Determinism is non-negotiable** — all randomness through the engine RNG.

## Open questions to revisit

- When does the pure-Python engine become the bottleneck, and is a Rust core worth it then?
- Observation encoding for MTG (variable-size state) — likely needs attention/set-based
  encoders rather than fixed tensors. Defer until Phase 5.
- Self-play opponent pool / league play (vs. naive self-play) for stronger agents —
  consider in Phase 2/4.
