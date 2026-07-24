# 2026-07-23 — Phase 3: visualization, interaction & progress measurement

## What we did
Turned the MVP visualization work (browser play UI, recorded-match explorer, match
logging, incremental experiment) into a coherent, tested, documented layer delivering the
three outcomes the user asked for. Built via orchestrated sub agents (implement → review →
fix per slice), on top of a stabilized WIP baseline (commit `fb7e575`).

## Deliverables (all done)
1. **Interact with a trained model** — play Connect Four in the browser vs `random`,
   `minimax`, or a `trained` policy; every move engine-adjudicated. (Slice 3c)
2. **Log games + step through + summary** — `record-matches` plays any two agents
   (`random` / `minimax[:depth]` / `trained:<ckpt>`) into a versioned ZIP; a **single
   self-contained HTML report** (`gamesim.viz.report`) shows a summary and steps through
   every game with no server; the browser explorer does the same and surfaces the summary.
   (Slices 3a, 3b, 3c)
3. **Measure progress** — the incremental experiment now evaluates each checkpoint vs
   random and minimax, tracks game length + opening strategy, plays **head-to-head vs
   earlier checkpoints**, saves a match log per stage, and a torch-free **progress report**
   (`gamesim.viz.progress_report`) charts the trends. (Slice 3d)

## Architecture note (ADR 0009)
All analysis/reporting is **torch-free and engine-replayed**: reports reconstruct every
board by replaying actions through `ConnectFourEngine` (never deriving state), and render
as self-contained HTML (inline CSS/JS, no CDN). Anything needing a trained model keeps its
torch imports isolated, so importing the analysis/web/report layers never imports torch.

## Outcome (verified in-sandbox)
- **146 tests pass + 1 skipped** (the torch training smoke test); ruff + format +
  mypy --strict clean; new analysis/report/web layers confirmed torch-free.
- New: `analysis/` (summary + replay), `viz/connect_four.py` (ASCII renderer),
  `viz/report.py` (match report), `viz/progress_report.py`, `experiments/progress.py`;
  generalized `rl/record_matches.py`; minimax web opponent + summary endpoint.

## Reviews & fixes (per slice)
- **3a:** approve-with-nits — fixed a regression where the generalized `record_matches`
  CLI broke the `make record-matches` target.
- **3b:** approve — reviewer executed the embedded JS under Node to confirm step-through
  indexing; added empty-log / draw / XSS-escaping tests.
- **3c:** self-verified (full-game API smoke vs minimax, torch-free; replay parity checked).
- **3d:** approve-with-nits — fixed a raw `IndexError` on malformed schema input and made
  the incremental run **persist a match log per stage** (openable in the 3b report),
  realizing the "log game simulations at each snapshot" goal.

## Learnings / surprises
- The MVP was already well-architected (agent-generic `record_match`, clean ZIP match
  logs), so Phase 3 was mostly completing + hardening, not rebuilding.
- Keeping the self-play env / analysis torch-free continues to pay off: the whole
  visualization + progress-measurement story is verifiable in the sandbox; only the
  model-backed runs need a local torch install.
- Two subtle-bug hotspots the reviews caught early: the per-game **seat/first-mover**
  attribution (in summary + head-to-head) and the report **step-through indexing** — both
  verified correct.

## To exercise fully (local, needs `make install-rl`)
Train (`make train`), then: play the **trained** web opponent (`make serve`), record a
trained match (`make record-matches`) and open `make report`, and run the incremental
experiment + `make progress-report`.

## Next
Phase 4 — prove game-agnosticism with a second game (Nim / Tic-Tac-Toe) reusing
core/agents/rl/viz with only game-specific code; introduce a lightweight ECS where it
earns its keep.
