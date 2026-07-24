# 0009 — Offline analysis & reporting is torch-free and engine-replayed

**Status:** Accepted — 2026-07-23

## Context

Phase 3 needs to summarize matches, step through recorded games, and measure training
progress. These consumers must not drag in the DRL stack (torch/sb3) or re-implement game
rules, and they should be usable in environments without a GPU or even without torch
installed (including the dev sandbox and simple review workflows).

## Decision

Put all summary/replay/report logic in a **torch-free** layer (`gamesim.analysis`,
`gamesim.viz`) that consumes recorded `MatchLog` artifacts and reconstructs every board
state by **replaying actions through `ConnectFourEngine`** — never deriving state itself
(extends [ADR 0008](0008-recorded-match-explorer.md)). Reports render as **self-contained
HTML** (inline CSS/JS, no CDN, no server) so they are portable artifacts. Anything that
requires a trained model (recording a policy into a log, the incremental training loop)
keeps its torch imports **local/isolated**, so importing the analysis or web layers never
imports torch.

## Consequences

- **+** Summaries, step-through reports, and progress reports run anywhere, are fully
  testable without torch, and stay honest (engine is the sole rules authority).
- **+** Reports are portable single files — easy to share, archive, or diff.
- **+** Clean separation lets the same summary/replay code back both the HTML report and
  the browser explorer.
- **−** No rich client-side charting libraries (kept to inline SVG/CSS to stay
  self-contained); fancier dashboards would need a different delivery.
- **−** Board states are precomputed and embedded, so very large logs make larger HTML
  files; acceptable for the sample sizes this is meant for.
