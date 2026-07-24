# Phase 3 — Visualization, interaction & progress measurement (detailed plan)

**Goal:** Turn the MVP visualization work into a coherent, tested, documented layer that
delivers three things:
1. **Interact with a trained model** — play Connect Four against random / minimax /
   trained opponents.
2. **Log a sample of games (e.g. model-v-model), step through them, and see a summary** —
   both as a portable **standalone HTML report** and via the existing browser explorer.
3. **Measure training progress** — incremental checkpoints, evaluated over games, with
   winrate / game-length / opening-strategy / head-to-head trends across checkpoints.

Priority (per the user): **1 and 2 first**, 3 is the stretch goal.

## What already exists (MVP baseline — commit `fb7e575`)
- `gamesim.web` — FastAPI local play UI + a recorded-match explorer (ADR 0007, 0008).
- `gamesim.recording.match` / `match_log` — `record_match(agent_a, agent_b, ...)` (already
  agent-generic) → versioned ZIP `MatchLog` (manifest + per-game JSON), engine-replayable.
- `gamesim.rl.record_matches` — CLI (currently hardcoded trained-vs-random).
- `gamesim.experiments.incremental` — segmented PPO training with per-checkpoint
  match recording + `progress.json` (currently winrate-vs-random only).
- `gamesim.rl.evaluate` — evaluation harness + CLI.

Phase 3 hardens/completes these and adds the torch-free **analysis + reporting** layer.

## Sandbox vs. local (unchanged constraint)
torch can't be installed in the dev sandbox. So everything that needs a **trained model**
(web "trained" opponent, recording a model into a log, the incremental training loop)
is built **torch-isolated** and verified with **non-torch agents** (`RandomAgent`,
`MinimaxAgent`). The analysis, reporting, replay, and rendering layers are **torch-free
and fully tested in-sandbox**. Real model-backed runs happen locally via `make`.

## Design principle (extends ADR 0008)
All visualization/analysis is a **read-only consumer of engine-adjudicated state**. Board
states shown in any report are reconstructed by **replaying actions through
`ConnectFourEngine`**, never derived by the client/report. See ADR 0009.

---

## Slice 3a — Analysis core + renderer + generalized recording (torch-free)

**Deliverables**
- `gamesim/viz/connect_four.py` — `ConnectFourRenderer` implementing the `Renderer`
  protocol: an ASCII/text board renderer usable live (during a game) and for stepping a
  replay. Fulfills the architecture doc §3 "game-specific renderer" promise.
- `gamesim/analysis/` (new, torch-free) — `MatchSummary` computed from a `MatchLog`:
  - outcome counts (agent_a / agent_b / draw), win-rate, and a first-mover / seat
    breakdown (does moving first matter);
  - game-length stats (mean / min / max, and a plies histogram);
  - opening-move distribution (first-move column frequencies);
  - overall column-usage distribution.
- A replay helper `replay_match_game(game: MatchGameLog) -> list[board-state]` (engine
  reconstruction of every ply), shared by the report (3b) and explorer (3c). If the web
  `game_service` already has equivalent logic, factor it into this shared helper.
- Generalize `gamesim/rl/record_matches.py` to arbitrary matchups: each side selectable as
  `random`, `minimax[:depth]`, or `trained:<checkpoint>`; so model-v-model,
  minimax-v-random, minimax-v-minimax, etc. Trained loading stays torch-isolated (local
  import). Non-torch matchups are fully testable.

**Test list (3a)**
1. `ConnectFourRenderer` renders a known board to the expected text; renders an empty
   board; is a valid `Renderer`.
2. `MatchSummary` outcome counts/winrate match a hand-built `MatchLog`.
3. Summary seat/first-mover breakdown is correct.
4. Game-length stats + histogram correct on a known log.
5. Opening-move and column-usage distributions correct on a known log.
6. `replay_match_game` reconstructs the exact final board (and intermediate boards) for a
   game, matching a direct engine replay; length == moves+1.
7. `record_matches` builds a valid `MatchLog` for minimax-v-random and minimax-v-minimax
   (non-torch), reproducible under seed; trained selection path is covered by a unit test
   with the model loader mocked/monkeypatched (no torch).

---

## Slice 3b — Standalone HTML match report (torch-free)

**Deliverables**
- `gamesim/viz/report.py` — `write_match_report(match_log, path)` producing a **single
  self-contained HTML file** (inline CSS/JS, no network, no server, no torch):
  - a **summary** section rendering the `MatchSummary` (outcomes, winrate, game-length
    histogram, opening distribution — simple inline SVG/CSS bars, no CDN);
  - an **interactive step-through**: game selector + prev/next/first/last move controls
    that render the board at each ply. Per-game board states are **engine-replayed in
    Python** and embedded as JSON; the JS only displays them (rules stay in the engine).
- A CLI `python -m gamesim.viz.report --log <zip> --output <html>` and a `make report`
  target.

**Test list (3b)**
1. `write_match_report` writes a non-empty `.html` file that embeds the correct number of
   games and, for a chosen game, the correct sequence of board states (parse the embedded
   JSON and compare to an engine replay).
2. The summary section reflects the log's outcome counts / winrate.
3. The report is self-contained: no `http(s)://` asset references (assert none in output).
4. CLI writes the report for a given input log.

---

## Slice 3c — Harden interactive play + explorer

**Deliverables**
- Add **minimax** as a web opponent option (currently `random` / `trained`); keep human as
  agent 0 / first mover (existing policy).
- Surface `MatchSummary` in the explorer (show summary stats for an uploaded match log).
- Keep the browser explorer thin (engine-replayed state only; reuse the 3a replay helper).

**Test list (3c)** — via FastAPI `TestClient`, non-torch:
1. Start + play a full game vs `random` and vs `minimax` through the API; engine-consistent
   outcomes; illegal columns rejected.
2. Trained opponent path is exercised with the loader monkeypatched (no torch).
3. Upload a match log, fetch its summary, and step to a specific game/move; board matches
   an engine replay.

---

## Slice 3d — Incremental progress measurement + report (stretch)

**Deliverables**
- Extend `gamesim/experiments/incremental.py`: per checkpoint, evaluate vs **random and
  minimax** (winrate/draw), capture **game-length** and **opening/strategy** stats, and run
  **head-to-head vs earlier checkpoints** (each new checkpoint plays the previous ones).
  Persist a richer `progress.json` (versioned).
- `gamesim/viz/progress_report.py` — torch-free: from a run's `progress.json` (+ its match
  logs) produce an HTML report showing trends across checkpoints: winrate vs random/minimax
  over timesteps, game-length trend, opening-distribution shift, and a head-to-head matrix.
- The training loop is torch/local; the **metrics aggregation + report are torch-free** and
  tested against synthetic `progress.json` / hand-built logs.

**Test list (3d)**
1. Progress-metrics aggregation over hand-built stage data yields correct winrates, game
   lengths, opening distributions, and a coherent head-to-head matrix.
2. `progress_report` writes a self-contained HTML file reflecting the synthetic run
   (correct number of stages, monotonic timestep axis, embedded head-to-head values).
3. `progress.json` schema round-trips (write → read) and is versioned.

---

## Definition of done (Phase 3)
- 3a–3c: all tests green; ruff + format + mypy --strict clean; the standalone HTML report
  and the browser explorer both step through recorded games and show a summary; play works
  vs random/minimax in-sandbox and vs trained locally.
- 3d (if reached): progress metrics + report verified with synthetic data in-sandbox; the
  full training-backed run documented for local use.
- No torch imports on any always-on path; `core`/engine remain DRL-free.
- ADR 0009 recorded; roadmap + progress notes updated.

## Out of scope
Multi-user/hosted play, persistence beyond local files, non-Connect-Four games (Phase 4),
and any training that must actually run in the sandbox.

## Open questions
- How much charting to inline in the HTML reports without a CDN (keep to simple inline
  SVG/CSS bars for now — no external JS libs, to stay self-contained).
- Whether the browser explorer should also render the standalone report inline (defer).

---

## As-built notes — Slice 3a (2026-07-23) ✅
Implemented, independently reviewed, findings fixed. **110 tests pass + 1 skipped**;
ruff + format + mypy --strict clean; torch-free confirmed.
- `viz/connect_four.py` — `ConnectFourRenderer` (+ pure `format_board`/`render_board`).
  Glyphs `.`/`X`/`O`; row 0 at the bottom so it reads upright.
- `analysis/summary.py` — `MatchSummary` + `summarize_match`: outcomes/winrate, a
  first-mover breakdown computed per game from `seats[0]` vs winner (reviewer hand-verified
  the tricky mixed-seat case), game-length stats+histogram, opening + column-usage
  distributions. Distribution fields are sorted tuples (deterministic, hashable).
- `analysis/replay.py` — `replay_match_game(game) -> list[BoardGrid]` reconstructs every
  ply through `ConnectFourEngine` (length moves+1, board[0] empty); reviewer diffed all
  boards against an independent engine replay.
- `rl/record_matches.py` — generalized to `--agent-a`/`--agent-b` specs (`random`,
  `minimax[:depth]`, `trained:<path>`); `build_agent()` unit-testable; trained loading
  torch-isolated.
- **Review finding fixed:** the CLI generalization had broken the `make record-matches`
  target (old `--checkpoint` flag). Makefile now uses `AGENT_A`/`AGENT_B` vars (default
  `trained:$(CHECKPOINT)` vs `random`, reproducing prior behavior). Bad `minimax:<depth>`
  now errors clearly.
- **Deferred to 3c:** `web/game_service.py` still has its own replay logic; fold it onto
  `replay_match_game` in 3c.

## As-built notes — Slice 3b (2026-07-23) ✅
Implemented, independently reviewed (**Approve**, no blocking bugs — the reviewer ran the
embedded JS under Node against a DOM stub to confirm step-through indexing), test-hardened.
**118 tests pass + 1 skipped**; ruff + format + mypy --strict clean; self-contained
verified (0 network refs).
- `viz/report.py` — `render_match_report_html(log) -> str` and `write_match_report(log,
  path)`; CLI `python -m gamesim.viz.report --log <zip> --output <html>`; `make report`.
- Single self-contained HTML: inline CSS/JS, no CDN. Embeds two JSON `<script>` blocks —
  `#match-data` (per-game `boards` from `replay_match_game`, engine truth) and
  `#match-summary` (`summarize_match`). JS only reads the embedded data (no rules in JS):
  game selector + first/prev/next/last, board render (row 0 bottom), who-is-to-move /
  outcome at the boundaries.
- Safety: `</` → `<\/` in JSON blocks and `html.escape` on visible names (XSS guard);
  pinned by tests, plus empty-log and all-games (incl. flipped-seat + draw) board checks.
