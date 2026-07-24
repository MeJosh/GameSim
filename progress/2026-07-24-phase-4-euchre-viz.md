# Progress — Euchre visualizer: standalone match report (2026-07-24)

**Status:** Complete. New/changed tests: 34 new (25 unit + a report validated via a
separate jsdom-based execution check, see below); 220 tests total across the repo
(216 collected + runnable in this sandbox + 1 pre-existing skip); ruff, `ruff format
--check`, and `mypy --strict` clean on `core`, `games`, `recording`, `analysis`,
`viz`. `scripts/`, `rl/`, and `web/` untouched except the one shared refactor noted
below.

## What was built

Following up on the Phase 4 Euchre engine (plans/phase-04-euchre.md), this adds the
first visualizer: a standalone, self-contained HTML match report, mirroring Connect
Four's `viz/report.py` (Phase 3) but reshaped for Euchre's 4-seat/hidden-information
structure.

- `games/euchre/cards.py`: `card_label`/`suit_symbol` display helpers (e.g. `"J♠"`).
- `recording/recorder.py`: promoted the private `_EventCollector` (previously
  duplicated) into a shared, public `EventCollector` -- used by both `match.py` and
  the new `euchre_match.py`.
- `recording/euchre_match_log.py` / `recording/euchre_match.py`: a Euchre-shaped
  `MatchLog`/`record_match` equivalent -- 4 named seats per hand, `team_a`/`team_b`
  outcome (no draw is possible in Euchre), and enough per-hand metadata (`maker_team`,
  `alone`, `points`, `trump`, `dealer`, `stick_the_dealer`) that summary stats never
  need to replay a hand.
- `analysis/replay_euchre.py`: `replay_euchre_match_game` -- engine-replayed,
  full-visibility ("god view") state per ply, with a precomputed human-readable
  action label per ply so the report's JavaScript never has to know what an action
  *means*.
- `analysis/summary_euchre.py`: `EuchreMatchSummary` -- win rates, march/lone-march/
  euchre counts, points and trump-suit distributions, computed purely from logged
  metadata (no replay).
- `viz/report_euchre.py`: the report itself -- summary bars + an interactive
  step-through with a god-view/per-seat toggle, 4 hand panels, current-trick display,
  dealer/trump/phase status line.
- `scripts/record_euchre_demo_match.py` + `make record-euchre-demo` / `make
  report-euchre`: a torch-free RandomAgent-vs-RandomAgent demo generator, since
  there's no trained Euchre policy yet -- this is the fastest path to an actual
  `EuchreMatchLog` to look at.
- `reports/euchre_match.html`: a checked-in sample report (50 demo hands), matching
  the precedent set by `reports/connect_four_match.html`.

## Decisions worth remembering

- **God view by default, per-seat toggle available** (user's explicit choice). The
  key design question this raised -- and Connect Four's report never had to answer,
  since its board has no hidden information -- is *where* the "show everything"
  view gets its data from. Resolved by: `EuchreEngine.observation(agent)` keeps
  hiding other hands (that boundary is for **live agents during play** and must stay
  intact); the report's replay module instead reaches directly into
  `EuchreEngine`'s internal `_state` for **already-completed, already-recorded**
  hands, since there's no live agent to leak information to at that point. The
  per-seat toggle in the browser is then just hiding already-fully-known columns of
  data, not a real information boundary -- documented explicitly in
  `replay_euchre.py`'s module docstring so this doesn't read as an accidental
  boundary violation later.
- **One `EuchreMatchLog` "team" can occupy either seat parity across hands.**
  Partnerships are fixed by seat (0&2 vs 1&3), and the dealer's left-hand seat bids
  first, so always putting `team_a` at seats 0&2 would confound "team_a" with a
  seating advantage. `record_euchre_match` alternates which parity `team_a` occupies
  every other hand (same idea as `record_match` alternating who moves first in
  Connect Four); the dealer seat itself stays fixed at 0 (only which *team* sits
  there varies).
- **`record_euchre_match` can't reuse `core.runner.run_game`** as-is, because
  `run_game` only forwards `seed` to `Engine.reset`, and `EuchreEngine.reset` also
  needs `dealer`/`rules`. Added a small local `_play_one_hand` loop instead of
  changing `run_game`'s signature (`core` stays generic on purpose -- see ADR 0002).
- **Found and fixed a latent numpy-int leak**: `EuchreState.deal_hand` built hands
  from `rng.permutation(...)` without casting to plain `int`, so `Card` values were
  actually `np.int64` everywhere in the engine, not true Python `int`. This never
  broke engine logic (numpy ints compare/hash/step like ints) but silently broke
  `json.dumps` the moment something needed to serialize a hand -- exactly what the
  report needs to do. Fixed at the source (`state.py`) rather than defensively
  converting at every consumer.

## Testing approach

The JS itself isn't executed by pytest (mirrors Slice 3b's precedent: JSON-payload
assertions, not JS execution, per `docs/adr/0009`). It *was* however actually
executed and interactively driven end-to-end outside the test suite, via a scratch
Node+jsdom script: loaded the generated HTML, stepped through every ply of every hand
in a 50-hand demo match (1,050 simulated button clicks) with no exceptions, and
specifically verified the god-view/per-seat toggle hides/shows the right hands --
including the sitting-out-partner edge case (a lone hand's non-playing partner still
holds their originally-dealt 5 cards throughout, correctly shown as face-down when
their seat isn't selected). This was a manual verification pass, not something wired
into CI/pytest; worth formalizing later if the report grows more interactive logic.

## Review outcome & post-review changes (2026-07-24)

An independent sub agent reviewed the implementation before commit: verdict
**approve with nits, no blocking bugs**. It independently traced the hidden-info
boundary claim (confirmed `_state()` is only ever reached from already-recorded-hand
replay, never a live-play path), hand-verified the seat-parity/team-name mapping in
both parities, confirmed the numpy-int64 fix has no reintroduction path (`_redeal`
uses the same `deal_hand`) and no behavioral side effects, and -- notably -- did not
just trust the as-built claim about the JS: it generated its own report and drove it
with an independent jsdom script (1,406 simulated interactions across 50 hands),
specifically re-verifying the sitting-out-partner display edge case. Nits were then
addressed:

- `EuchreMatchGameLog` archive validation accepted Python `bool` wherever an `int`
  field was checked (`bool` is an `int` subclass, so `"dealer": true` silently
  parsed as `dealer=1`) -- added a `_is_int` `TypeGuard` helper and applied it to
  every integer field (`index`, `seed`, `dealer`, `points`, `trump`, per-action
  `agent`/`action`).
- Per-action `action` values were never range-checked (only `agent` was) -- added
  `0 <= value < NUM_ACTIONS`.
- Replaced two blanket `# type: ignore[arg-type]` comments in the new tests with
  explicit `assert x is not None` narrowing, matching how the rest of the test suite
  handles `Optional` fields post-terminal.

Verified after fixes: confirmed both new validation checks actually reject malformed
input (bool-typed `dealer`, out-of-range `action`) with a quick script, not just that
the code compiles. All tests still green (78 in the new/changed files, 216 passing +
1 pre-existing skip repo-wide); ruff/format/mypy --strict clean.

## What's next

- The interactive browser play UI (`web/`) and a live text/CLI renderer were
  explicitly deferred this round (user chose "standalone match report" first) --
  still open.
- DRL encoder/PettingZoo adapter for Euchre, so `record_euchre_match` can record a
  trained policy instead of only `RandomAgent` vs `RandomAgent`.
- If a live renderer is built later, `analysis/replay_euchre.py`'s god-view/per-seat
  distinction doesn't apply the same way -- a *live* renderer showing a human player
  their own seat is back to needing the real `observation(agent)` boundary, not the
  analysis module's full-visibility snapshot.
