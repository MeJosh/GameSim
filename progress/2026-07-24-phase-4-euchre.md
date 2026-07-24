# Progress — Phase 4: Euchre (2026-07-24)

**Status:** Engine complete. 49 new tests green (236 total across the repo); ruff +
`ruff format --check` + `mypy --strict` clean on `core` + `games` (RL/web modules
untouched, still gated on torch/fastapi per Phase 2/3 notes). DRL encoder/adapter,
visualization, and the multi-hand match wrapper are deliberately out of scope — see
"What's next" below.

## What was built

- `games/euchre/cards.py` — card encoding (`card_id = suit*6 + rank`), the 24-card
  deck, and the bower rule (`effective_suit`, `trump_rank`, `plain_rank`) shared by
  masking and trick resolution.
- `games/euchre/state.py` — `EuchreState` (one hand's worth of state), `EuchreRules`
  (currently just `stick_the_dealer`), seeded dealing.
- `games/euchre/actions.py` — a single flat 35-action space (cards 0-23 reused across
  discard/play, plus pass/order-up/call-suit +alone variants), masked by phase.
- `games/euchre/engine.py` — `EuchreEngine`: full phase machine (round-1 bid → discard
  or round-2 bid → trick play → scoring), stick-the-dealer forcing, redeal fallback,
  going-alone turn-skipping, per-agent hidden-hand observation boundary.
- `tests/games/test_euchre.py` — 49 tests covering plan groups A-K (bower logic,
  bidding both rounds, discard, follow-suit masking, trick-winner determination, going
  alone, scoring formula, observation boundary, determinism, full random-agent games
  via `Runner`).
- `plans/phase-04-euchre.md` — the detailed spec this was built against.
- `plans/roadmap.md` — Phase 4 updated to reflect Euchre replacing the original
  Nim/Tic-Tac-Toe placeholder (user decision, 2026-07-24).

## Decisions worth remembering

- **One hand = one engine episode**, not a first-to-10 match. `reset()` deals one
  hand; `is_terminal()` fires once the 5th trick is scored. A `EuchreMatch` wrapper
  (cumulative score, dealer rotation, stop at 10) is a thin layer to add later, kept
  out of the engine on purpose — mirrors how `Runner` sits *outside* `Engine` rather
  than the engine knowing about game loops.
- **Action space is one flat 35-int space**, masked by phase — not a tagged union per
  phase. This keeps the same "action is just an int, mask says what's legal" contract
  Connect Four established, which matters for `MaskablePPO` later (Phase 4 follow-up
  or Phase 5): no encoder rework needed to add a new phase's actions, just extend the
  mask logic.
- **`to_act` lives directly on `EuchreState`** as the single source of truth for whose
  turn it is, updated explicitly by the engine at every transition (bid advance, trick
  leader after resolution, skip-sitting-out-partner). Chosen over re-deriving "whose
  turn" from phase + other fields each call, because going-alone's skip logic is
  genuinely stateful (which seat is skipped changes hand-to-hand) and re-deriving it
  every call would just reimplement the same bookkeeping less directly.
- **`EuchreRules(stick_the_dealer=True)` is the default**, matching what the user
  confirmed. The `False` path (redeal on an all-pass round 2) is implemented and has
  one dedicated test, but hasn't been exercised as heavily as the default path — worth
  more scrutiny before leaning on it.
- **Bower logic centralized in `cards.py`** (`effective_suit`/`trump_rank`) rather than
  inlined in the engine's masking/trick-resolution code, specifically so a future
  encoder or renderer doesn't have to re-derive "is this card trump" from scratch.

## Testing approach

Bidding/discard/masking (groups C-E) is driven through real `step()` calls from a
fresh `reset()`, same as Connect Four. Trick-winner logic, going-alone scoring, and the
scoring-formula tests instead construct `EuchreState` fields directly after a normal
reset (documented in the test file's module docstring) — dealt hands are random, so
pinning an exact bower/trick scenario through real bidding would mean fighting the RNG
rather than testing the rule. This is the same pragmatic call Connect Four made with
its hardcoded verified-draw fixture, just applied more often because Euchre has more
distinct rule branches to isolate.

## What's next (not done here, noted so it isn't mistaken for forgotten)

- `EuchreMatch` wrapper for first-to-10 play (dealer rotation, cumulative score).
- DRL encoder + PettingZoo adapter for Euchre (Phase 2's Connect Four equivalent).
- A renderer (Phase 3's equivalent) — trickier here since trump/hands are hidden by
  design; a "god view" debug renderer vs. a per-seat renderer are different asks.
- Heavier exercise of the `stick_the_dealer=False` / redeal path.
- Revisit `core` for any Euchre-shaped assumptions that leaked in — none found during
  this build (no `core` changes were needed), but worth a second look once DRL touches
  it.

## Review outcome & post-review changes (2026-07-24)

An independent sub agent reviewed the implementation before commit: verdict **approve
with nits, no blocking bugs**. It independently ran the test suite, ruff, and
`mypy --strict`; fuzzed 300 seeded games checking every `EuchreObservation` against
every other seat's real hand at that instant (zero leaks found); and live-traced two
edge cases not yet covered by a test (dealer orders up alone on their own upcard;
maker's partner is the dealer when going alone) to confirm both already worked
correctly. Nits were then addressed:

- Added dedicated tests for both edge cases the review traced manually
  (`test_dealer_orders_up_alone_on_own_upcard`,
  `test_dealer_still_discards_when_sitting_out_as_makers_partner`).
- Split the single 3-or-4-tricks scoring test into two (`test_partnered_three_tricks_
  scores_one`, `test_partnered_four_tricks_scores_one`) — the original name promised
  both branches but only exercised one.
- Fixed doc drift in phase-04-euchre.md: `plain_rank` was documented with a
  `(card, led_suit, trump)` signature it never had; corrected to `plain_rank(card)`.
- Strengthened the `stick_the_dealer=False` redeal coverage: the review noted it never
  fired during the plain-`RandomAgent` fuzz runs (the trigger's true probability under
  uniform-random bidding is astronomically low, ~1-in-200,000 per hand). Replaced the
  no-op fuzz assertion with a pass-biased (but still per-turn-randomized) agent that
  makes a redeal likely within a handful of hands, and asserted one actually occurs
  across 20 seeds — not just that the code path exists.

Test count: 49 → **53**. All green; ruff/format/mypy --strict clean after the fixes.
