# Phase 4 — Euchre: a second, larger, hidden-information game (detailed plan)

**Status:** ✅ Engine complete, independently reviewed (approve with nits, no blocking
bugs; nits addressed) — 53 tests green; ruff, format, mypy --strict clean on `core` +
`games`. See [progress/2026-07-24-phase-4-euchre.md](../progress/2026-07-24-phase-4-euchre.md)
for as-built notes. **Visualization (standalone HTML match report) is also done** —
see [progress/2026-07-24-phase-4-euchre-viz.md](../progress/2026-07-24-phase-4-euchre-viz.md).
DRL encoder, the interactive browser play UI, and the first-to-10 match wrapper
remain open follow-ups (see that write-up's "What's next").

**Goal:** Implement a 4-player, partnership Euchre engine against the existing `core`
interfaces with **no changes to `core`**, proving the framework generalizes beyond
Connect Four's 2-player perfect-information case. Euchre replaces the roadmap's original
Phase 4 placeholder (Nim/Tic-Tac-Toe) — see [roadmap.md](roadmap.md) and the decision
note there.

**Why Euchre over the original placeholder:** Nim/Tic-Tac-Toe would have proven
game-agnosticism on the file-layout axis only. Euchre exercises three things Connect
Four cannot and that MTG (Phase 5) will need for real: **N > 2 agents with fixed
partnerships** (team scoring, one agent's action affecting a teammate's outcome),
**hidden information** (`observation(agent)` must hide other hands — the first real test
of the boundary Connect Four only stubbed), and **a heterogeneous, multi-phase action
space** (bidding, discarding, and card play all share one engine, not one action kind
repeated as in Connect Four).

**Rules baseline (confirmed with the user):** standard 24-card (9–A) Euchre, 2 fixed
partnerships seated across from each other, two-round bidding (order-up / name trump),
right+left bower, going alone, standard scoring (1 / 2 / 4, euchre = 2 to defenders),
**with "stick the dealer"** (dealer cannot pass in round 2 if everyone else has passed).
No no-trump variant.

## Scoping decision: one hand = one engine episode

A full Euchre *match* is first-to-10 across many hands. Modeling the whole match as one
`Engine` episode would tangle match-level bookkeeping (dealer rotation, running score,
"should I go alone to close out the game") into the same class as single-hand rules.
Instead, **the engine's episode boundary is one hand**: `reset()` deals a new hand,
`is_terminal()` becomes true once the hand's 5 tricks are played and scored, and
`rewards()` reports that hand's point swing. This mirrors Connect Four (one `reset()` =
one game) and keeps `EuchreEngine` focused on rules, not match orchestration.

A `EuchreMatch` wrapper (run N hands, carry cumulative score, rotate the dealer, stop at
10) is deliberately **out of scope for this phase** — it's a thin loop over the engine,
analogous to how `Runner` is a thin loop over `Engine` + `Agent`, and can be added later
without touching the engine. Noted as a follow-up, not a gap.

## Action space

A single fixed-size discrete space (35 actions), matching Connect Four's "one action
type" pattern but sized for Euchre's heterogeneous phases; `legal_actions()` masks by
game phase exactly like Connect Four masks by board state:

| Indices | Meaning |
|---|---|
| 0–23 | Play/discard card `card_id` (`suit*6 + rank`; see `cards.py`) |
| 24 | Pass (round-1 or round-2 bidding) |
| 25 | Order up the upcard (round 1) |
| 26 | Order up the upcard, going alone (round 1) |
| 27–30 | Call trump = suit `i` (round 2, `i` in `SPADES..CLUBS` order) |
| 31–34 | Call trump = suit `i`, going alone (round 2) |

Card actions are reused across two phases (dealer discard vs. trick play) the same way
Connect Four's column action is reused across every turn — the phase, not the action
value, determines interpretation, and `legal_actions()` restricts which cards are legal
in each case (discard: the dealer's own post-pickup hand; trick play: follow-suit rule).

## Bower rule (the one genuinely new piece of card logic)

Trump suit has 7 effective cards: its own 9/10/J/Q/K/A (right bower = trump's jack) plus
the jack of the **same-color** suit (left bower), which is treated as trump — both for
follow-suit and for trick-winner comparison — not as a member of its printed suit. The
same-color suit therefore plays this hand with only 5 cards (missing its jack).
`cards.py` centralizes this as `effective_suit(card, trump)`, `trump_rank(card, trump)`,
and `plain_rank(card)` so the engine and any future encoder share one source of truth
instead of re-deriving bower logic ad hoc.

## Definition of done

- 4 `RandomAgent`s play a complete, legal hand to a correct terminal outcome through
  `Runner`, only ever choosing legal moves (bidding, discard, and trick play all masked
  correctly).
- Bower ranking and follow-suit are correct, including the "left bower borrowed from the
  other suit" cases in both directions (red trump / black trump).
- Going alone correctly sits the partner out of trick play and pays out the 4-point lone
  march.
- Stick-the-dealer forces a call on the dealer's round-2 turn when everyone else passed;
  the non-stick-the-dealer path (config off) redeals instead, tested separately.
- `observation(agent)` never leaks another agent's hand contents.
- Deterministic: same seed ⇒ identical deal, identical legal random-agent play sequence.
- `pytest` green; `ruff` clean; `mypy --strict` clean on `core` + `games/euchre`.

## Test list (the specification)

### A. Cards & bower logic
1. Deck has 24 unique cards, 6 ranks × 4 suits.
2. `effective_suit`: the left bower's effective suit is trump, not its printed suit.
3. Trump ranking order: right bower > left bower > A > K > Q > 10 > 9 of trump.
4. Off-trump-color suit (the one missing its jack to the left bower) ranks A>K>Q>10>9
   with no jack; unaffected suits rank the normal A>K>Q>J>10>9.
5. Symmetry: this holds for both a red trump (hearts/diamonds pairing) and a black trump
   (spades/clubs pairing).

### B. Deal & construction
6. A fresh hand deals 5 cards to each of 4 agents, one upcard, 3 cards left undealt (24 −
   4×5 − 1 = 3, consistent with a 24-card deck).
7. `agents()` reports all 4 `AgentId`s; `current_agent()` at hand start is the player
   left of the dealer (round-1 bidding).
8. `is_terminal()` is `False` and `rewards()` is 0 for all agents at hand start.

### C. Round-1 bidding (order-up)
9. Each of the 4 players in turn (starting left of dealer) may order-up, order-up-alone,
   or pass; mask reflects exactly those 3 options in round 1, nothing else.
10. Ordering up sets trump = upcard's suit, maker = that agent's team, and transitions to
    `DEALER_DISCARD` (dealer picks up the upcard: 6 cards in hand).
11. Ordering up alone additionally records the alone flag and marks the maker's partner
    as sitting out for this hand.
12. All 4 passing round 1 transitions to round-2 bidding; no trump is set yet.

### D. Round-2 bidding (name trump) + stick the dealer
13. Round 2 offers call-suit / call-suit-alone for the 3 suits **other than** the turned-
    down suit, plus pass — mask excludes the turned-down suit's two actions and all
    card/round-1 actions.
14. Naming a suit sets trump/maker (and alone/sitting-out if alone) directly — no
    discard step (upcard was never picked up).
15. Stick-the-dealer: when it's the dealer's turn in round 2 and the other 3 already
    passed, `pass` (24) is illegal in the mask; the dealer must call.
16. With stick-the-dealer disabled (`EuchreRules(stick_the_dealer=False)`), all 4 passing
    round 2 triggers a redeal (new shuffle, dealer advances, back to round-1 bidding)
    rather than a rules error.

### E. Dealer discard
17. After an order-up, only the dealer may act; mask is exactly the dealer's 6 held
    cards (the 5 dealt + picked-up upcard).
18. Discarding returns the dealer to 5 cards and transitions to `TRICK_PLAY`, first lead
    = player left of dealer.

### F. Trick play: follow-suit masking
19. A player holding a card of the suit led (bower-aware) must play one; mask contains
    only those cards.
20. A player with no card of the suit led (bower-aware) may play any card in hand.
21. The left bower is legal when trump, not the upcard's suit, is led (it "is" trump for
    follow-suit purposes).

### G. Trick play: winner determination
22. Right bower beats every other card including the left bower and every off-suit ace.
23. Left bower beats every trump-suit card except the right bower.
24. Highest card of the suit led wins when no trump was played.
25. A trump played "in" (not led) wins over the suit led even if lower-ranked than the
    card led, as long as it's the highest trump in the trick.

### H. Going alone
26. The sitting-out partner never appears as `current_agent()` during trick play for that
    hand; trick-leader rotation skips them.
27. A lone hand that takes all 5 tricks scores 4 (not 2).
28. A lone hand that is euchred still costs the defenders' 2, same as a partnered hand.

### I. Scoring & terminal
29. Maker team takes 3–4 tricks ⇒ 1 point; both agents on that team get reward +1, the
    other team −1.
30. Maker team takes all 5 (not alone) ⇒ 2 points (march).
31. Maker team takes < 3 tricks ⇒ euchred, defending team scores 2.
32. `is_terminal()` becomes `True` exactly when the 5th trick resolves and scoring is
    applied; `step` after terminal raises; masks are all-false post-terminal.

### J. Observation boundary (hidden information — the new axis vs. Connect Four)
33. `observation(agent)` exposes only that agent's own hand contents; other agents'
    hand sizes may be public but their card identities must not appear anywhere in the
    observation.
34. The upcard is visible to all during round-1 bidding and the discard step; once
    turned down (round 2 begins), it is no longer part of any observation as a live
    card (only the turned-down *suit* is public).
35. `perspective_agent` is honored (mirrors Connect Four's fix): querying two different
    agents mid-hand returns different hand contents, not the same object reused.

### K. Determinism & the Runner
36. Same seed ⇒ identical deal (same hands, same upcard, same dealer).
37. 4 `RandomAgent`s driven by `Runner` complete a full hand choosing only masked-legal
    actions, for many seeds, without the engine raising.
38. Same seed + same action sequence ⇒ identical final state/rewards across two runs.

## Build order

1. `games/euchre/cards.py` — suits, ranks, `Card` (int alias like Connect Four's
   `Action`), deck, `effective_suit`, trump/plain rank ordering. Tests A.
2. `games/euchre/state.py` — `EuchreState` (hands, dealer, phase enum, upcard,
   trump/maker/alone/sitting-out, current trick, tricks won, scores), `EuchreRules`
   (`stick_the_dealer: bool = True`), seeded deal. Tests B.
3. `games/euchre/actions.py` — the 35-action space as named `int` constants (mirrors
   `connect_four/actions.py`'s "just an int" simplicity).
4. `games/euchre/engine.py` — `EuchreEngine` implementing `core.engine.Engine`: phase
   state machine (round-1 bid → discard-or-round-2 → round-2 bid → trick play →
   scoring), masking per phase, bower-aware trick resolution, going-alone turn skipping,
   `EuchreObservation`. Tests C–K.
5. `tests/games/test_euchre.py` — the full list above, following the red→green method
   from [ADR 0005](../docs/adr/0005-tdd-red-green.md).

## Out of scope for Phase 4

Multi-hand match wrapper (first-to-10), DRL encoder/PettingZoo adapter for Euchre, a
live text/CLI renderer or interactive browser play UI, ECS. Card-play strategy
heuristics beyond `RandomAgent`. (The standalone HTML match report *is* now built —
see progress/2026-07-24-phase-4-euchre-viz.md — but the rest of this list still
stands.) Those
are natural Phase-4-followup or Phase-5-adjacent work once the engine itself is proven.
