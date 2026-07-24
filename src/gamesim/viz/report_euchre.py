"""Standalone, self-contained HTML Euchre match report (torch-free).

Same shape and guarantees as ``report.py`` (inline CSS/JS, no CDN, opens from
``file://`` with no network -- docs/adr/0009-offline-analysis-and-reporting.md):
every ply is engine-replayed in Python via
``gamesim.analysis.replay_euchre.replay_euchre_match_game`` and embedded as JSON; the
page's JavaScript only reads that data and draws it.

**Hidden information / "god view":** each embedded ply snapshot already contains all
four hands (see ``replay_euchre.py``'s module docstring for why that's the correct,
deliberate choice for a report over an *already-completed* hand). The page defaults
to showing all four hands ("god view") with a per-seat toggle that hides the other
three -- purely a client-side display filter over already-fully-known data, not a
real information boundary; nothing sensitive is withheld from the browser either way.

``render_euchre_match_report_html`` returns the HTML as a string; ``write_euchre_
match_report`` writes it to disk; ``main`` is the
``python -m gamesim.viz.report_euchre --log <zip> --output <html>`` CLI entry point.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from html import escape as _escape
from pathlib import Path
from typing import Any

from gamesim.analysis.replay_euchre import replay_euchre_match_game
from gamesim.analysis.summary_euchre import EuchreMatchSummary, summarize_euchre_match
from gamesim.recording.euchre_match_log import (
    EuchreMatchGameLog,
    EuchreMatchLog,
    read_euchre_match_log,
)


def render_euchre_match_report_html(log: EuchreMatchLog) -> str:
    """Render ``log`` to a self-contained HTML match report (as a string)."""
    summary = summarize_euchre_match(log)
    title = f"GameSim Euchre match report: {log.team_a} vs {log.team_b}"

    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            f"<title>{_escape(title)}</title>",
            f"<style>{_STYLE}</style>",
            "</head>",
            "<body>",
            f"<h1>{_escape(title)}</h1>",
            _summary_section(summary),
            _INTERACTIVE_SECTION,
            _json_script("match-data", _match_data_payload(log)),
            _json_script("match-summary", asdict(summary)),
            f"<script>{_SCRIPT}</script>",
            "</body>",
            "</html>",
        ]
    )


def write_euchre_match_report(log: EuchreMatchLog, path: str | Path) -> Path:
    """Write ``log``'s HTML report to ``path`` and return the written path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_euchre_match_report_html(log), encoding="utf-8")
    return output_path


# --- embedded JSON payloads ------------------------------------------------------


def _game_payload(game: EuchreMatchGameLog) -> dict[str, Any]:
    return {
        "index": game.index,
        "seed": game.seed,
        "dealer": game.dealer,
        "seats": list(game.seats),
        "actions": [[agent, action] for agent, action in game.actions],
        "outcome": game.outcome,
        "points": game.points,
        "maker_team": game.maker_team,
        "trump": game.trump,
        "alone": game.alone,
        # Engine-replayed, never derived here -- see module docstring.
        "snapshots": [asdict(snapshot) for snapshot in replay_euchre_match_game(game)],
    }


def _match_data_payload(log: EuchreMatchLog) -> dict[str, Any]:
    return {
        "team_a": log.team_a,
        "team_b": log.team_b,
        "games": [_game_payload(game) for game in log.games],
    }


def _json_script(element_id: str, payload: object) -> str:
    """Serialize ``payload`` as a ``<script type="application/json">`` block.

    ``</`` is escaped to ``<\\/`` (a JSON-legal escape for ``/``) so no embedded
    string value can prematurely close the ``<script>`` tag.
    """
    body = json.dumps(payload).replace("</", "<\\/")
    return f'<script type="application/json" id="{element_id}">{body}</script>'


# --- summary section (inline CSS bars, no chart library) -------------------------

_SUIT_NAMES = ("spades", "hearts", "diamonds", "clubs")


def _summary_section(summary: EuchreMatchSummary) -> str:
    outcome_bars = "\n".join(
        [
            _bar_row(f"{summary.team_a} wins", summary.team_a_wins, summary.total_hands),
            _bar_row(f"{summary.team_b} wins", summary.team_b_wins, summary.total_hands),
        ]
    )
    outcome_kind_bars = "\n".join(
        [
            _bar_row("march (maker took all 5)", summary.march_count, summary.total_hands),
            _bar_row("...of which lone march", summary.lone_march_count, summary.total_hands),
            _bar_row("euchre (defenders scored)", summary.euchre_count, summary.total_hands),
        ]
    )
    points_section = _bar_section(
        "Points distribution", summary.points_distribution, label_prefix="", label_suffix=" pts"
    )
    trump_rows = tuple(
        (_SUIT_NAMES[suit] if 0 <= suit < 4 else str(suit), count)
        for suit, count in summary.trump_suit_distribution
    )
    trump_section = _named_bar_section("Trump suit distribution", trump_rows)

    return f"""<section id="summary">
  <h2>Summary</h2>
  <p>
    {_escape(summary.team_a)} vs {_escape(summary.team_b)} &mdash; {summary.total_hands} hands
  </p>
  <p>
    {_escape(summary.team_a)} wins: <b>{summary.team_a_wins}</b>
    ({summary.team_a_win_rate:.1%}) &nbsp;|&nbsp;
    {_escape(summary.team_b)} wins: <b>{summary.team_b_wins}</b>
    ({summary.team_b_win_rate:.1%})
  </p>
  <div class="bars">
{outcome_bars}
  </div>
  <h3>Hand outcomes (maker success rate {summary.maker_success_rate:.1%})</h3>
  <div class="bars">
{outcome_kind_bars}
  </div>
  <p>Going alone was called on {summary.alone_call_count} hand(s)
    ({summary.alone_call_rate:.1%}).</p>
  {points_section}
  {trump_section}
</section>"""


def _bar_section(
    title: str,
    rows: tuple[tuple[int, int], ...],
    *,
    label_prefix: str,
    label_suffix: str = "",
) -> str:
    if not rows:
        return f'<h3>{_escape(title)}</h3>\n<p class="empty">No data.</p>'
    total = sum(count for _key, count in rows)
    bars = "\n".join(
        _bar_row(f"{label_prefix}{key}{label_suffix}", count, total) for key, count in rows
    )
    return f'<h3>{_escape(title)}</h3>\n<div class="bars">\n{bars}\n</div>'


def _named_bar_section(title: str, rows: tuple[tuple[str, int], ...]) -> str:
    if not rows:
        return f'<h3>{_escape(title)}</h3>\n<p class="empty">No data.</p>'
    total = sum(count for _key, count in rows)
    bars = "\n".join(_bar_row(name, count, total) for name, count in rows)
    return f'<h3>{_escape(title)}</h3>\n<div class="bars">\n{bars}\n</div>'


def _bar_row(label: str, count: int, total: int) -> str:
    percentage = (count / total * 100) if total else 0.0
    return (
        '<div class="bar-row">'
        f'<span class="bar-label">{_escape(label)}</span>'
        '<div class="bar-track">'
        f'<div class="bar-fill" style="width:{percentage:.2f}%"></div>'
        "</div>"
        f'<span class="bar-count">{count}</span>'
        "</div>"
    )


# --- interactive step-through (markup + inline CSS/JS) ----------------------------

_INTERACTIVE_SECTION = """<section id="step-through">
  <h2>Step through hands</h2>
  <div class="controls">
    <label for="game-select">Hand:</label>
    <select id="game-select"></select>
    <button id="btn-first" type="button">|&lt;</button>
    <button id="btn-prev" type="button">&lt;</button>
    <span id="ply-indicator">0 / 0</span>
    <button id="btn-next" type="button">&gt;</button>
    <button id="btn-last" type="button">&gt;|</button>
    <label class="toggle"><input type="checkbox" id="god-view" checked> God view</label>
    <label class="toggle">Seat:
      <select id="seat-select">
        <option value="0">P0</option>
        <option value="1">P1</option>
        <option value="2">P2</option>
        <option value="3">P3</option>
      </select>
    </label>
  </div>
  <p id="status-line"></p>
  <div id="hands" class="hands"></div>
  <p id="trick-label"></p>
  <div id="trick" class="trick"></div>
  <p id="tricks-won"></p>
  <p id="outcome"></p>
</section>"""

_STYLE = """
:root { color-scheme: light; }
body { font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }
h1 { margin-bottom: 0.25rem; }
section { margin-bottom: 2rem; }
.bars { display: flex; flex-direction: column; gap: 0.25rem; margin: 0.5rem 0 1rem; }
.bar-row { display: grid; grid-template-columns: 14rem 1fr 3rem; align-items: center; gap: 0.5rem; }
.bar-label { font-size: 0.85rem; color: #444; }
.bar-track { background: #e5e5e5; border-radius: 3px; height: 0.75rem; overflow: hidden; }
.bar-fill { background: #3366cc; height: 100%; }
.bar-count { font-size: 0.8rem; text-align: right; color: #444; }
.empty { color: #777; font-style: italic; }
.controls {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;
}
.controls button { cursor: pointer; }
.toggle { font-size: 0.9rem; display: flex; align-items: center; gap: 0.3rem; }
#status-line { font-weight: 600; }
.hands { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin: 0.75rem 0; }
.seat-panel { border: 1px solid #ddd; border-radius: 6px; padding: 0.5rem; background: #fafafa; }
.seat-panel.on-turn { border-color: #3366cc; box-shadow: 0 0 0 2px rgba(51,102,204,0.25); }
.seat-title { font-weight: 600; font-size: 0.85rem; margin-bottom: 0.35rem; }
.seat-tag { font-weight: 400; color: #666; font-size: 0.8rem; }
.card-row { display: flex; flex-wrap: wrap; gap: 0.25rem; }
.card {
  display: inline-block; min-width: 1.8rem; padding: 0.15rem 0.35rem;
  border: 1px solid #999; border-radius: 4px; background: #fff;
  font-size: 0.9rem; text-align: center;
}
.card.suit-red { color: #b3261e; }
.card.suit-black { color: #1a1a1a; }
.card.card-back { background: #2a4d8f; color: transparent; border-color: #1b3563; }
.trick { display: flex; gap: 0.75rem; margin: 0.5rem 0; }
.trick .played {
  display: flex; flex-direction: column; align-items: center; gap: 0.2rem; font-size: 0.8rem;
}
"""

_SCRIPT = r"""
(function () {
  var SUITS = ['♠', '♥', '♦', '♣'];  // spades, hearts, diamonds, clubs
  var SUIT_COLOR = ['black', 'red', 'red', 'black'];
  var RANKS = ['9', '10', 'J', 'Q', 'K', 'A'];

  function cardSuit(card) { return Math.floor(card / 6); }
  function cardLabel(card) { return RANKS[card % 6] + SUITS[cardSuit(card)]; }
  function cardColorClass(card) { return 'suit-' + SUIT_COLOR[cardSuit(card)]; }
  function suitName(suit) {
    return ['spades', 'hearts', 'diamonds', 'clubs'][suit] || '?';
  }

  var matchData = JSON.parse(document.getElementById('match-data').textContent);
  var select = document.getElementById('game-select');
  var godViewEl = document.getElementById('god-view');
  var seatSelectEl = document.getElementById('seat-select');
  var handsEl = document.getElementById('hands');
  var tricksWonEl = document.getElementById('tricks-won');
  var trickLabelEl = document.getElementById('trick-label');
  var trickEl = document.getElementById('trick');
  var statusEl = document.getElementById('status-line');
  var outcomeEl = document.getElementById('outcome');
  var plyIndicator = document.getElementById('ply-indicator');

  var gameIndex = 0;
  var plyIndex = 0;

  matchData.games.forEach(function (game, index) {
    var option = document.createElement('option');
    option.value = String(index);
    var winner = game.outcome === 'team_a' ? matchData.team_a : matchData.team_b;
    option.textContent = 'Hand ' + game.index + ': ' + game.seats.join('/') +
      ' (' + winner + ' +' + game.points + ')';
    select.appendChild(option);
  });

  function currentGame() { return matchData.games[gameIndex]; }
  function currentSnapshot() { return currentGame().snapshots[plyIndex]; }

  function renderCardRow(cards, hidden) {
    var row = document.createElement('div');
    row.className = 'card-row';
    cards.forEach(function (card) {
      var el = document.createElement('span');
      if (hidden) {
        el.className = 'card card-back';
        el.textContent = '##';
      } else {
        el.className = 'card ' + cardColorClass(card);
        el.textContent = cardLabel(card);
      }
      row.appendChild(el);
    });
    return row;
  }

  function renderHands(snap, game) {
    handsEl.innerHTML = '';
    var godView = godViewEl.checked;
    var selectedSeat = parseInt(seatSelectEl.value, 10);
    for (var seat = 0; seat < 4; seat++) {
      var panel = document.createElement('div');
      panel.className = 'seat-panel' + (seat === snap.to_act && !snap.terminal ? ' on-turn' : '');

      var title = document.createElement('div');
      title.className = 'seat-title';
      var tag = '';
      if (seat === snap.dealer) tag += ' (dealer)';
      if (seat === snap.maker) tag += ' (maker' + (snap.alone ? ' alone' : '') + ')';
      if (seat === snap.sitting_out) tag += ' (sitting out)';
      title.textContent = 'P' + seat + ' -- ' + game.seats[seat];
      if (tag) {
        var tagEl = document.createElement('span');
        tagEl.className = 'seat-tag';
        tagEl.textContent = tag;
        title.appendChild(tagEl);
      }
      panel.appendChild(title);

      var hidden = !godView && seat !== selectedSeat;
      panel.appendChild(renderCardRow(snap.hands[seat], hidden));
      handsEl.appendChild(panel);
    }
  }

  function renderTrick(snap) {
    trickEl.innerHTML = '';
    if (snap.current_trick.length === 0) {
      trickLabelEl.textContent = 'Current trick: (none played yet)';
      return;
    }
    trickLabelEl.textContent = 'Current trick:';
    snap.current_trick.forEach(function (pair) {
      var agent = pair[0], card = pair[1];
      var wrap = document.createElement('div');
      wrap.className = 'played';
      var seatLabel = document.createElement('span');
      seatLabel.textContent = 'P' + agent;
      var cardEl = document.createElement('span');
      cardEl.className = 'card ' + cardColorClass(card);
      cardEl.textContent = cardLabel(card);
      wrap.appendChild(seatLabel);
      wrap.appendChild(cardEl);
      trickEl.appendChild(wrap);
    });
  }

  function renderStatus(snap) {
    var parts = [];
    parts.push('Dealer: P' + snap.dealer);
    parts.push('Trump: ' + (snap.trump === null ? '(undecided)' : suitName(snap.trump)));
    parts.push('Phase: ' + snap.phase);
    if (snap.last_action) {
      parts.push(snap.last_action.label);
    } else {
      parts.push('Hand dealt');
    }
    statusEl.textContent = parts.join('  |  ');
  }

  function renderTricksWon(snap) {
    tricksWonEl.textContent = 'Tricks won -- P0: ' + snap.tricks_won[0] +
      '  P1: ' + snap.tricks_won[1] + '  P2: ' + snap.tricks_won[2] +
      '  P3: ' + snap.tricks_won[3];
  }

  function renderOutcome(snap, game) {
    if (!snap.terminal) {
      outcomeEl.textContent = '';
      return;
    }
    var winner = game.outcome === 'team_a' ? matchData.team_a : matchData.team_b;
    outcomeEl.textContent = 'Outcome: ' + winner + ' scores ' + snap.points +
      ' point(s)' + (game.alone && snap.points === 4 ? ' (lone march)' : '') + '.';
  }

  function render() {
    var game = currentGame();
    var snap = currentSnapshot();
    plyIndicator.textContent = plyIndex + ' / ' + (game.snapshots.length - 1);
    renderStatus(snap);
    renderHands(snap, game);
    renderTrick(snap);
    renderTricksWon(snap);
    renderOutcome(snap, game);
  }

  function setGame(index) {
    gameIndex = index;
    plyIndex = 0;
    render();
  }

  select.addEventListener('change', function () { setGame(parseInt(select.value, 10)); });
  godViewEl.addEventListener('change', render);
  seatSelectEl.addEventListener('change', render);
  document.getElementById('btn-first').addEventListener('click', function () {
    plyIndex = 0;
    render();
  });
  document.getElementById('btn-prev').addEventListener('click', function () {
    plyIndex = Math.max(0, plyIndex - 1);
    render();
  });
  document.getElementById('btn-next').addEventListener('click', function () {
    plyIndex = Math.min(currentGame().snapshots.length - 1, plyIndex + 1);
    render();
  });
  document.getElementById('btn-last').addEventListener('click', function () {
    plyIndex = currentGame().snapshots.length - 1;
    render();
  });

  if (matchData.games.length > 0) {
    select.value = '0';
    setGame(0);
  }
})();
"""


# --- CLI ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a standalone, self-contained HTML Euchre match report from a "
            "recorded EuchreMatchLog."
        )
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to a EuchreMatchLog ZIP (or legacy JSON) file.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination HTML file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Read a Euchre match log and write its standalone HTML report."""
    args = _parse_args(argv)
    log = read_euchre_match_log(args.log)
    output_path = write_euchre_match_report(log, args.output)
    print(f"Wrote Euchre match report ({len(log.games)} hands) to {output_path}")


if __name__ == "__main__":
    main()
