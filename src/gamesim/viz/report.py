"""Standalone, self-contained HTML match report (Slice 3b, torch-free).

Renders a single ``.html`` file with no external assets: all CSS/JS are inline,
there is no CDN or ``<script src=...>``, and it opens correctly from ``file://``
with no network (see docs/adr/0009-offline-analysis-and-reporting.md). Board
state is never derived by the report itself -- every ply is engine-replayed in
Python via ``gamesim.analysis.replay.replay_match_game`` and embedded as JSON;
the page's JavaScript only reads that data and draws it, so game rules never
leak into the client.

``render_match_report_html`` returns the HTML as a string (so tests don't need
the filesystem); ``write_match_report`` writes it to disk. ``main`` is the
``python -m gamesim.viz.report --log <zip> --output <html>`` CLI entry point.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from html import escape as _escape
from pathlib import Path
from typing import Any

from gamesim.analysis.replay import replay_match_game
from gamesim.analysis.summary import MatchSummary, summarize_match
from gamesim.recording.match_log import MatchGameLog, MatchLog, read_match_log


def render_match_report_html(log: MatchLog) -> str:
    """Render ``log`` to a self-contained HTML match report (as a string)."""
    summary = summarize_match(log)
    title = f"GameSim match report: {log.agent_a} vs {log.agent_b}"

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


def write_match_report(log: MatchLog, path: str | Path) -> Path:
    """Write ``log``'s HTML report to ``path`` and return the written path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_match_report_html(log), encoding="utf-8")
    return output_path


# --- embedded JSON payloads ------------------------------------------------------


def _game_payload(game: MatchGameLog) -> dict[str, Any]:
    return {
        "index": game.index,
        "seed": game.seed,
        "seats": list(game.seats),
        "actions": [[agent, column] for agent, column in game.actions],
        "outcome": game.outcome,
        # Engine-replayed, never derived here -- see module docstring.
        "boards": replay_match_game(game),
    }


def _match_data_payload(log: MatchLog) -> dict[str, Any]:
    return {
        "agent_a": log.agent_a,
        "agent_b": log.agent_b,
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


def _summary_section(summary: MatchSummary) -> str:
    outcome_bars = "\n".join(
        [
            _bar_row(f"{summary.agent_a} wins", summary.agent_a_wins, summary.total_games),
            _bar_row(f"{summary.agent_b} wins", summary.agent_b_wins, summary.total_games),
            _bar_row("draws", summary.draws, summary.total_games),
        ]
    )
    first_mover_bars = "\n".join(
        [
            _bar_row("first-mover wins", summary.first_mover_wins, summary.total_games),
            _bar_row("first-mover losses", summary.first_mover_losses, summary.total_games),
            _bar_row("first-mover draws", summary.first_mover_draws, summary.total_games),
        ]
    )
    length_section = _bar_section(
        "Game-length histogram (plies)", summary.game_length_histogram, label_prefix="plies: "
    )
    opening_section = _bar_section(
        "Opening-move distribution (column)",
        summary.opening_move_distribution,
        label_prefix="col ",
    )
    column_section = _bar_section(
        "Column-usage distribution", summary.column_usage_distribution, label_prefix="col "
    )

    return f"""<section id="summary">
  <h2>Summary</h2>
  <p>
    {_escape(summary.agent_a)} vs {_escape(summary.agent_b)} &mdash; {summary.total_games} games
  </p>
  <p>
    {_escape(summary.agent_a)} wins: <b>{summary.agent_a_wins}</b>
    ({summary.agent_a_win_rate:.1%}) &nbsp;|&nbsp;
    {_escape(summary.agent_b)} wins: <b>{summary.agent_b_wins}</b>
    ({summary.agent_b_win_rate:.1%}) &nbsp;|&nbsp;
    draws: <b>{summary.draws}</b> ({summary.draw_rate:.1%})
  </p>
  <div class="bars">
{outcome_bars}
  </div>
  <h3>First-mover breakdown (win rate {summary.first_mover_win_rate:.1%})</h3>
  <div class="bars">
{first_mover_bars}
  </div>
  <p>
    Game length: mean {summary.game_length_mean:.2f},
    min {summary.game_length_min}, max {summary.game_length_max}
  </p>
  {length_section}
  {opening_section}
  {column_section}
</section>"""


def _bar_section(title: str, rows: tuple[tuple[int, int], ...], *, label_prefix: str) -> str:
    if not rows:
        return f'<h3>{_escape(title)}</h3>\n<p class="empty">No data.</p>'
    total = sum(count for _key, count in rows)
    bars = "\n".join(_bar_row(f"{label_prefix}{key}", count, total) for key, count in rows)
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
  <h2>Step through games</h2>
  <div class="controls">
    <label for="game-select">Game:</label>
    <select id="game-select"></select>
    <button id="btn-first" type="button">|&lt;</button>
    <button id="btn-prev" type="button">&lt;</button>
    <span id="move-indicator">0 / 0</span>
    <button id="btn-next" type="button">&gt;</button>
    <button id="btn-last" type="button">&gt;|</button>
  </div>
  <p id="to-move"></p>
  <div id="board" class="board"></div>
  <p id="game-outcome"></p>
</section>"""

_STYLE = """
:root { color-scheme: light; }
body { font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }
h1 { margin-bottom: 0.25rem; }
section { margin-bottom: 2rem; }
.bars { display: flex; flex-direction: column; gap: 0.25rem; margin: 0.5rem 0 1rem; }
.bar-row { display: grid; grid-template-columns: 12rem 1fr 3rem; align-items: center; gap: 0.5rem; }
.bar-label { font-size: 0.85rem; color: #444; }
.bar-track { background: #e5e5e5; border-radius: 3px; height: 0.75rem; overflow: hidden; }
.bar-fill { background: #3366cc; height: 100%; }
.bar-count { font-size: 0.8rem; text-align: right; color: #444; }
.empty { color: #777; font-style: italic; }
.controls { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; }
.controls button { cursor: pointer; }
.board {
  display: flex;
  flex-direction: column;
  gap: 4px;
  background: #1b4fa0;
  padding: 8px;
  border-radius: 6px;
  width: fit-content;
}
.board-row { display: flex; gap: 4px; }
.cell { width: 2.2rem; height: 2.2rem; border-radius: 50%; background: #f4f4f4; }
.cell-1 { background: #e2b93b; }
.cell-2 { background: #d9483f; }
"""

_SCRIPT = """
(function () {
  var matchData = JSON.parse(document.getElementById('match-data').textContent);
  var select = document.getElementById('game-select');
  var boardEl = document.getElementById('board');
  var moveIndicator = document.getElementById('move-indicator');
  var toMoveEl = document.getElementById('to-move');
  var outcomeEl = document.getElementById('game-outcome');

  var gameIndex = 0;
  var moveIndex = 0;

  matchData.games.forEach(function (game, index) {
    var option = document.createElement('option');
    option.value = String(index);
    option.textContent =
      'Game ' + game.index + ': ' + game.seats[0] + ' vs ' + game.seats[1] +
      ' (' + game.outcome + ')';
    select.appendChild(option);
  });

  function currentGame() {
    return matchData.games[gameIndex];
  }

  function renderBoard() {
    var game = currentGame();
    var board = game.boards[moveIndex];
    boardEl.innerHTML = '';
    for (var r = board.length - 1; r >= 0; r--) {
      var rowEl = document.createElement('div');
      rowEl.className = 'board-row';
      for (var c = 0; c < board[r].length; c++) {
        var cell = document.createElement('div');
        cell.className = 'cell cell-' + board[r][c];
        rowEl.appendChild(cell);
      }
      boardEl.appendChild(rowEl);
    }
    moveIndicator.textContent = moveIndex + ' / ' + (game.boards.length - 1);
    if (moveIndex < game.actions.length) {
      var mover = game.seats[game.actions[moveIndex][0]];
      toMoveEl.textContent = 'To move: ' + mover;
      outcomeEl.textContent = '';
    } else {
      toMoveEl.textContent = '';
      outcomeEl.textContent = 'Outcome: ' + game.outcome;
    }
  }

  function setGame(index) {
    gameIndex = index;
    moveIndex = 0;
    renderBoard();
  }

  select.addEventListener('change', function () {
    setGame(parseInt(select.value, 10));
  });
  document.getElementById('btn-first').addEventListener('click', function () {
    moveIndex = 0;
    renderBoard();
  });
  document.getElementById('btn-prev').addEventListener('click', function () {
    moveIndex = Math.max(0, moveIndex - 1);
    renderBoard();
  });
  document.getElementById('btn-next').addEventListener('click', function () {
    var game = currentGame();
    moveIndex = Math.min(game.boards.length - 1, moveIndex + 1);
    renderBoard();
  });
  document.getElementById('btn-last').addEventListener('click', function () {
    var game = currentGame();
    moveIndex = game.boards.length - 1;
    renderBoard();
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
            "Write a standalone, self-contained HTML match report from a recorded MatchLog."
        )
    )
    parser.add_argument(
        "--log", type=Path, required=True, help="Path to a MatchLog ZIP (or legacy JSON) file."
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination HTML file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Read a match log and write its standalone HTML report."""
    args = _parse_args(argv)
    log = read_match_log(args.log)
    output_path = write_match_report(log, args.output)
    print(f"Wrote match report ({len(log.games)} games) to {output_path}")


if __name__ == "__main__":
    main()
