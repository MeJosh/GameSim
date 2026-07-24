"""Standalone, self-contained HTML training-progress report (Slice 3d, torch-free).

Mirrors ``gamesim.viz.report``'s approach (see its module docstring and
docs/adr/0009-offline-analysis-and-reporting.md): a single ``.html`` file, inline
CSS only, no CDN, no ``<script src=...>``, no network. Unlike the match report this
page needs no client-side interactivity -- every chart (win rate vs baselines,
game-length trend, opening-move distribution shift, head-to-head matrix) is a static
SVG/HTML table rendered entirely in Python from a ``gamesim.experiments.progress.
ProgressLog``. The same numeric data is additionally embedded as a JSON
``<script type="application/json">`` block purely for inspection/testing -- no page
logic depends on it.

Torch-free: this module only imports ``gamesim.experiments.progress`` (itself
torch-free) plus the standard library.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from html import escape as _escape
from pathlib import Path

from gamesim.experiments.progress import (
    HeadToHeadEntry,
    ProgressLog,
    StageMetrics,
    read_progress_log,
)

_CHART_WIDTH = 560
_CHART_HEIGHT = 180
_CHART_PADDING = 28
_RANDOM_COLOR = "#3366cc"
_MINIMAX_COLOR = "#d9483f"


def render_progress_report_html(progress: ProgressLog) -> str:
    """Render ``progress`` to a self-contained HTML progress report (as a string)."""
    title = "GameSim training progress report"

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
            _stages_section(progress.stages),
            _winrate_section(progress.stages),
            _game_length_section(progress.stages),
            _opening_distribution_section(progress.stages),
            _head_to_head_section(progress.head_to_head),
            _json_script("progress-data", progress.to_dict()),
            "</body>",
            "</html>",
        ]
    )


def write_progress_report(progress: ProgressLog, path: str | Path) -> Path:
    """Write ``progress``'s HTML report to ``path`` and return the written path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_progress_report_html(progress), encoding="utf-8")
    return output_path


# --- embedded JSON -----------------------------------------------------------------


def _json_script(element_id: str, payload: object) -> str:
    """Serialize ``payload`` as a ``<script type="application/json">`` block.

    ``</`` is escaped to ``<\\/`` (a JSON-legal escape for ``/``) so no embedded
    string value can prematurely close the ``<script>`` tag -- same guard as
    ``gamesim.viz.report``.
    """
    body = json.dumps(payload).replace("</", "<\\/")
    return f'<script type="application/json" id="{element_id}">{body}</script>'


# --- stages table --------------------------------------------------------------------


def _stages_section(stages: tuple[StageMetrics, ...]) -> str:
    if not stages:
        return '<section id="stages"><h2>Stages</h2><p class="empty">No stages.</p></section>'
    rows = "\n".join(
        "<tr>"
        f"<td>{_escape(stage.label)}</td>"
        f"<td>{stage.cumulative_timesteps:,}</td>"
        f"<td>{stage.vs_random.win_rate:.1%} "
        f"({stage.vs_random.wins}-{stage.vs_random.losses}-{stage.vs_random.draws})</td>"
        f"<td>{stage.vs_minimax.win_rate:.1%} "
        f"({stage.vs_minimax.wins}-{stage.vs_minimax.losses}-{stage.vs_minimax.draws})</td>"
        f"<td>{stage.vs_random.game_length_mean:.2f}</td>"
        f"<td>{stage.vs_minimax.game_length_mean:.2f}</td>"
        f"<td>{_match_log_links(stage.match_log_paths)}</td>"
        "</tr>"
        for stage in stages
    )
    return f"""<section id="stages">
  <h2>Stages</h2>
  <table>
    <thead>
      <tr>
        <th>Stage</th><th>Cumulative timesteps</th>
        <th>Win rate vs random (W-L-D)</th><th>Win rate vs minimax (W-L-D)</th>
        <th>Game length vs random</th><th>Game length vs minimax</th>
        <th>Match logs</th>
      </tr>
    </thead>
    <tbody>
{rows}
    </tbody>
  </table>
</section>"""


def _match_log_links(match_log_paths: Mapping[str, str]) -> str:
    """Render links to a stage's recorded per-opponent match logs, if any.

    Paths are relative to the run directory (see ``StageMetrics.match_log_paths``),
    which is where this report is expected to be written alongside ``progress.json``
    -- so a plain relative ``href`` is enough to open them (e.g. in
    ``gamesim.viz.report``) without embedding or copying the archives themselves.
    """
    if not match_log_paths:
        return '<span class="empty">&mdash;</span>'
    return ", ".join(
        f'<a href="{_escape(path)}">{_escape(key)}</a>' for key, path in match_log_paths.items()
    )


# --- line charts (win rate / game length over cumulative timesteps) ----------------


def _winrate_section(stages: tuple[StageMetrics, ...]) -> str:
    x_values = [float(stage.cumulative_timesteps) for stage in stages]
    series = [
        ("vs random", [stage.vs_random.win_rate for stage in stages], _RANDOM_COLOR),
        ("vs minimax", [stage.vs_minimax.win_rate for stage in stages], _MINIMAX_COLOR),
    ]
    chart = _line_chart(
        "Win rate vs baselines over cumulative timesteps", x_values, series, y_max=1.0
    )
    return f'<section id="winrate">\n{chart}\n</section>'


def _game_length_section(stages: tuple[StageMetrics, ...]) -> str:
    x_values = [float(stage.cumulative_timesteps) for stage in stages]
    series = [
        (
            "vs random",
            [stage.vs_random.game_length_mean for stage in stages],
            _RANDOM_COLOR,
        ),
        (
            "vs minimax",
            [stage.vs_minimax.game_length_mean for stage in stages],
            _MINIMAX_COLOR,
        ),
    ]
    chart = _line_chart("Mean game length over cumulative timesteps", x_values, series)
    return f'<section id="game-length">\n{chart}\n</section>'


def _line_chart(
    title: str,
    x_values: list[float],
    series: Sequence[tuple[str, list[float], str]],
    *,
    y_max: float | None = None,
) -> str:
    if not x_values:
        return f'<h3>{_escape(title)}</h3>\n<p class="empty">No data.</p>'

    x_min, x_max = min(x_values), max(x_values)
    x_span = (x_max - x_min) or 1.0
    if y_max is None:
        all_y = [y for _label, y_values, _color in series for y in y_values]
        y_max = max(all_y) if all_y else 1.0
    y_max = y_max or 1.0

    def plot_x(x: float) -> float:
        return _CHART_PADDING + (x - x_min) / x_span * (_CHART_WIDTH - 2 * _CHART_PADDING)

    def plot_y(y: float) -> float:
        y_max_local = y_max if y_max is not None else 1.0
        return (
            _CHART_HEIGHT
            - _CHART_PADDING
            - (y / y_max_local) * (_CHART_HEIGHT - 2 * _CHART_PADDING)
        )

    parts: list[str] = []
    legend: list[str] = []
    for label, y_values, color in series:
        points = " ".join(
            f"{plot_x(x):.1f},{plot_y(y):.1f}" for x, y in zip(x_values, y_values, strict=True)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" />'
        )
        for x, y in zip(x_values, y_values, strict=True):
            parts.append(
                f'<circle cx="{plot_x(x):.1f}" cy="{plot_y(y):.1f}" r="3" fill="{color}" />'
            )
        legend.append(
            f'<span class="legend-item"><span class="swatch" style="background:{color}">'
            f"</span>{_escape(label)}</span>"
        )

    svg = (
        f'<svg viewBox="0 0 {_CHART_WIDTH} {_CHART_HEIGHT}" class="chart" role="img" '
        f'aria-label="{_escape(title)}">' + "".join(parts) + "</svg>"
    )
    return f'<h3>{_escape(title)}</h3>\n<div class="legend">{"".join(legend)}</div>\n{svg}'


# --- opening-move distribution shift -------------------------------------------------


def _opening_distribution_section(stages: tuple[StageMetrics, ...]) -> str:
    if not stages:
        return (
            '<section id="openings"><h2>Opening-move distribution by stage</h2>'
            '<p class="empty">No data.</p></section>'
        )
    blocks = []
    for stage in stages:
        blocks.append(
            _distribution_block(stage.label, "vs random", stage.vs_random.opening_move_distribution)
        )
        blocks.append(
            _distribution_block(
                stage.label, "vs minimax", stage.vs_minimax.opening_move_distribution
            )
        )
    return (
        '<section id="openings">\n<h2>Opening-move distribution by stage</h2>\n'
        + "\n".join(blocks)
        + "\n</section>"
    )


def _distribution_block(
    stage_label: str, opponent_label: str, distribution: tuple[tuple[int, int], ...]
) -> str:
    heading = f"{_escape(stage_label)} &mdash; {_escape(opponent_label)}"
    if not distribution:
        return f'<h4>{heading}</h4>\n<p class="empty">No data.</p>'
    total = sum(count for _column, count in distribution)
    bars = "\n".join(_bar_row(f"col {column}", count, total) for column, count in distribution)
    return f'<h4>{heading}</h4>\n<div class="bars">\n{bars}\n</div>'


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


# --- head-to-head matrix ---------------------------------------------------------------


def _head_to_head_section(head_to_head: tuple[HeadToHeadEntry, ...]) -> str:
    if not head_to_head:
        return (
            '<section id="head-to-head"><h2>Head-to-head win rates</h2>'
            '<p class="empty">No data.</p></section>'
        )
    labels: list[str] = []
    for entry in head_to_head:
        if entry.row not in labels:
            labels.append(entry.row)
        if entry.column not in labels:
            labels.append(entry.column)
    lookup = {(entry.row, entry.column): entry for entry in head_to_head}

    header_cells = "".join(f"<th>{_escape(label)}</th>" for label in labels)
    body_rows = []
    for row_label in labels:
        cells = []
        for column_label in labels:
            if row_label == column_label:
                cells.append('<td class="diagonal">&mdash;</td>')
                continue
            found = lookup.get((row_label, column_label))
            if found is None:
                cells.append('<td class="empty">n/a</td>')
            else:
                cells.append(
                    f"<td>{found.win_rate:.1%} ({found.wins}-{found.losses}-{found.draws})</td>"
                )
        body_rows.append(f"<tr><th>{_escape(row_label)}</th>{''.join(cells)}</tr>")

    return f"""<section id="head-to-head">
  <h2>Head-to-head win rates (row vs column)</h2>
  <table>
    <thead><tr><th></th>{header_cells}</tr></thead>
    <tbody>
{chr(10).join(body_rows)}
    </tbody>
  </table>
</section>"""


# --- styling -------------------------------------------------------------------------

_STYLE = """
:root { color-scheme: light; }
body { font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }
h1 { margin-bottom: 0.25rem; }
section { margin-bottom: 2rem; }
table { border-collapse: collapse; }
th, td { border: 1px solid #ccc; padding: 0.3rem 0.6rem; text-align: right; font-size: 0.85rem; }
th:first-child, td:first-child { text-align: left; }
td.diagonal, td.empty { color: #999; text-align: center; }
.bars { display: flex; flex-direction: column; gap: 0.25rem; margin: 0.25rem 0 0.75rem; }
.bar-row { display: grid; grid-template-columns: 6rem 1fr 3rem; align-items: center; gap: 0.5rem; }
.bar-label { font-size: 0.8rem; color: #444; }
.bar-track { background: #e5e5e5; border-radius: 3px; height: 0.6rem; overflow: hidden; }
.bar-fill { background: #3366cc; height: 100%; }
.bar-count { font-size: 0.75rem; text-align: right; color: #444; }
.empty { color: #777; font-style: italic; }
.chart { width: 100%; max-width: 560px; border: 1px solid #ddd; background: #fafafa; }
.legend { display: flex; gap: 1rem; font-size: 0.8rem; margin-bottom: 0.25rem; }
.legend-item { display: inline-flex; align-items: center; gap: 0.25rem; }
.swatch { width: 0.7rem; height: 0.7rem; border-radius: 50%; display: inline-block; }
"""


# --- CLI ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a standalone, self-contained HTML training-progress report from "
            "a versioned progress.json (see gamesim.experiments.progress)."
        )
    )
    parser.add_argument(
        "--progress", type=Path, required=True, help="Path to a progress.json (v2 schema)."
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination HTML file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Read a progress log and write its standalone HTML report."""
    args = _parse_args(argv)
    progress = read_progress_log(args.progress)
    output_path = write_progress_report(progress, args.output)
    print(f"Wrote progress report ({len(progress.stages)} stages) to {output_path}")


if __name__ == "__main__":
    main()
