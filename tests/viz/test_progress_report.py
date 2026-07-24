"""Tests for the standalone HTML progress report -- plan Slice 3d, test list item 2."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from gamesim.experiments.progress import (
    BaselineMetrics,
    HeadToHeadEntry,
    ProgressLog,
    StageMetrics,
)
from gamesim.viz.progress_report import render_progress_report_html, write_progress_report

_JSON_SCRIPT_RE = re.compile(
    r'<script type="application/json" id="([^"]+)">(.*?)</script>',
    re.DOTALL,
)


def _baseline(
    opponent: str,
    *,
    wins: int,
    losses: int,
    draws: int,
    game_length_mean: float,
    opening: tuple[tuple[int, int], ...],
) -> BaselineMetrics:
    total = wins + losses + draws
    return BaselineMetrics(
        opponent=opponent,
        total_games=total,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=wins / total,
        game_length_mean=game_length_mean,
        opening_move_distribution=opening,
    )


def _synthetic_progress() -> ProgressLog:
    stages = (
        StageMetrics(
            label="baseline",
            cumulative_timesteps=0,
            vs_random=_baseline(
                "random", wins=5, losses=5, draws=0, game_length_mean=6.0, opening=((3, 10),)
            ),
            vs_minimax=_baseline(
                "minimax", wins=0, losses=10, draws=0, game_length_mean=7.0, opening=((3, 10),)
            ),
        ),
        StageMetrics(
            label="step-0002048",
            cumulative_timesteps=2_048,
            vs_random=_baseline(
                "random",
                wins=15,
                losses=5,
                draws=0,
                game_length_mean=8.0,
                opening=((2, 4), (3, 16)),
            ),
            vs_minimax=_baseline(
                "minimax", wins=4, losses=15, draws=1, game_length_mean=9.0, opening=((3, 20),)
            ),
        ),
        StageMetrics(
            label="step-0006144",
            cumulative_timesteps=6_144,
            vs_random=_baseline(
                "random", wins=19, losses=1, draws=0, game_length_mean=10.0, opening=((3, 20),)
            ),
            vs_minimax=_baseline(
                "minimax",
                wins=12,
                losses=7,
                draws=1,
                game_length_mean=11.0,
                opening=((3, 15), (4, 5)),
            ),
        ),
    )
    head_to_head = (
        HeadToHeadEntry(
            row="step-0006144",
            column="baseline",
            wins=18,
            losses=2,
            draws=0,
            games=20,
            win_rate=0.9,
            loss_rate=0.1,
            draw_rate=0.0,
        ),
        HeadToHeadEntry(
            row="baseline",
            column="step-0006144",
            wins=2,
            losses=18,
            draws=0,
            games=20,
            win_rate=0.1,
            loss_rate=0.9,
            draw_rate=0.0,
        ),
        HeadToHeadEntry(
            row="step-0006144",
            column="step-0002048",
            wins=14,
            losses=5,
            draws=1,
            games=20,
            win_rate=0.7,
            loss_rate=0.25,
            draw_rate=0.05,
        ),
        HeadToHeadEntry(
            row="step-0002048",
            column="step-0006144",
            wins=5,
            losses=14,
            draws=1,
            games=20,
            win_rate=0.25,
            loss_rate=0.7,
            draw_rate=0.05,
        ),
    )
    return ProgressLog(stages=stages, head_to_head=head_to_head)


def _extract_json_block(html: str, element_id: str) -> object:
    pattern = re.compile(
        r'<script type="application/json" id="' + re.escape(element_id) + r'">(.*?)</script>',
        re.DOTALL,
    )
    match = pattern.search(html)
    assert match is not None, f"missing #{element_id} JSON script block"
    return json.loads(match.group(1))


def test_render_progress_report_html_is_self_contained() -> None:
    html = render_progress_report_html(_synthetic_progress())

    assert html.strip() != ""
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "http://" not in html
    assert "https://" not in html
    assert "<script src" not in html
    assert "cdn" not in html.lower()


def test_render_progress_report_html_embeds_the_correct_number_of_stages() -> None:
    progress = _synthetic_progress()

    html = render_progress_report_html(progress)
    data = _extract_json_block(html, "progress-data")

    assert isinstance(data, dict)
    assert len(data["stages"]) == len(progress.stages) == 3
    labels = [stage["label"] for stage in data["stages"]]
    assert labels == ["baseline", "step-0002048", "step-0006144"]


def test_render_progress_report_html_has_a_monotonic_timestep_axis() -> None:
    html = render_progress_report_html(_synthetic_progress())
    data = _extract_json_block(html, "progress-data")
    assert isinstance(data, dict)

    timesteps = [stage["cumulative_timesteps"] for stage in data["stages"]]
    assert timesteps == [0, 2_048, 6_144]
    assert all(earlier < later for earlier, later in zip(timesteps, timesteps[1:], strict=False))

    # And visibly in the stages table (not just the embedded JSON), formatted with
    # thousands separators.
    assert "2,048" in html
    assert "6,144" in html


def test_render_progress_report_html_embeds_specific_head_to_head_values() -> None:
    progress = _synthetic_progress()

    html = render_progress_report_html(progress)
    data = _extract_json_block(html, "progress-data")
    assert isinstance(data, dict)

    entries = {(entry["row"], entry["column"]): entry for entry in data["head_to_head"]}
    assert len(entries) == 4
    assert entries[("step-0006144", "baseline")]["win_rate"] == 0.9
    assert entries[("step-0006144", "baseline")]["wins"] == 18
    assert entries[("baseline", "step-0006144")]["win_rate"] == 0.1
    assert entries[("step-0006144", "step-0002048")]["draws"] == 1

    # And rendered visibly in the head-to-head table.
    assert "90.0%" in html
    assert "18-2-0" in html


def test_render_progress_report_html_shows_opening_and_game_length_data() -> None:
    progress = _synthetic_progress()

    html = render_progress_report_html(progress)
    data = _extract_json_block(html, "progress-data")
    assert isinstance(data, dict)

    # Opening-move distributions are embedded per stage per baseline.
    baseline_stage = data["stages"][0]
    assert baseline_stage["vs_random"]["opening_move_distribution"] == [[3, 10]]
    late_stage = data["stages"][2]
    assert late_stage["vs_minimax"]["opening_move_distribution"] == [[3, 15], [4, 5]]

    # Game-length means round-trip exactly.
    assert late_stage["vs_random"]["game_length_mean"] == 10.0
    assert late_stage["vs_minimax"]["game_length_mean"] == 11.0
    assert "10.00" in html
    assert "11.00" in html


def test_render_progress_report_html_handles_an_empty_progress_log() -> None:
    empty = ProgressLog(stages=())

    html = render_progress_report_html(empty)

    assert html.strip() != ""
    assert html.lstrip().startswith("<!DOCTYPE html>")
    data = _extract_json_block(html, "progress-data")
    assert isinstance(data, dict)
    assert data["stages"] == []
    assert data["head_to_head"] == []


def test_render_progress_report_html_escapes_malicious_stage_labels() -> None:
    malicious_label = "<script>alert(1)</script>"
    progress = ProgressLog(
        stages=(
            StageMetrics(
                label=malicious_label,
                cumulative_timesteps=0,
                vs_random=_baseline(
                    "random", wins=1, losses=1, draws=0, game_length_mean=5.0, opening=((3, 2),)
                ),
                vs_minimax=_baseline(
                    "minimax", wins=1, losses=1, draws=0, game_length_mean=5.0, opening=((3, 2),)
                ),
            ),
        ),
    )

    html = render_progress_report_html(progress)

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html

    blocks = _JSON_SCRIPT_RE.findall(html)
    block_ids = {element_id for element_id, _body in blocks}
    assert "progress-data" in block_ids
    for _element_id, body in blocks:
        payload = json.loads(body)
        assert isinstance(payload, dict)


def test_write_progress_report_writes_a_file(tmp_path: Path) -> None:
    progress = _synthetic_progress()
    output = tmp_path / "progress.html"

    result_path = write_progress_report(progress, output)

    assert result_path == output
    assert output.exists()
    html = output.read_text(encoding="utf-8")
    data = _extract_json_block(html, "progress-data")
    assert isinstance(data, dict)
    assert len(data["stages"]) == 3


def test_progress_report_module_does_not_import_torch() -> None:
    script = "import sys\nimport gamesim.viz.progress_report\nassert 'torch' not in sys.modules\n"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
