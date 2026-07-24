"""Tests for the standalone HTML Euchre match report."""

from __future__ import annotations

import json
import re
from pathlib import Path

from gamesim.analysis import replay_euchre_match_game, summarize_euchre_match
from gamesim.core.agent import RandomAgent
from gamesim.games.euchre import EuchreObservation
from gamesim.recording import record_euchre_match, write_euchre_match_log
from gamesim.recording.euchre_match_log import EuchreMatchLog
from gamesim.viz import report_euchre

_ALL_JSON_SCRIPT_BLOCK_RE = re.compile(
    r'<script type="application/json" id="([^"]+)">(.*?)</script>',
    re.DOTALL,
)


def _recorded_log(num_hands: int = 3, seed: int = 5) -> EuchreMatchLog:
    return record_euchre_match(
        RandomAgent[EuchreObservation](seed=1),
        RandomAgent[EuchreObservation](seed=2),
        team_a_name="alpha",
        team_b_name="beta",
        num_hands=num_hands,
        seed=seed,
    )


def _extract_json_block(html: str, element_id: str) -> object:
    pattern = re.compile(
        r'<script type="application/json" id="' + re.escape(element_id) + r'">(.*?)</script>',
        re.DOTALL,
    )
    match = pattern.search(html)
    assert match is not None, f"missing #{element_id} JSON script block"
    return json.loads(match.group(1))


def test_write_euchre_match_report_embeds_all_hands_and_matches_engine_replay(
    tmp_path: Path,
) -> None:
    log = _recorded_log()
    output = tmp_path / "report.html"

    result_path = report_euchre.write_euchre_match_report(log, output)

    assert result_path == output
    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert html.strip() != ""
    assert html.lstrip().startswith("<!DOCTYPE html>")

    match_data = _extract_json_block(html, "match-data")
    assert isinstance(match_data, dict)
    assert len(match_data["games"]) == len(log.games)

    chosen_game = log.games[1]
    embedded_snapshots = match_data["games"][1]["snapshots"]
    expected = [_snapshot_dict(s) for s in replay_euchre_match_game(chosen_game)]
    assert embedded_snapshots == expected


def _snapshot_dict(snapshot: object) -> object:
    """``asdict`` keeps tuples as tuples; the embedded data has already been through
    ``json.dumps``/``json.loads`` (tuples -> lists) by the time the test reads it
    back out of the HTML. Round-trip through JSON here too so the comparison is
    apples-to-apples instead of failing on ``[1, 2] != (1, 2)``."""
    from dataclasses import asdict

    return json.loads(json.dumps(asdict(snapshot)))  # type: ignore[call-overload]


def test_render_euchre_match_report_html_returns_the_same_content_without_filesystem() -> None:
    log = _recorded_log()

    html = report_euchre.render_euchre_match_report_html(log)

    match_data = _extract_json_block(html, "match-data")
    assert isinstance(match_data, dict)
    assert len(match_data["games"]) == len(log.games)


def test_report_summary_reflects_outcome_counts_and_win_rates() -> None:
    log = _recorded_log()
    expected_summary = summarize_euchre_match(log)

    html = report_euchre.render_euchre_match_report_html(log)
    embedded_summary = _extract_json_block(html, "match-summary")

    assert isinstance(embedded_summary, dict)
    assert embedded_summary["total_hands"] == expected_summary.total_hands
    assert embedded_summary["team_a_wins"] == expected_summary.team_a_wins
    assert embedded_summary["team_b_wins"] == expected_summary.team_b_wins
    assert embedded_summary["march_count"] == expected_summary.march_count
    assert embedded_summary["euchre_count"] == expected_summary.euchre_count

    # And rendered visibly in the summary section (not just embedded as JSON).
    assert f">{expected_summary.team_a_wins}<" in html
    assert f">{expected_summary.team_b_wins}<" in html


def test_report_is_self_contained_with_no_network_or_cdn_references() -> None:
    log = _recorded_log()

    html = report_euchre.render_euchre_match_report_html(log)

    assert "http://" not in html
    assert "https://" not in html
    assert "<script src" not in html
    assert "cdn" not in html.lower()


def test_report_god_view_default_embeds_all_four_hands_every_ply() -> None:
    """The report's data always contains all 4 hands (god view is a client-side
    display toggle, not a real information boundary -- see replay_euchre.py's
    module docstring); confirm the embedded data actually has that shape."""
    log = _recorded_log(num_hands=1)

    html = report_euchre.render_euchre_match_report_html(log)
    match_data = _extract_json_block(html, "match-data")
    assert isinstance(match_data, dict)

    for snapshot in match_data["games"][0]["snapshots"]:
        assert len(snapshot["hands"]) == 4


def test_render_euchre_match_report_html_renders_safely_for_an_empty_log() -> None:
    log = EuchreMatchLog(team_a="a", team_b="b", games=())

    html = report_euchre.render_euchre_match_report_html(log)

    assert html.strip() != ""
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "http://" not in html

    match_data = _extract_json_block(html, "match-data")
    match_summary = _extract_json_block(html, "match-summary")
    assert isinstance(match_data, dict)
    assert isinstance(match_summary, dict)
    assert match_data["games"] == []
    assert match_summary["total_hands"] == 0


def test_render_euchre_match_report_html_escapes_malicious_team_names_and_keeps_json_valid() -> (
    None
):
    malicious_a = "<b>alpha</b><script>alert(1)</script>"
    malicious_b = "beta</script><img src=x onerror=alert(2)>"
    log = EuchreMatchLog(team_a=malicious_a, team_b=malicious_b, games=())

    html = report_euchre.render_euchre_match_report_html(log)

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;b&gt;alpha&lt;/b&gt;" in html

    blocks = _ALL_JSON_SCRIPT_BLOCK_RE.findall(html)
    block_ids = {element_id for element_id, _body in blocks}
    assert block_ids == {"match-data", "match-summary"}
    for _element_id, body in blocks:
        payload = json.loads(body)
        assert isinstance(payload, dict)

    match_data = _extract_json_block(html, "match-data")
    match_summary = _extract_json_block(html, "match-summary")
    assert isinstance(match_data, dict)
    assert isinstance(match_summary, dict)
    assert match_data["team_a"] == malicious_a
    assert match_data["team_b"] == malicious_b
    assert match_summary["team_a"] == malicious_a
    assert match_summary["team_b"] == malicious_b


def test_report_cli_writes_report_for_an_on_disk_log(tmp_path: Path) -> None:
    log = _recorded_log(num_hands=2, seed=3)
    log_path = tmp_path / "match.zip"
    write_euchre_match_log(log_path, log)
    output_path = tmp_path / "report.html"

    report_euchre.main(["--log", str(log_path), "--output", str(output_path)])

    assert output_path.exists()
    html = output_path.read_text(encoding="utf-8")
    match_data = _extract_json_block(html, "match-data")
    assert isinstance(match_data, dict)
    assert len(match_data["games"]) == 2
