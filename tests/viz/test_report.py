"""Tests for the standalone HTML match report -- plan Slice 3b, tests 1-4."""

from __future__ import annotations

import json
import re
from pathlib import Path

from gamesim.analysis import replay_match_game, summarize_match
from gamesim.core.agent import RandomAgent
from gamesim.games.connect_four.engine import ConnectFourObservation
from gamesim.recording import record_match, write_match_log
from gamesim.recording.match_log import MatchGameLog, MatchLog, MatchOutcome
from gamesim.viz import report

_ALL_JSON_SCRIPT_BLOCK_RE = re.compile(
    r'<script type="application/json" id="([^"]+)">(.*?)</script>',
    re.DOTALL,
)


def _game(
    index: int, seats: tuple[str, str], moves: list[int], outcome: MatchOutcome
) -> MatchGameLog:
    """Build a MatchGameLog whose actions alternate agent 0/1, starting with 0."""
    actions = tuple((move_index % 2, column) for move_index, column in enumerate(moves))
    return MatchGameLog(index=index, seed=index, seats=seats, actions=actions, outcome=outcome)


def _hand_built_log() -> MatchLog:
    # Same fixture shape as tests/analysis/test_summary.py, so the summary numbers
    # embedded in the report can be cross-checked against known values.
    return MatchLog(
        agent_a="alpha",
        agent_b="beta",
        games=(
            _game(0, ("alpha", "beta"), [3, 2, 3], "agent_a"),
            _game(1, ("beta", "alpha"), [0, 1, 0, 1, 0], "agent_b"),
            _game(2, ("alpha", "beta"), [4, 5, 4, 5], "agent_b"),
            _game(3, ("beta", "alpha"), [6, 6], "draw"),
        ),
    )


def _extract_json_block(html: str, element_id: str) -> object:
    pattern = re.compile(
        r'<script type="application/json" id="' + re.escape(element_id) + r'">(.*?)</script>',
        re.DOTALL,
    )
    match = pattern.search(html)
    assert match is not None, f"missing #{element_id} JSON script block"
    return json.loads(match.group(1))


def test_write_match_report_embeds_all_games_and_matches_engine_replay(tmp_path: Path) -> None:
    log = _hand_built_log()
    output = tmp_path / "report.html"

    result_path = report.write_match_report(log, output)

    assert result_path == output
    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert html.strip() != ""
    assert html.lstrip().startswith("<!DOCTYPE html>")

    match_data = _extract_json_block(html, "match-data")
    assert isinstance(match_data, dict)
    assert len(match_data["games"]) == len(log.games)

    chosen_game = log.games[2]
    embedded_boards = match_data["games"][2]["boards"]
    assert embedded_boards == replay_match_game(chosen_game)


def test_render_match_report_html_returns_the_same_content_without_filesystem() -> None:
    log = _hand_built_log()

    html = report.render_match_report_html(log)

    match_data = _extract_json_block(html, "match-data")
    assert isinstance(match_data, dict)
    assert len(match_data["games"]) == len(log.games)
    assert match_data["games"][0]["boards"] == replay_match_game(log.games[0])


def test_report_summary_reflects_outcome_counts_and_win_rates() -> None:
    log = _hand_built_log()
    expected_summary = summarize_match(log)

    html = report.render_match_report_html(log)
    embedded_summary = _extract_json_block(html, "match-summary")

    assert isinstance(embedded_summary, dict)
    assert embedded_summary["total_games"] == expected_summary.total_games
    assert embedded_summary["agent_a_wins"] == expected_summary.agent_a_wins
    assert embedded_summary["agent_b_wins"] == expected_summary.agent_b_wins
    assert embedded_summary["draws"] == expected_summary.draws
    assert embedded_summary["agent_a_win_rate"] == expected_summary.agent_a_win_rate
    assert embedded_summary["agent_b_win_rate"] == expected_summary.agent_b_win_rate
    assert embedded_summary["draw_rate"] == expected_summary.draw_rate
    assert [tuple(pair) for pair in embedded_summary["game_length_histogram"]] == list(
        expected_summary.game_length_histogram
    )
    assert [tuple(pair) for pair in embedded_summary["opening_move_distribution"]] == list(
        expected_summary.opening_move_distribution
    )

    # And rendered visibly in the summary section (not just embedded as JSON).
    assert f">{expected_summary.agent_a_wins}<" in html
    assert f">{expected_summary.agent_b_wins}<" in html
    assert f">{expected_summary.draws}<" in html


def test_report_is_self_contained_with_no_network_or_cdn_references() -> None:
    log = _hand_built_log()

    html = report.render_match_report_html(log)

    assert "http://" not in html
    assert "https://" not in html
    assert "<script src" not in html
    assert "cdn" not in html.lower()


def test_all_games_boards_match_engine_replay_including_flipped_seats_and_draw() -> None:
    log = _hand_built_log()

    # Sanity: the fixture actually covers a seat-flipped game and a draw, so this
    # test isn't accidentally only exercising index 0/2 like the pre-existing tests.
    assert any(game.seats[0] != log.agent_a for game in log.games)
    assert any(game.outcome == "draw" for game in log.games)

    html = report.render_match_report_html(log)
    match_data = _extract_json_block(html, "match-data")
    assert isinstance(match_data, dict)
    embedded_games = match_data["games"]
    assert len(embedded_games) == len(log.games)

    for game, embedded_game in zip(log.games, embedded_games, strict=True):
        expected_boards = replay_match_game(game)
        embedded_boards = embedded_game["boards"]

        assert embedded_boards == expected_boards
        assert len(embedded_boards) == len(game.actions) + 1
        assert all(cell == 0 for row in embedded_boards[0] for cell in row)


def test_render_match_report_html_renders_safely_for_an_empty_log() -> None:
    log = MatchLog(agent_a="a", agent_b="b", games=())

    html = report.render_match_report_html(log)

    assert html.strip() != ""
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "http://" not in html
    assert "https://" not in html

    match_data = _extract_json_block(html, "match-data")
    match_summary = _extract_json_block(html, "match-summary")
    assert isinstance(match_data, dict)
    assert isinstance(match_summary, dict)
    assert match_data["games"] == []
    assert match_summary["total_games"] == 0


def test_render_match_report_html_escapes_malicious_agent_names_and_keeps_json_valid() -> None:
    malicious_a = "<b>alpha</b><script>alert(1)</script>"
    malicious_b = "beta</script><img src=x onerror=alert(2)>"
    log = MatchLog(agent_a=malicious_a, agent_b=malicious_b, games=())

    html = report.render_match_report_html(log)

    # (a) visible HTML must show html-escaped agent names, never a raw <script> tag.
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;b&gt;alpha&lt;/b&gt;" in html

    # (b) both embedded JSON script blocks must still be well-formed / parseable --
    # i.e. the "</script>" inside the malicious agent name did not prematurely
    # close the <script type="application/json"> tag it lives in.
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
    assert match_data["agent_a"] == malicious_a
    assert match_data["agent_b"] == malicious_b
    assert match_summary["agent_a"] == malicious_a
    assert match_summary["agent_b"] == malicious_b


def test_report_cli_writes_report_for_an_on_disk_log(tmp_path: Path) -> None:
    match = record_match(
        RandomAgent[ConnectFourObservation](seed=1),
        RandomAgent[ConnectFourObservation](seed=2),
        agent_a_name="a",
        agent_b_name="b",
        num_games=2,
        seed=3,
    )
    log_path = tmp_path / "match.zip"
    write_match_log(log_path, match)
    output_path = tmp_path / "report.html"

    report.main(["--log", str(log_path), "--output", str(output_path)])

    assert output_path.exists()
    html = output_path.read_text(encoding="utf-8")
    match_data = _extract_json_block(html, "match-data")
    assert isinstance(match_data, dict)
    assert len(match_data["games"]) == 2
