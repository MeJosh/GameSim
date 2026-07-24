"""Versioned, replayable records for a batch of recorded Euchre hands.

Mirrors ``match_log.py``'s structure and robustness (versioned format tags, archive
size/member limits, strict validation on load) but shaped for Euchre: 4 named seats
rather than 2, a ``team_a``/``team_b`` outcome (there is no draw -- some team always
scores a hand -- see ``games.euchre.engine.EuchreEngine._score_hand``), and enough
per-hand metadata (``maker_team``, ``alone``, ``points``, ``trump``, ``dealer``,
``stick_the_dealer``) that summary stats never need to re-replay a hand (see
``analysis.summary_euchre``) and replay can reconstruct the exact engine
configuration (see ``analysis.replay_euchre``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, TypeGuard
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

from gamesim.games.euchre.actions import NUM_ACTIONS

EuchreMatchOutcome = Literal["team_a", "team_b"]
EUCHRE_MATCH_LOG_FORMAT = "gamesim.euchre.match/v1"
EUCHRE_MATCH_ARCHIVE_FORMAT = "gamesim.euchre.match-archive/v1"
_MAX_ARCHIVE_MEMBERS = 10_001
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
_VALID_OUTCOMES = {"team_a", "team_b"}


def _is_int(value: object) -> TypeGuard[int]:
    """``isinstance(x, int)`` alone also accepts ``bool`` (a subclass of ``int`` in
    Python), so a malformed archive with e.g. ``"dealer": true`` would silently
    parse as ``dealer=1`` -- reject bools explicitly everywhere an integer field is
    validated below. Typed as a ``TypeGuard`` so the ``if not _is_int(x): raise``
    pattern below actually narrows ``x`` to ``int`` afterward, same as the inline
    ``isinstance`` checks it replaced."""
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class EuchreMatchGameLog:
    """One replayable hand within a Euchre match log.

    ``seats[i]`` is the logical agent name occupying seat ``i`` (0-3) for this hand;
    ``record_euchre_match`` alternates which physical seats ``team_a``/``team_b``
    occupy across hands, so ``seats`` is stored per-hand rather than once per match.
    """

    index: int
    seed: int
    dealer: int
    stick_the_dealer: bool
    seats: tuple[str, str, str, str]
    actions: tuple[tuple[int, int], ...]
    outcome: EuchreMatchOutcome
    points: int
    maker_team: EuchreMatchOutcome
    trump: int
    alone: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "seed": self.seed,
            "dealer": self.dealer,
            "stick_the_dealer": self.stick_the_dealer,
            "seats": list(self.seats),
            "actions": [{"agent": agent, "action": action} for agent, action in self.actions],
            "outcome": self.outcome,
            "points": self.points,
            "maker_team": self.maker_team,
            "trump": self.trump,
            "alone": self.alone,
        }

    def to_manifest_entry(self) -> dict[str, object]:
        """Return the small manifest entry that points to this hand's file."""
        return {
            "index": self.index,
            "path": _game_path(self.index),
            "seats": list(self.seats),
            "outcome": self.outcome,
            "points": self.points,
            "total_actions": len(self.actions),
        }


@dataclass(frozen=True)
class EuchreMatchLog:
    """A complete, versioned batch record for two named logical teams."""

    team_a: str
    team_b: str
    games: tuple[EuchreMatchGameLog, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "format": EUCHRE_MATCH_LOG_FORMAT,
            "teams": {"team_a": self.team_a, "team_b": self.team_b},
            "games": [game.to_dict() for game in self.games],
        }

    def to_archive_bytes(self) -> bytes:
        """Build the canonical ZIP artifact: manifest plus one file per hand."""
        manifest = {
            "format": EUCHRE_MATCH_ARCHIVE_FORMAT,
            "teams": {"team_a": self.team_a, "team_b": self.team_b},
            "games": [game.to_manifest_entry() for game in self.games],
        }
        buffer = BytesIO()
        with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
            for game in self.games:
                archive.writestr(
                    _game_path(game.index), json.dumps(game.to_dict(), indent=2) + "\n"
                )
        return buffer.getvalue()

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> EuchreMatchLog:
        """Validate and deserialize a JSON-compatible match record."""
        if record.get("format") != EUCHRE_MATCH_LOG_FORMAT:
            raise ValueError(f"unsupported euchre match log format: {record.get('format')!r}")
        teams = record.get("teams")
        games = record.get("games")
        if not isinstance(teams, Mapping) or not isinstance(games, list):
            raise ValueError("euchre match log must contain teams and games")
        team_a = teams.get("team_a")
        team_b = teams.get("team_b")
        if not isinstance(team_a, str) or not isinstance(team_b, str):
            raise ValueError("euchre match log team labels must be strings")

        parsed_games = tuple(_parse_game(game, index) for index, game in enumerate(games))
        return cls(team_a=team_a, team_b=team_b, games=parsed_games)

    @classmethod
    def from_archive_bytes(cls, payload: bytes) -> EuchreMatchLog:
        """Validate and deserialize a canonical ZIP match archive."""
        try:
            with ZipFile(BytesIO(payload)) as archive:
                infos = archive.infolist()
                if len(infos) > _MAX_ARCHIVE_MEMBERS:
                    raise ValueError("euchre match archive contains too many files")
                if sum(info.file_size for info in infos) > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise ValueError("euchre match archive is too large when extracted")
                manifest = _read_json_member(archive, "manifest.json")
                if manifest.get("format") != EUCHRE_MATCH_ARCHIVE_FORMAT:
                    raise ValueError(
                        f"unsupported euchre match archive format: {manifest.get('format')!r}"
                    )
                teams = manifest.get("teams")
                entries = manifest.get("games")
                if not isinstance(teams, Mapping) or not isinstance(entries, list):
                    raise ValueError("euchre match archive must contain teams and games")
                team_a = teams.get("team_a")
                team_b = teams.get("team_b")
                if not isinstance(team_a, str) or not isinstance(team_b, str):
                    raise ValueError("euchre match archive team labels must be strings")
                games = tuple(
                    _read_archive_game(archive, entry, index) for index, entry in enumerate(entries)
                )
                return cls(team_a=team_a, team_b=team_b, games=games)
        except BadZipFile as error:
            raise ValueError("invalid ZIP euchre match archive") from error


def _parse_game(record: object, expected_index: int) -> EuchreMatchGameLog:
    if not isinstance(record, Mapping):
        raise ValueError(f"hand {expected_index} must be an object")
    index = record.get("index")
    seed = record.get("seed")
    dealer = record.get("dealer")
    stick_the_dealer = record.get("stick_the_dealer")
    seats = record.get("seats")
    actions = record.get("actions")
    outcome = record.get("outcome")
    points = record.get("points")
    maker_team = record.get("maker_team")
    trump = record.get("trump")
    alone = record.get("alone")

    if not _is_int(index) or index != expected_index or not _is_int(seed):
        raise ValueError(f"hand {expected_index} has an invalid index or seed")
    if not _is_int(dealer) or not (0 <= dealer < 4):
        raise ValueError(f"hand {expected_index} has an invalid dealer")
    if not isinstance(stick_the_dealer, bool):
        raise ValueError(f"hand {expected_index} has an invalid stick_the_dealer flag")
    if (
        not isinstance(seats, list)
        or len(seats) != 4
        or not all(isinstance(seat, str) for seat in seats)
    ):
        raise ValueError(f"hand {expected_index} must name exactly four seats")
    if outcome not in _VALID_OUTCOMES:
        raise ValueError(f"hand {expected_index} has an invalid outcome")
    if maker_team not in _VALID_OUTCOMES:
        raise ValueError(f"hand {expected_index} has an invalid maker_team")
    if not _is_int(points) or points not in (1, 2, 4):
        raise ValueError(f"hand {expected_index} has an invalid points value")
    if not _is_int(trump) or not (0 <= trump < 4):
        raise ValueError(f"hand {expected_index} has an invalid trump suit")
    if not isinstance(alone, bool):
        raise ValueError(f"hand {expected_index} has an invalid alone flag")
    if not isinstance(actions, list):
        raise ValueError(f"hand {expected_index} actions must be a list")

    parsed_actions: list[tuple[int, int]] = []
    for action_index, action in enumerate(actions):
        if not isinstance(action, Mapping):
            raise ValueError(f"hand {expected_index} action {action_index} must be an object")
        agent = action.get("agent")
        value = action.get("action")
        if (
            not _is_int(agent)
            or agent not in {0, 1, 2, 3}
            or not _is_int(value)
            or not (0 <= value < NUM_ACTIONS)
        ):
            raise ValueError(f"hand {expected_index} action {action_index} is invalid")
        parsed_actions.append((agent, value))

    return EuchreMatchGameLog(
        index=index,
        seed=seed,
        dealer=dealer,
        stick_the_dealer=stick_the_dealer,
        seats=(seats[0], seats[1], seats[2], seats[3]),
        actions=tuple(parsed_actions),
        outcome=outcome,
        points=points,
        maker_team=maker_team,
        trump=trump,
        alone=alone,
    )


def _read_json_member(archive: ZipFile, path: str) -> Mapping[str, Any]:
    try:
        record = json.loads(archive.read(path))
    except KeyError as error:
        raise ValueError(f"euchre match archive is missing {path}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"euchre match archive has invalid JSON at {path}") from error
    if not isinstance(record, Mapping):
        raise ValueError(f"euchre match archive member {path} must be an object")
    return record


def _read_archive_game(archive: ZipFile, entry: object, expected_index: int) -> EuchreMatchGameLog:
    if not isinstance(entry, Mapping):
        raise ValueError(f"manifest hand {expected_index} must be an object")
    path = entry.get("path")
    if path != _game_path(expected_index):
        raise ValueError(f"manifest hand {expected_index} has an invalid path")
    game = _parse_game(_read_json_member(archive, path), expected_index)
    if entry.get("seats") != list(game.seats) or entry.get("outcome") != game.outcome:
        raise ValueError(f"manifest hand {expected_index} disagrees with its hand file")
    if entry.get("points") != game.points:
        raise ValueError(f"manifest hand {expected_index} disagrees on points")
    if entry.get("total_actions") != len(game.actions):
        raise ValueError(f"manifest hand {expected_index} has an invalid action count")
    return game


def _game_path(index: int) -> str:
    return f"games/{index:04d}.json"


def write_euchre_match_log(path: str | Path, log: EuchreMatchLog) -> Path:
    """Write the canonical ZIP match archive and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(log.to_archive_bytes())
    return output_path


def read_euchre_match_log(path: str | Path) -> EuchreMatchLog:
    """Read either the canonical ZIP archive or a legacy single-file JSON log."""
    input_path = Path(path)
    payload = input_path.read_bytes()
    if input_path.suffix.lower() == ".zip":
        return EuchreMatchLog.from_archive_bytes(payload)
    try:
        record = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid JSON euchre match log") from error
    if not isinstance(record, Mapping):
        raise ValueError("euchre match log must be an object")
    return EuchreMatchLog.from_dict(record)
