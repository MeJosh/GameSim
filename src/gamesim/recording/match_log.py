"""Versioned, replayable records for a batch of Connect Four games.

Unlike the event-level JSONL recorder, a match log groups many complete games with
the logical agent labels needed to compare and browse a run. Individual actions
remain explicit, so each game is still replayed through the authoritative engine.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

MatchOutcome = Literal["agent_a", "agent_b", "draw"]
MATCH_LOG_FORMAT = "gamesim.connect_four.match/v1"
MATCH_ARCHIVE_FORMAT = "gamesim.connect_four.match-archive/v1"
_MAX_ARCHIVE_MEMBERS = 10_001
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class MatchGameLog:
    """One replayable game within a match log."""

    index: int
    seed: int
    seats: tuple[str, str]
    actions: tuple[tuple[int, int], ...]
    outcome: MatchOutcome

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "seed": self.seed,
            "seats": list(self.seats),
            "actions": [{"agent": agent, "action": action} for agent, action in self.actions],
            "outcome": self.outcome,
        }

    def to_manifest_entry(self) -> dict[str, object]:
        """Return the small manifest entry that points to this game's file."""
        return {
            "index": self.index,
            "path": _game_path(self.index),
            "seats": list(self.seats),
            "outcome": self.outcome,
            "total_moves": len(self.actions),
        }


@dataclass(frozen=True)
class MatchLog:
    """A complete, versioned batch record for two named logical agents."""

    agent_a: str
    agent_b: str
    games: tuple[MatchGameLog, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "format": MATCH_LOG_FORMAT,
            "agents": {"agent_a": self.agent_a, "agent_b": self.agent_b},
            "games": [game.to_dict() for game in self.games],
        }

    def to_archive_bytes(self) -> bytes:
        """Build the canonical ZIP artifact: manifest plus one file per game."""
        manifest = {
            "format": MATCH_ARCHIVE_FORMAT,
            "agents": {"agent_a": self.agent_a, "agent_b": self.agent_b},
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
    def from_dict(cls, record: Mapping[str, Any]) -> MatchLog:
        """Validate and deserialize a JSON-compatible match record."""
        if record.get("format") != MATCH_LOG_FORMAT:
            raise ValueError(f"unsupported match log format: {record.get('format')!r}")
        agents = record.get("agents")
        games = record.get("games")
        if not isinstance(agents, Mapping) or not isinstance(games, list):
            raise ValueError("match log must contain agents and games")
        agent_a = agents.get("agent_a")
        agent_b = agents.get("agent_b")
        if not isinstance(agent_a, str) or not isinstance(agent_b, str):
            raise ValueError("match log agent labels must be strings")

        parsed_games = tuple(_parse_game(game, index) for index, game in enumerate(games))
        return cls(agent_a=agent_a, agent_b=agent_b, games=parsed_games)

    @classmethod
    def from_archive_bytes(cls, payload: bytes) -> MatchLog:
        """Validate and deserialize a canonical ZIP match archive."""
        try:
            with ZipFile(BytesIO(payload)) as archive:
                infos = archive.infolist()
                if len(infos) > _MAX_ARCHIVE_MEMBERS:
                    raise ValueError("match archive contains too many files")
                if sum(info.file_size for info in infos) > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise ValueError("match archive is too large when extracted")
                manifest = _read_json_member(archive, "manifest.json")
                if manifest.get("format") != MATCH_ARCHIVE_FORMAT:
                    raise ValueError(
                        f"unsupported match archive format: {manifest.get('format')!r}"
                    )
                agents = manifest.get("agents")
                entries = manifest.get("games")
                if not isinstance(agents, Mapping) or not isinstance(entries, list):
                    raise ValueError("match archive must contain agents and games")
                agent_a = agents.get("agent_a")
                agent_b = agents.get("agent_b")
                if not isinstance(agent_a, str) or not isinstance(agent_b, str):
                    raise ValueError("match archive agent labels must be strings")
                games = tuple(
                    _read_archive_game(archive, entry, index) for index, entry in enumerate(entries)
                )
                return cls(agent_a=agent_a, agent_b=agent_b, games=games)
        except BadZipFile as error:
            raise ValueError("invalid ZIP match archive") from error


def _parse_game(record: object, expected_index: int) -> MatchGameLog:
    if not isinstance(record, Mapping):
        raise ValueError(f"game {expected_index} must be an object")
    index = record.get("index")
    seed = record.get("seed")
    seats = record.get("seats")
    actions = record.get("actions")
    outcome = record.get("outcome")
    if index != expected_index or not isinstance(seed, int):
        raise ValueError(f"game {expected_index} has an invalid index or seed")
    if (
        not isinstance(seats, list)
        or len(seats) != 2
        or not all(isinstance(seat, str) for seat in seats)
    ):
        raise ValueError(f"game {expected_index} must name exactly two seats")
    if outcome not in {"agent_a", "agent_b", "draw"}:
        raise ValueError(f"game {expected_index} has an invalid outcome")
    if not isinstance(actions, list):
        raise ValueError(f"game {expected_index} actions must be a list")

    parsed_actions: list[tuple[int, int]] = []
    for action_index, action in enumerate(actions):
        if not isinstance(action, Mapping):
            raise ValueError(f"game {expected_index} action {action_index} must be an object")
        agent = action.get("agent")
        column = action.get("action")
        if agent not in {0, 1} or not isinstance(column, int):
            raise ValueError(f"game {expected_index} action {action_index} is invalid")
        parsed_actions.append((agent, column))
    return MatchGameLog(
        index=index,
        seed=seed,
        seats=(seats[0], seats[1]),
        actions=tuple(parsed_actions),
        outcome=outcome,
    )


def _read_json_member(archive: ZipFile, path: str) -> Mapping[str, Any]:
    try:
        record = json.loads(archive.read(path))
    except KeyError as error:
        raise ValueError(f"match archive is missing {path}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"match archive has invalid JSON at {path}") from error
    if not isinstance(record, Mapping):
        raise ValueError(f"match archive member {path} must be an object")
    return record


def _read_archive_game(archive: ZipFile, entry: object, expected_index: int) -> MatchGameLog:
    if not isinstance(entry, Mapping):
        raise ValueError(f"manifest game {expected_index} must be an object")
    path = entry.get("path")
    if path != _game_path(expected_index):
        raise ValueError(f"manifest game {expected_index} has an invalid path")
    game = _parse_game(_read_json_member(archive, path), expected_index)
    if entry.get("seats") != list(game.seats) or entry.get("outcome") != game.outcome:
        raise ValueError(f"manifest game {expected_index} disagrees with its game file")
    if entry.get("total_moves") != len(game.actions):
        raise ValueError(f"manifest game {expected_index} has an invalid move count")
    return game


def _game_path(index: int) -> str:
    return f"games/{index:04d}.json"


def write_match_log(path: str | Path, log: MatchLog) -> Path:
    """Write the canonical ZIP match archive and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(log.to_archive_bytes())
    return output_path


def read_match_log(path: str | Path) -> MatchLog:
    """Read either the canonical ZIP archive or a legacy single-file JSON log."""
    input_path = Path(path)
    payload = input_path.read_bytes()
    if input_path.suffix.lower() == ".zip":
        return MatchLog.from_archive_bytes(payload)
    try:
        record = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid JSON match log") from error
    if not isinstance(record, Mapping):
        raise ValueError("match log must be an object")
    return MatchLog.from_dict(record)
