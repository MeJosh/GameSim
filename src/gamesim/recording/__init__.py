"""Logging subsystem: record the engine's event stream, and replay it exactly.

Toggleable and event-sourced (see docs/adr/0006-deterministic-event-logging.md).
Named ``recording`` (not ``logging``) to avoid shadowing the standard library.
"""

from .euchre_match import record_euchre_match
from .euchre_match_log import (
    EuchreMatchGameLog,
    EuchreMatchLog,
    read_euchre_match_log,
    write_euchre_match_log,
)
from .match import record_match
from .match_log import MatchGameLog, MatchLog, read_match_log, write_match_log
from .recorder import EventCollector, JsonlRecorder, NullRecorder, Recorder

__all__ = [
    "Recorder",
    "NullRecorder",
    "JsonlRecorder",
    "EventCollector",
    "MatchGameLog",
    "MatchLog",
    "record_match",
    "read_match_log",
    "write_match_log",
    "EuchreMatchGameLog",
    "EuchreMatchLog",
    "record_euchre_match",
    "read_euchre_match_log",
    "write_euchre_match_log",
]
