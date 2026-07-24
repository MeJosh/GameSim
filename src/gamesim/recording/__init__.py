"""Logging subsystem: record the engine's event stream, and replay it exactly.

Toggleable and event-sourced (see docs/adr/0006-deterministic-event-logging.md).
Named ``recording`` (not ``logging``) to avoid shadowing the standard library.
"""

from .recorder import JsonlRecorder, NullRecorder, Recorder

__all__ = ["Recorder", "NullRecorder", "JsonlRecorder"]
