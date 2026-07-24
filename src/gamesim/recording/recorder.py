"""Recorders consume the engine's event stream and (optionally) persist it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from gamesim.core.events import Event


@runtime_checkable
class Recorder(Protocol):
    """Consumes engine events. Implementations decide whether to persist them."""

    def record(self, event: Event) -> None: ...

    def close(self) -> None: ...


class NullRecorder:
    """Discards everything. The default during bulk training — zero overhead."""

    def record(self, event: Event) -> None:  # noqa: D102
        return None

    def close(self) -> None:  # noqa: D102
        return None


class JsonlRecorder:
    """Appends one JSON object per event to a ``.jsonl`` file, in order.

    The resulting file, together with the game seed captured in the opening
    ``GameStarted`` event, is a complete, replayable log.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", encoding="utf-8")

    def record(self, event: Event) -> None:
        self._fh.write(json.dumps(event.to_dict()) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> JsonlRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
