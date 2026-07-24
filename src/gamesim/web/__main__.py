"""Run the optional local browser UI with ``python -m gamesim.web``."""

from __future__ import annotations

import uvicorn

from .app import create_app


def main() -> None:
    """Start a local-only server for manual play."""
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
