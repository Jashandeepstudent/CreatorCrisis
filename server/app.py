"""
server/app.py — OpenEnv entry point
CreatorCrisisEnv | Meta OpenEnv Hackathon
Author: Jashandeep Singh

This module satisfies the `openenv validate` requirement for a
`server/app.py` entry point with a `main()` function.

It delegates entirely to the existing FastAPI application in server/api.py.
The [project.scripts] entry in pyproject.toml points here:
    server = "server.app:main"
"""

from __future__ import annotations

import os
import sys

# ── Path bootstrap ────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in [_HERE, _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from server.api import app  # noqa: F401  — re-exported for uvicorn


def main() -> None:
    """
    Launch the CreatorCrisisEnv FastAPI server.

    Entry point for:
        uv run server
        python -m server.app
        uvicorn server.app:app --host 0.0.0.0 --port 7860
    """
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))

    uvicorn.run(
        "server.app:app",
        host      = host,
        port      = port,
        log_level = "info",
        reload    = False,
    )


if __name__ == "__main__":
    main()
