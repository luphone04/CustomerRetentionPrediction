"""Deprecated entrypoint kept for backward compatibility.

The supported app is src/app/main.py (`uvicorn src.app.main:app`), which the
Dockerfile runs. This module re-exports it so `uvicorn src.app.app:app` serves
the same API and UI instead of a second, divergent implementation.
"""

from src.app.main import app  # noqa: F401
