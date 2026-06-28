"""
driftwatch/__init__.py
──────────────────────
Public API for the DriftWatch library.

Minimal surface area by design:

    import anthropic
    import driftwatch

    client = driftwatch.wrap(
        anthropic.Anthropic(),
        goal="Summarise this codebase and identify dead code",
    )

    # Use exactly like anthropic.Anthropic():
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=messages,
    )

    # Inspect health history:
    for event in client.drift_history:
        print(event.turn, event.health_score)
"""
from __future__ import annotations

from driftwatch.engine import DriftEvent, SignalEngine
from driftwatch.wrapper import DriftWatchClient, wrap

__all__ = [
    "wrap",
    "DriftWatchClient",
    "DriftEvent",
    "SignalEngine",
]

__version__ = "0.1.0"
