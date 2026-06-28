"""
tests/conftest.py
──────────────────
Pytest configuration for DriftWatch tests.
"""
from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers to suppress PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers",
        "requires_api_key: mark test as requiring a real ANTHROPIC_API_KEY",
    )
