"""
driftwatch/wrapper.py
─────────────────────
Transparent Anthropic SDK intercept layer.

DriftWatchClient wraps anthropic.Anthropic so the caller can continue using
the exact same API surface (client.messages.create) while DriftWatch silently
monitors memory health after every turn.

The returned response object is NEVER modified — DriftWatch is purely
observational except when on_drift="compact", where the messages list passed
by the caller is updated in place after compaction.

Public factory function:
    client = driftwatch.wrap(
        anthropic.Anthropic(),
        goal="...",
        threshold=0.55,
        on_drift="checkpoint",      # or "compact" | "alert" | callable
        checkpoint_dir="./dw_checkpoints",
        dashboard=True,
        max_context_tokens=200_000,
    )

    # Use exactly like the real Anthropic client:
    response = client.messages.create(model=..., messages=..., max_tokens=...)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

from driftwatch.checkpoint import CheckpointManager
from driftwatch.dashboard import DriftDashboard
from driftwatch.engine import DriftEvent, SignalEngine

# Lazy import of anthropic to avoid hard dependency at import time
# (allows signals/engine to be used without Anthropic credentials installed)
try:
    import anthropic as _anthropic
except ImportError:  # pragma: no cover
    _anthropic = None  # type: ignore[assignment]

import jsonlines as _jsonlines


# ---------------------------------------------------------------------------
# On-drift handler types
# ---------------------------------------------------------------------------

OnDriftLiteral = Literal["checkpoint", "compact", "alert", "none"]
OnDriftHandler = Union[OnDriftLiteral, Callable[["DriftWatchClient", DriftEvent], None]]


# ---------------------------------------------------------------------------
# Internal messages proxy — intercepts .create() calls
# ---------------------------------------------------------------------------

class _MessagesProxy:
    """
    Mimics the anthropic.resources.Messages API surface.

    Intercepts .create() to inject drift evaluation after each turn.
    """

    def __init__(self, owner: "DriftWatchClient") -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        """
        Pass the call through to the real Anthropic messages.create(),
        then evaluate drift and handle accordingly.

        All kwargs are forwarded verbatim — the response is returned unmodified.
        """
        owner = self._owner

        # ── 1. Forward to real Anthropic client ──────────────────────────
        response = owner._real_client.messages.create(**kwargs)

        # ── 2. Extract token count from response ─────────────────────────
        token_count = 0
        if hasattr(response, "usage") and response.usage is not None:
            token_count = getattr(response.usage, "input_tokens", 0)

        # ── 3. Build updated history for evaluation ───────────────────────
        # The messages list the caller passed in
        messages: list[dict] = kwargs.get("messages", [])

        # Append the assistant's response for evaluation purposes
        # (we build a temporary view — we do NOT mutate the caller's list here)
        assistant_content = []
        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "model_dump"):
                    assistant_content.append(block.model_dump())
                elif isinstance(block, dict):
                    assistant_content.append(block)
                else:
                    assistant_content.append({"type": "text", "text": str(block)})

        eval_history = list(messages) + [
            {"role": "assistant", "content": assistant_content}
        ]

        # ── 4. Evaluate drift ─────────────────────────────────────────────
        event = owner._engine.evaluate(eval_history, token_count=token_count)

        # ── 5. Log event to JSONL ─────────────────────────────────────────
        owner._log_event(event)

        # ── 6. Update dashboard ───────────────────────────────────────────
        if owner._dashboard is not None:
            owner._dashboard.update(event)

        # ── 7. Handle drift if threshold breached ────────────────────────
        if event.health_score < owner._threshold:
            owner._handle_drift(event, kwargs)

        # ── 8. Return original response unmodified ───────────────────────
        return response

    def stream(self, **kwargs: Any) -> Any:
        """Passthrough for streaming (no drift evaluation during stream)."""
        return self._owner._real_client.messages.stream(**kwargs)


# ---------------------------------------------------------------------------
# DriftWatchClient
# ---------------------------------------------------------------------------

class DriftWatchClient:
    """
    A transparent wrapper around anthropic.Anthropic that monitors drift.

    Do not instantiate directly — use driftwatch.wrap() instead.
    """

    def __init__(
        self,
        real_client: Any,
        goal: str,
        threshold: float = 0.55,
        on_drift: OnDriftHandler = "checkpoint",
        checkpoint_dir: str | os.PathLike = "./dw_checkpoints",
        dashboard: bool = True,
        max_context_tokens: int = 200_000,
        weights: Optional[dict[str, float]] = None,
        log_path: Optional[str | os.PathLike] = None,
    ) -> None:
        self._real_client = real_client
        self._threshold = threshold
        self._on_drift = on_drift
        self._checkpoint_dir = Path(checkpoint_dir)
        self._max_context_tokens = max_context_tokens

        # Engine
        self._engine = SignalEngine(goal=goal, threshold=threshold, weights=weights)

        # Checkpoint manager
        self._checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        # Dashboard (only in TTY environments when enabled)
        self._dashboard: Optional[DriftDashboard] = None
        if dashboard and sys.stdout.isatty():
            self._dashboard = DriftDashboard(
                goal=goal, max_tokens=max_context_tokens
            )
            self._dashboard.start()

        # JSONL event log
        if log_path is None:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            log_path = self._checkpoint_dir / "events.jsonl"
        self._log_path = Path(log_path)

        # Messages proxy (the intercept point)
        self.messages = _MessagesProxy(self)

    # ------------------------------------------------------------------
    # Public passthrough attributes / methods
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Forward any unknown attribute access to the real client."""
        return getattr(self._real_client, name)

    @property
    def engine(self) -> SignalEngine:
        """Access the underlying SignalEngine for the current session."""
        return self._engine

    @property
    def drift_history(self) -> list[DriftEvent]:
        """All DriftEvents recorded so far."""
        return self._engine.history

    def stop_dashboard(self) -> None:
        """Manually stop the Rich live dashboard."""
        if self._dashboard is not None:
            self._dashboard.stop()
            self._dashboard = None

    def save_checkpoint(
        self,
        messages: list[dict],
        metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Manually trigger a checkpoint save."""
        return self._checkpoint_manager.save(
            messages=messages,
            drift_events=self._engine.history,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_event(self, event: DriftEvent) -> None:
        """Append a DriftEvent to the JSONL log file."""
        try:
            with _jsonlines.open(self._log_path, mode="a") as writer:
                writer.write(event.model_dump(mode="json"))
        except Exception:
            pass  # Non-fatal — don't crash the caller's agent loop

    def _handle_drift(self, event: DriftEvent, create_kwargs: dict[str, Any]) -> None:
        """Dispatch the on_drift handler."""
        handler = self._on_drift

        if callable(handler):
            handler(self, event)
            return

        if handler == "alert":
            self._alert(event)

        elif handler == "checkpoint":
            messages = create_kwargs.get("messages", [])
            self._checkpoint_manager.save(
                messages=messages,
                drift_events=self._engine.history,
                metadata={"turn": event.turn, "health_score": event.health_score},
            )

        elif handler == "compact":
            messages: list[dict] = create_kwargs.get("messages", [])
            model: str = create_kwargs.get("model", "claude-sonnet-4-6")
            try:
                _, updated_messages = self._checkpoint_manager.save_with_compaction(
                    client=self._real_client,
                    messages=messages,
                    drift_events=self._engine.history,
                    model=model,
                    metadata={"turn": event.turn, "reason": "drift_threshold"},
                )
                # Update the caller's list in place
                messages[:] = updated_messages
            except Exception as exc:
                # Fall back to plain checkpoint if compaction fails
                self._alert(event, extra=f" (compaction failed: {exc})")
                self._checkpoint_manager.save(
                    messages=messages,
                    drift_events=self._engine.history,
                )

        elif handler == "none":
            pass  # silently swallow

    @staticmethod
    def _alert(event: DriftEvent, extra: str = "") -> None:
        """Print a drift alert to stderr."""
        print(
            f"[DriftWatch] ⚠  DRIFT DETECTED — Turn {event.turn} "
            f"health={event.health_score:.3f} "
            f"(gc={event.goal_coherence:.2f} "
            f"re={event.repetition_entropy:.2f} "
            f"md={event.memory_delta:.2f}){extra}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def wrap(
    client: Any,
    goal: str,
    threshold: float = 0.55,
    on_drift: OnDriftHandler = "checkpoint",
    checkpoint_dir: str | os.PathLike = "./dw_checkpoints",
    dashboard: bool = True,
    max_context_tokens: int = 200_000,
    weights: Optional[dict[str, float]] = None,
    log_path: Optional[str | os.PathLike] = None,
) -> DriftWatchClient:
    """
    Wrap an Anthropic SDK client with DriftWatch memory health monitoring.

    Args:
        client:             An ``anthropic.Anthropic()`` instance.
        goal:               Original task description — the semantic anchor for
                            the GoalCoherence signal.
        threshold:          Health score below which the on_drift handler fires.
                            Defaults to 0.55.
        on_drift:           What to do when health < threshold.
                            One of "checkpoint" | "compact" | "alert" | "none",
                            or a callable ``fn(client, event) -> None``.
        checkpoint_dir:     Directory for checkpoint and log files.
        dashboard:          If True (default), show the Rich live dashboard.
                            Automatically suppressed in non-TTY environments.
        max_context_tokens: Context window size for token fill percentage in
                            the dashboard.  Defaults to 200,000.
        weights:            Optional dict overriding signal weights.
        log_path:           Custom path for the JSONL event log.  Defaults to
                            ``{checkpoint_dir}/events.jsonl``.

    Returns:
        A DriftWatchClient that behaves identically to the wrapped Anthropic
        client but evaluates drift after every ``messages.create()`` call.

    Example::

        import anthropic
        import driftwatch

        client = driftwatch.wrap(
            anthropic.Anthropic(),
            goal="Summarise this codebase and identify dead code",
            threshold=0.55,
            on_drift="compact",
            dashboard=True,
        )

        # Normal Anthropic usage:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=messages,
        )
    """
    return DriftWatchClient(
        real_client=client,
        goal=goal,
        threshold=threshold,
        on_drift=on_drift,
        checkpoint_dir=checkpoint_dir,
        dashboard=dashboard,
        max_context_tokens=max_context_tokens,
        weights=weights,
        log_path=log_path,
    )
