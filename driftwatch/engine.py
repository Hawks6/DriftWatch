"""
driftwatch/engine.py
────────────────────
Composite scorer and DriftEvent emitter.

Orchestrates the three drift signals from signals.py, produces a weighted
health score, and emits structured DriftEvent records.

health_score = 0.50 * goal_coherence
             + 0.30 * repetition_entropy
             + 0.20 * memory_delta

Weights are configurable at SignalEngine initialisation time.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from driftwatch.signals import (
    GoalCoherenceSignal,
    MemoryDeltaSignal,
    RepetitionEntropySignal,
)

# ---------------------------------------------------------------------------
# Default signal weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "goal_coherence": 0.50,
    "repetition_entropy": 0.30,
    "memory_delta": 0.20,
}


# ---------------------------------------------------------------------------
# DriftEvent — structured record of one evaluation turn
# ---------------------------------------------------------------------------

class DriftEvent(BaseModel):
    """
    Immutable record of one health evaluation turn.

    All float fields are in [0.0, 1.0] where 1.0 = fully healthy.
    """

    turn: int = Field(..., description="Monotonically increasing turn counter (1-based)")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp of the evaluation",
    )
    goal_coherence: float = Field(
        ..., ge=0.0, le=1.0, description="GoalCoherenceSignal score"
    )
    repetition_entropy: float = Field(
        ..., ge=0.0, le=1.0, description="RepetitionEntropySignal score"
    )
    memory_delta: float = Field(
        ..., ge=0.0, le=1.0, description="MemoryDeltaSignal score"
    )
    health_score: float = Field(
        ..., ge=0.0, le=1.0, description="Weighted composite health score"
    )
    token_count: int = Field(
        ..., ge=0, description="input_tokens from the API response usage field"
    )
    triggered_checkpoint: bool = Field(
        default=False,
        description="True if this event triggered a checkpoint or compaction",
    )
    notes: str = Field(
        default="",
        description="Free-text annotation (e.g. 'compacted', 'threshold breached')",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# SignalEngine — orchestrates signals and maintains event history
# ---------------------------------------------------------------------------

class SignalEngine:
    """
    Drives the three drift signals and produces DriftEvent records.

    Args:
        goal:      The original task description (used to anchor GoalCoherence).
        threshold: Health score below which the caller should take action.
                   Defaults to 0.55.
        weights:   Optional dict overriding signal weights.  Keys must be a
                   subset of {"goal_coherence", "repetition_entropy",
                   "memory_delta"}.  Values must sum to 1.0.
    """

    def __init__(
        self,
        goal: str,
        threshold: float = 0.55,
        weights: Optional[dict[str, float]] = None,
    ) -> None:
        self.goal = goal
        self.threshold = threshold
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self._validate_weights()

        # Initialise signal classes
        self._gc_signal = GoalCoherenceSignal(goal=goal)
        self._re_signal = RepetitionEntropySignal(window=10)
        self._md_signal = MemoryDeltaSignal(window=5)

        self._turn_counter: int = 0
        self.history: list[DriftEvent] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        history: list[dict],
        token_count: int,
        *,
        triggered_checkpoint: bool = False,
        notes: str = "",
    ) -> DriftEvent:
        """
        Compute all three signals, assemble the composite health score, and
        append a DriftEvent to self.history.

        Args:
            history:              The full message history at this turn.
            token_count:          input_tokens reported by the API response.
            triggered_checkpoint: Pass True if a checkpoint was triggered.
            notes:                Optional free-text annotation.

        Returns:
            The new DriftEvent (also appended to self.history).
        """
        self._turn_counter += 1

        gc = self._gc_signal.score(history)
        re = self._re_signal.score(history)
        md = self._md_signal.score(history)

        health = (
            self.weights["goal_coherence"] * gc
            + self.weights["repetition_entropy"] * re
            + self.weights["memory_delta"] * md
        )
        # Guard against tiny floating point overshoots
        health = float(max(0.0, min(1.0, health)))

        event = DriftEvent(
            turn=self._turn_counter,
            goal_coherence=round(gc, 4),
            repetition_entropy=round(re, 4),
            memory_delta=round(md, 4),
            health_score=round(health, 4),
            token_count=token_count,
            triggered_checkpoint=triggered_checkpoint,
            notes=notes,
        )
        self.history.append(event)
        return event

    @property
    def is_drifting(self) -> bool:
        """True if the most recent health score is below the threshold."""
        if not self.history:
            return False
        return self.history[-1].health_score < self.threshold

    @property
    def current_turn(self) -> int:
        return self._turn_counter

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_weights(self) -> None:
        required = {"goal_coherence", "repetition_entropy", "memory_delta"}
        unknown = set(self.weights.keys()) - required
        if unknown:
            raise ValueError(f"Unknown weight keys: {unknown}")
        total = sum(self.weights.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Signal weights must sum to 1.0 (got {total:.3f}).  "
                f"Current weights: {self.weights}"
            )
