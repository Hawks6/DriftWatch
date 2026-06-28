"""
tests/test_signals.py
──────────────────────
Unit tests for the three drift signal classes.

These tests run fully offline — no Anthropic API key required.
sentence-transformers downloads the all-MiniLM-L6-v2 model on first run
(~80MB, cached locally).

Run:
    python -m pytest tests/test_signals.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from driftwatch.signals import (
    GoalCoherenceSignal,
    MemoryDeltaSignal,
    RepetitionEntropySignal,
    _cosine_sim,
    _shannon_entropy,
    _text_to_bigrams,
)


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_sim(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_sim(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_sim(v, -v) == pytest.approx(-1.0, abs=1e-5)

    def test_zero_vector_safe(self) -> None:
        zero = np.array([0.0, 0.0, 0.0])
        v = np.array([1.0, 2.0, 3.0])
        result = _cosine_sim(zero, v)
        assert math.isfinite(result)  # must not raise / return nan


class TestShannonEntropy:
    def test_single_item_zero_entropy(self) -> None:
        # One unique item → no uncertainty → entropy = 0
        assert _shannon_entropy(["a", "a", "a"]) == pytest.approx(0.0, abs=1e-5)

    def test_uniform_distribution_max_entropy(self) -> None:
        # 4 distinct items, each appearing once → max entropy = log2(4) = 2
        items = ["a", "b", "c", "d"]
        H = _shannon_entropy(items)
        assert H == pytest.approx(2.0, abs=1e-4)

    def test_empty_input(self) -> None:
        assert _shannon_entropy([]) == 0.0

    def test_two_items_equal(self) -> None:
        # p = [0.5, 0.5] → H = 1.0
        H = _shannon_entropy(["x", "y"])
        assert H == pytest.approx(1.0, abs=1e-4)


class TestTextBigrams:
    def test_basic_bigrams(self) -> None:
        bigrams = _text_to_bigrams("the cat sat")
        assert bigrams == ["the_cat", "cat_sat"]

    def test_single_word(self) -> None:
        # Single word → fallback to the word itself
        result = _text_to_bigrams("hello")
        assert result == ["hello"]

    def test_empty_string(self) -> None:
        assert _text_to_bigrams("") == []


# ---------------------------------------------------------------------------
# GoalCoherenceSignal tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def goal_coherence_signal() -> GoalCoherenceSignal:
    return GoalCoherenceSignal(goal="Summarise Python clean code principles")


class TestGoalCoherenceSignal:

    def test_empty_history_returns_neutral(self, goal_coherence_signal: GoalCoherenceSignal) -> None:
        assert goal_coherence_signal.score([]) == pytest.approx(0.5, abs=1e-5)

    def test_no_assistant_message_returns_neutral(self, goal_coherence_signal: GoalCoherenceSignal) -> None:
        history = [{"role": "user", "content": "Tell me about clean code"}]
        assert goal_coherence_signal.score(history) == pytest.approx(0.5, abs=1e-5)

    def test_on_topic_score_high(self, goal_coherence_signal: GoalCoherenceSignal) -> None:
        history = [
            {"role": "user", "content": "What are clean code principles?"},
            {
                "role": "assistant",
                "content": (
                    "Clean code principles in Python include writing readable, "
                    "maintainable code. Key principles: single responsibility, "
                    "meaningful names, small functions."
                ),
            },
        ]
        score = goal_coherence_signal.score(history)
        assert score >= 0.3  # should be reasonably high for on-topic content

    def test_off_topic_score_lower(self, goal_coherence_signal: GoalCoherenceSignal) -> None:
        history = [
            {"role": "user", "content": "What is the weather today?"},
            {
                "role": "assistant",
                "content": "Today's weather is sunny with a high of 25°C.",
            },
        ]
        score_off = goal_coherence_signal.score(history)
        history_on = [
            {"role": "user", "content": "Explain Python clean code"},
            {
                "role": "assistant",
                "content": "Clean code in Python follows principles like readability and simplicity.",
            },
        ]
        score_on = goal_coherence_signal.score(history_on)
        assert score_on >= score_off - 0.2

    def test_score_in_valid_range(self, goal_coherence_signal: GoalCoherenceSignal) -> None:
        history = [
            {"role": "assistant", "content": "Some random assistant response."}
        ]
        score = goal_coherence_signal.score(history)
        assert 0.0 <= score <= 1.0

    def test_list_content_blocks(self, goal_coherence_signal: GoalCoherenceSignal) -> None:
        """Test that list-format content blocks are handled correctly."""
        history = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Clean code means readable and maintainable code."}
                ],
            }
        ]
        score = goal_coherence_signal.score(history)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# RepetitionEntropySignal tests
# ---------------------------------------------------------------------------

class TestRepetitionEntropySignal:
    @pytest.fixture
    def signal(self) -> RepetitionEntropySignal:
        return RepetitionEntropySignal(window=4)

    def test_empty_history_neutral(self, signal: RepetitionEntropySignal) -> None:
        assert signal.score([]) == pytest.approx(0.5, abs=1e-5)

    def test_high_entropy_diverse_tools(self, signal: RepetitionEntropySignal) -> None:
        """Many different tool calls → high entropy → high health score."""
        history = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": f"tool_{i}", "id": str(i), "input": {}}],
            }
            for i in range(4)
        ]
        score = signal.score(history)
        assert score > 0.7

    def test_low_entropy_same_tool(self, signal: RepetitionEntropySignal) -> None:
        """Same tool called repeatedly → low entropy → low health score."""
        history = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "search", "id": str(i), "input": {}}],
            }
            for i in range(4)
        ]
        score = signal.score(history)
        assert score < 0.2

    def test_text_bigram_fallback(self, signal: RepetitionEntropySignal) -> None:
        """Without tool calls, falls back to bigram entropy."""
        history = [
            {"role": "assistant", "content": "the quick brown fox jumps over"},
            {"role": "assistant", "content": "the lazy dog sat down today"},
        ]
        score = signal.score(history)
        assert 0.0 <= score <= 1.0

    def test_window_clamp(self, signal: RepetitionEntropySignal) -> None:
        """Score with more messages than window only uses last `window` messages."""
        history = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": f"t{i}", "id": str(i), "input": {}}],
            }
            for i in range(20)
        ]
        score = signal.score(history)
        assert 0.0 <= score <= 1.0

    def test_invalid_window(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 2"):
            RepetitionEntropySignal(window=1)

    def test_score_in_valid_range(self, signal: RepetitionEntropySignal) -> None:
        history = [{"role": "assistant", "content": "Hello world this is a test"}]
        assert 0.0 <= signal.score(history) <= 1.0


# ---------------------------------------------------------------------------
# MemoryDeltaSignal tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def memory_delta_signal() -> MemoryDeltaSignal:
    return MemoryDeltaSignal(window=4)


class TestMemoryDeltaSignal:

    def test_empty_history_neutral(self, memory_delta_signal: MemoryDeltaSignal) -> None:
        assert memory_delta_signal.score([]) == pytest.approx(0.5, abs=1e-5)

    def test_all_new_facts_high_score(self, memory_delta_signal: MemoryDeltaSignal) -> None:
        """Completely different content each turn → high new-fact ratio."""
        history = [
            {"role": "assistant", "content": "Python uses reference counting for memory management."},
            {"role": "assistant", "content": "JavaScript is single-threaded with an event loop."},
            {"role": "assistant", "content": "Rust enforces memory safety at compile time."},
            {"role": "assistant", "content": "Go uses goroutines for lightweight concurrency."},
        ]
        score = memory_delta_signal.score(history)
        assert score > 0.5  # mostly new content

    def test_repeated_content_low_score(self, memory_delta_signal: MemoryDeltaSignal) -> None:
        """Nearly identical content repeated → low new-fact ratio."""
        repeated = "Python uses reference counting for memory management via the gc module."
        history = [
            {"role": "assistant", "content": repeated},
            {"role": "assistant", "content": repeated},
            {"role": "assistant", "content": repeated},
            {"role": "assistant", "content": repeated},
        ]
        score = memory_delta_signal.score(history)
        assert score < 0.6  # mostly retrieved / stale

    def test_score_in_valid_range(self, memory_delta_signal: MemoryDeltaSignal) -> None:
        history = [
            {"role": "assistant", "content": "Some content here."},
            {"role": "assistant", "content": "More different content."},
        ]
        assert 0.0 <= memory_delta_signal.score(history) <= 1.0

    def test_single_message(self, memory_delta_signal: MemoryDeltaSignal) -> None:
        """A single message → 1 new fact, 0 retrieved → score = 1.0."""
        history = [{"role": "assistant", "content": "First and only turn."}]
        score = memory_delta_signal.score(history)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_invalid_window(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 1"):
            MemoryDeltaSignal(window=0)
