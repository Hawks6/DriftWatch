"""
driftwatch/signals.py
─────────────────────
Three drift-signal classes, each returning a health score in [0.0, 1.0]
where 1.0 = fully healthy and 0.0 = fully drifted.

No Anthropic SDK imports — this module is independently testable offline.

Scientific basis:
  - Semantic drift: cosine similarity (arXiv:2601.04170, arXiv:2505.02709)
  - Repetition/looping: Shannon entropy (arXiv:2601.04170)
  - Memory delta: new-fact ratio via embedding centroids
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Shared model singleton (lazy-loaded, shared across all signal instances)
# ---------------------------------------------------------------------------

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, safe against zero norm."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _shannon_entropy(items: list[str]) -> float:
    """Shannon entropy H = -Σ p(x) log₂ p(x) over a sequence of discrete items."""
    if not items:
        return 0.0
    counts = Counter(items)
    total = len(items)
    probs = np.array([c / total for c in counts.values()], dtype=np.float64)
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


def _extract_assistant_texts(history: list[dict]) -> list[str]:
    """Return a list of text strings from all assistant messages in history."""
    texts: list[str] = []
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        # Extract text blocks
        if isinstance(content, str) and content.strip():
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif hasattr(block, "type") and block.type == "text":
                    texts.append(block.text)
                    
        # If there are tool calls (OpenAI format)
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                func = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
                if func:
                    name = func.get("name", "") if isinstance(func, dict) else getattr(func, "name", "")
                    args = func.get("arguments", "") if isinstance(func, dict) else getattr(func, "arguments", "")
                    texts.append(f"Tool: {name} {args}")
        
        # If there are tool uses (Anthropic format inside content)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    texts.append(f"Tool: {block.get('name', '')} {block.get('input', '')}")
                elif hasattr(block, "type") and block.type == "tool_use":
                    texts.append(f"Tool: {getattr(block, 'name', '')} {getattr(block, 'input', '')}")
                    
    return [t for t in texts if t.strip()]


def _extract_tool_calls(msg: dict) -> list[str]:
    """Extract tool call names from a single message's content blocks."""
    names: list[str] = []
    # 1. Anthropic format
    content = msg.get("content", [])
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                names.append(block.get("name", "unknown_tool"))
            elif hasattr(block, "type") and block.type == "tool_use":
                names.append(getattr(block, "name", "unknown_tool"))
                
    # 2. OpenAI format
    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            func = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
            if func:
                name = func.get("name", "unknown_tool") if isinstance(func, dict) else getattr(func, "name", "unknown_tool")
                names.append(name)
                
    return names


def _text_to_bigrams(text: str) -> list[str]:
    """Tokenise text into word bigrams for entropy computation."""
    words = text.lower().split()
    if len(words) < 2:
        return words
    return [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]


# ---------------------------------------------------------------------------
# Signal 1 — GoalCoherenceSignal
# ---------------------------------------------------------------------------

class GoalCoherenceSignal:
    """
    Measures how closely the agent's latest response aligns with the original
    stated goal by computing cosine similarity between the goal embedding and
    the most-recent assistant message embedding.

    Returns 1.0 when the agent is on-topic, approaching 0.0 as it drifts away.
    """

    def __init__(self, goal: str) -> None:
        self.goal = goal
        model = _get_model()
        self._goal_embedding: np.ndarray = model.encode(goal, convert_to_numpy=True)

    def score(self, history: list[dict]) -> float:
        """
        Args:
            history: List of message dicts (role/content pairs).

        Returns:
            Cosine similarity in [0.0, 1.0].  Returns 0.5 if no assistant
            message exists yet (neutral / unknown health).
        """
        texts = _extract_assistant_texts(history)
        if not texts:
            return 0.5  # neutral — no data yet

        last_text = texts[-1]
        model = _get_model()
        turn_embedding: np.ndarray = model.encode(last_text, convert_to_numpy=True)
        sim = _cosine_sim(self._goal_embedding, turn_embedding)
        # Cosine similarity ∈ [−1, 1]; clip to [0, 1]
        return float(np.clip(sim, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Signal 2 — RepetitionEntropySignal
# ---------------------------------------------------------------------------

class RepetitionEntropySignal:
    """
    Detects looping / repetitive behaviour by measuring the Shannon entropy
    of tool-call names (or word bigrams) over a sliding window of turns.

    High entropy → diverse actions → healthy (score near 1.0).
    Low entropy  → looping / stuck  → drifted (score near 0.0).
    """

    def __init__(self, window: int = 10) -> None:
        if window < 2:
            raise ValueError("window must be >= 2 for meaningful entropy")
        self.window = window

    def score(self, history: list[dict]) -> float:
        """
        Args:
            history: List of message dicts.

        Returns:
            Normalised entropy in [0.0, 1.0].
        """
        # Collect the last `window` assistant messages
        assistant_msgs = [m for m in history if m.get("role") == "assistant"]
        window_msgs = assistant_msgs[-self.window :]

        items: list[str] = []
        for msg in window_msgs:
            tool_names = _extract_tool_calls(msg)
            if tool_names:
                items.extend(tool_names)
            else:
                # Fall back to bigrams of the text response
                content = msg.get("content", "")
                if isinstance(content, str):
                    items.extend(_text_to_bigrams(content))
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            items.extend(_text_to_bigrams(block.get("text", "")))
                        elif hasattr(block, "type") and block.type == "text":
                            items.extend(_text_to_bigrams(block.text))

        if not items:
            return 0.5  # neutral — no data

        H = _shannon_entropy(items)
        max_H = math.log2(self.window)  # theoretical max entropy
        if max_H == 0:
            return 1.0
        return float(min(H / max_H, 1.0))


# ---------------------------------------------------------------------------
# Signal 3 — MemoryDeltaSignal
# ---------------------------------------------------------------------------

class MemoryDeltaSignal:
    """
    Detects memory stagnation by tracking how many new facts the agent
    introduces versus how many it merely retrieves / repeats.

    High new-fact rate → the agent is making progress (score near 1.0).
    Flat / repetitive → stalling (score near 0.0).
    """

    # Similarity threshold above which a block is considered a "retrieval"
    RETRIEVAL_THRESHOLD: float = 0.85

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window

    def score(self, history: list[dict]) -> float:
        """
        Args:
            history: List of message dicts.

        Returns:
            Ratio of new facts to total facts in [0.0, 1.0].
        """
        texts = _extract_assistant_texts(history)
        window_texts = texts[-self.window :]

        if not window_texts:
            return 0.5  # neutral

        model = _get_model()
        embeddings: list[np.ndarray] = [
            model.encode(t, convert_to_numpy=True) for t in window_texts
        ]

        new_facts = 0
        retrieved_facts = 0
        centroid: Optional[np.ndarray] = None

        for emb in embeddings:
            if centroid is None:
                # First block always counts as new
                new_facts += 1
                centroid = emb.copy()
            else:
                sim = _cosine_sim(emb, centroid)
                if sim >= self.RETRIEVAL_THRESHOLD:
                    retrieved_facts += 1
                else:
                    new_facts += 1
                    # Update centroid (running mean)
                    n = new_facts
                    centroid = centroid + (emb - centroid) / n

        total = new_facts + retrieved_facts
        return float(new_facts / max(total, 1))
