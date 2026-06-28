"""
examples/basic_agent.py
────────────────────────
Hello DriftWatch — a minimal 5-turn agent demo.

Requirements:
    ANTHROPIC_API_KEY environment variable

Run:
    python examples/basic_agent.py

DriftWatch wraps the Anthropic client transparently.
After every turn you'll see a printed health score.
If you're in a terminal, the live Rich dashboard also appears.
"""
from __future__ import annotations

import os
import sys

import anthropic

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import driftwatch

# ── 1. Create a real Anthropic client ───────────────────────────────────────
raw_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ── 2. Wrap it with DriftWatch ───────────────────────────────────────────────
client = driftwatch.wrap(
    raw_client,
    goal="Explain the key principles of clean code and give Python examples",
    threshold=0.55,
    on_drift="alert",          # print a warning to stderr on drift
    checkpoint_dir="./dw_checkpoints",
    dashboard=True,            # Rich live panel (auto-suppressed in non-TTY)
)

# ── 3. Run a 5-turn conversation loop ───────────────────────────────────────
messages: list[dict] = []

topics = [
    "What are the most important principles of clean code?",
    "Can you give a Python example of the Single Responsibility Principle?",
    "How does dependency injection improve testability?",
    "What's the difference between early return and guard clauses?",
    "Give me a before/after refactor of a messy Python function.",
]

print("\n🔍 DriftWatch — basic agent demo\n" + "─" * 42)

for turn, question in enumerate(topics, start=1):
    messages.append({"role": "user", "content": question})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=messages,
    )

    assistant_text = response.content[0].text
    messages.append({"role": "assistant", "content": assistant_text})

    # Retrieve the health score from the last DriftEvent
    event = client.drift_history[-1]
    status = "✓" if event.health_score >= 0.70 else ("⚠" if event.health_score >= 0.55 else "✗")
    print(
        f"Turn {turn:2d} │ health={event.health_score:.3f} {status} │ "
        f"gc={event.goal_coherence:.2f} "
        f"re={event.repetition_entropy:.2f} "
        f"md={event.memory_delta:.2f} │ "
        f"tokens={event.token_count:,}"
    )

# ── 4. Clean up ──────────────────────────────────────────────────────────────
client.stop_dashboard()
print("\n✅  Session complete.  Checkpoint log:", "./dw_checkpoints/events.jsonl")
print("    Replay with: driftwatch replay ./dw_checkpoints/events.jsonl")
