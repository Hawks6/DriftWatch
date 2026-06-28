"""
examples/long_research_agent.py
────────────────────────────────
A simulated long-running research agent that deliberately triggers drift.

This demo uses on_drift="compact" to automatically trigger Anthropic's
compact-2026-01-12 compaction API when health drops below the threshold.

The agent runs 20 turns analysing Python best practices.  Around turn 8–12,
the prompt repetition is designed to cause entropy to drop, triggering the
drift handler.

Requirements:
    ANTHROPIC_API_KEY environment variable

Run:
    python examples/long_research_agent.py
"""
from __future__ import annotations

import os
import sys
import time

import anthropic

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import driftwatch

# ── Setup ────────────────────────────────────────────────────────────────────
raw_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

client = driftwatch.wrap(
    raw_client,
    goal="Conduct a comprehensive research survey on Python performance optimisation techniques",
    threshold=0.60,       # slightly higher threshold to trigger sooner
    on_drift="compact",   # auto-compact when drifting
    checkpoint_dir="./dw_checkpoints",
    dashboard=True,
)

messages: list[dict] = []

# ── Prompt sequence designed to drift (repetition from turn 8 onward) ────────
prompts = [
    # Diverse opening turns (high health expected)
    "What are the top Python performance bottlenecks in data processing pipelines?",
    "Explain how Python's GIL affects multi-threaded performance.",
    "What are the benefits of using numpy arrays over native Python lists?",
    "How does profiling with cProfile help identify hotspots?",
    "Explain memory management in Python — reference counting and the gc module.",
    "What is the difference between multiprocessing and asyncio for I/O-bound tasks?",
    "How do Python generators reduce memory overhead compared to lists?",
    # Repetitive turns (designed to trigger drift)
    "Summarise Python performance bottlenecks again.",
    "Can you repeat the key points about the GIL?",
    "Tell me again about numpy vs lists.",
    "Repeat the profiling advice from earlier.",
    "Summarise everything about performance so far.",
    # Recovery turns (diverse topics again)
    "What are Cython and Numba, and when should I use them?",
    "How does functools.lru_cache improve repeated computation performance?",
    "Explain slot classes (__slots__) for memory-efficient objects.",
    "What is PyPy and when does it outperform CPython significantly?",
    "How do I benchmark Python code reliably with timeit and pytest-benchmark?",
    "Explain the performance impact of comprehensions vs map/filter.",
    "What tools exist for visualising Python memory usage over time?",
    "Summarise the top 5 actionable performance tips for production Python.",
]

print("\n🔬 DriftWatch — long research agent demo")
print("   (designed to trigger drift + auto-compaction)")
print("─" * 55)

for turn, prompt in enumerate(prompts, start=1):
    messages.append({"role": "user", "content": prompt})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=messages,
    )

    assistant_text = response.content[0].text
    messages.append({"role": "assistant", "content": assistant_text})

    event = client.drift_history[-1]
    status_icon = "✓" if event.health_score >= 0.70 else ("⚠" if event.health_score >= 0.60 else "✗")
    compacted = " ← COMPACTED" if event.triggered_checkpoint else ""

    print(
        f"T{turn:02d} │ {event.health_score:.3f} {status_icon} │ "
        f"gc={event.goal_coherence:.2f} "
        f"re={event.repetition_entropy:.2f} "
        f"md={event.memory_delta:.2f}{compacted}"
    )

    time.sleep(0.5)  # be kind to the API

client.stop_dashboard()

print("\n─" * 55)
print("✅  Session complete.")
print("   Replay: driftwatch replay ./dw_checkpoints/events.jsonl")
print("   Report: driftwatch report ./dw_checkpoints/events.jsonl --format md")
