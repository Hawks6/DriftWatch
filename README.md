# DriftWatch 🧭

> **Real-time memory health monitoring for AI agents.**
> Detect context rot before your agent goes off the rails.

[![PyPI version](https://img.shields.io/pypi/v/driftwatch.svg?color=blue&label=pypi)](https://pypi.org/project/driftwatch/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2601.04170-b31b1b.svg)](https://arxiv.org/abs/2601.04170)

```
┌─ DriftWatch ─────────────────────────────────────────────────────────┐
│ Goal: "Conduct a comprehensive research survey on Python performance" │
├──────────────────────────┬───────────────────────────────────────────┤
│ Health Score             │ Signal Breakdown                          │
│                          │                                           │
│  ████████░░░  0.72       │  Goal Coherence   ██████████░  0.81      │
│  [HEALTHY]               │  Entropy          ████████░░░  0.68      │
│                          │  Memory Delta     █████░░░░░░  0.54      │
├──────────────────────────┼───────────────────────────────────────────┤
│ Turn 12                  │ Tokens: 48,230 / 200,000  (24%)          │
├──────────────────────────┴───────────────────────────────────────────┤
│ Recent: T08 0.79✓  T09 0.76✓  T10 0.68⚠  T11 0.61⚠  T12 0.72✓    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## The problem

Long-running AI agents don't fail all at once — they drift.
By the time your agent produces clearly wrong output, it has been silently
degrading for dozens of turns. **Context rot** is the progressive loss of
reasoning quality that starts at 60–70% context fill, not at 100%.

A 2025–2026 industry analysis found that **~65% of enterprise AI agent
failures** were caused by context drift or memory loss during multi-step
reasoning — *not* by raw context exhaustion. The degradation is measurable,
predictable, and preventable. DriftWatch does all three.

---

## Install

```bash
pip install agent-driftwatch
```

Or from source:

```bash
git clone https://github.com/your-org/driftwatch
cd driftwatch
pip install -e .
```

---

## 30-second start

```python
import os
import anthropic
import driftwatch

# Wrap your existing Anthropic client — one line change
client = driftwatch.wrap(
    anthropic.Anthropic(),
    goal="Explain the key principles of clean code and give Python examples",
    threshold=0.55,   # trigger action below this health score
    on_drift="alert", # "checkpoint" | "compact" | "alert" | callable
    dashboard=True,   # Rich live terminal panel
)

messages = []
topics = [
    "What are the most important principles of clean code?",
    "Can you give a Python example of the Single Responsibility Principle?",
    "How does dependency injection improve testability?",
    "What's the difference between early return and guard clauses?",
    "Give me a before/after refactor of a messy Python function.",
]

for turn, question in enumerate(topics, start=1):
    messages.append({"role": "user", "content": question})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=messages,
    )

    messages.append({"role": "assistant", "content": response.content[0].text})

    event = client.drift_history[-1]
    print(f"Turn {turn} | health={event.health_score:.3f} | tokens={event.token_count:,}")
```

Output:
```
Turn  1 | health=0.914 | tokens=1,240
Turn  2 | health=0.882 | tokens=2,890
Turn  3 | health=0.856 | tokens=4,780
Turn  4 | health=0.824 | tokens=7,120
Turn  5 | health=0.793 | tokens=9,870
```

---

## How it works

DriftWatch computes a composite **health score** (0.0–1.0) after every turn
by combining three independently validated signals:

| Signal | What it measures | Method |
|--------|-----------------|--------|
| **Goal Coherence** | How closely the agent's response aligns with the original task intent | Cosine similarity between goal embedding and last-turn embedding ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) |
| **Repetition Entropy** | Whether the agent is looping or executing diverse actions | Shannon entropy over tool call names / word bigrams in a sliding window |
| **Memory Delta** | Whether the agent is introducing new facts or just repeating prior context | New-fact ratio via embedding centroid comparison |

The composite score is a configurable weighted average:

```
health_score = 0.50 × goal_coherence
             + 0.30 × repetition_entropy
             + 0.20 × memory_delta
```

**Color thresholds:**
- 🟢 `>= 0.70` — Healthy
- 🟡 `0.55–0.70` — Warning (drift beginning)
- 🔴 `< 0.55` — Drift detected

**Research basis:** arXiv:2601.04170 (Rath, Jan 2026) — *"Agent Drift:
Quantifying Behavioral Degradation in Multi-Agent LLM Systems"* — formally
defines semantic drift, coordination drift, and behavioral drift, and
introduces the Agent Stability Index (ASI) composite metric that DriftWatch
implements.

---

## Auto-compaction

When `on_drift="compact"`, DriftWatch automatically triggers Anthropic's
**compact-2026-01-12** API to summarise the conversation before continuing:

```python
client = driftwatch.wrap(
    anthropic.Anthropic(),
    goal="Analyse this codebase for dead code",
    on_drift="compact",  # ← auto-compaction on drift
)
```

Under the hood, when `health_score < threshold`:

```python
# DriftWatch calls this automatically:
response = client.beta.messages.create(
    betas=["compact-2026-01-12"],
    model=model,
    max_tokens=1024,
    messages=messages,
    context_management={
        "edits": [{
            "type": "compact_20260112",
            "pause_after_compaction": True,
            "instructions": "Preserve: original goal, all tool call results, "
                            "decisions made, files modified. "
                            "Discard: repeated tool outputs, exploratory tangents.",
        }]
    },
)
```

The compacted summary replaces the conversation history, token count resets,
and health scores recover — all transparently. Your agent loop code doesn't
change at all.

---

## on_drift handlers

| Handler | Behaviour |
|---------|-----------|
| `"checkpoint"` | Save messages + DriftEvent log to `checkpoint_dir/` |
| `"compact"` | Trigger Anthropic compaction, then save checkpoint |
| `"alert"` | Print a warning to `stderr` and continue |
| `"none"` | Monitor silently, take no action |
| `callable` | Call `fn(client, event)` — fully custom handler |

```python
def my_handler(client, event):
    send_slack_alert(f"Agent drift detected! health={event.health_score:.2f}")
    client.save_checkpoint(messages)

client = driftwatch.wrap(anthropic.Anthropic(), goal="...", on_drift=my_handler)
```

---

## CLI

### Replay a session

Visualise a saved event log as a turn-by-turn health timeline:

```bash
driftwatch replay ./dw_checkpoints/events.jsonl
```

```
                  DriftWatch Replay — events.jsonl
 Turn │ Health │  GC   │ Entropy │ MemDelta │ Tokens  │ Status
──────┼────────┼───────┼─────────┼──────────┼─────────┼──────────────
    1 │  0.92  │  0.95 │   0.88  │   0.93   │   1,240 │ ✓ healthy
    2 │  0.89  │  0.91 │   0.85  │   0.91   │   2,890 │ ✓ healthy
  ...
   10 │  0.52  │  0.58 │   0.35  │   0.42   │  28,700 │ ✗ DRIFT
   11 │  0.48  │  0.54 │   0.28  │   0.38   │  31,200 │ ✗ DRIFT
   12 │  0.44  │  0.49 │   0.22  │   0.33   │  33,800 │ ★ compacted
   13 │  0.83  │  0.85 │   0.78  │   0.88   │   4,200 │ ✓ healthy
```

### Generate a session report

```bash
driftwatch report ./dw_checkpoints/events.jsonl --format md
```

```markdown
# DriftWatch Session Report

| Metric | Value |
|--------|-------|
| Total turns | 20 |
| Average health | 0.741 |
| First drift turn | T10 |
| Worst health turn | T12 (0.438) |
| Drift events (< 0.55) | 3 |
| Compaction events | 1 |
```

Or as JSON:
```bash
driftwatch report ./dw_checkpoints/events.jsonl --format json
```

### Try the fixture

```bash
driftwatch replay tests/fixtures/demo_session.jsonl
```

---

## Configuration reference

```python
client = driftwatch.wrap(
    anthropic.Anthropic(),
    goal="...",                    # required: the semantic anchor
    threshold=0.55,                # health score that triggers on_drift
    on_drift="checkpoint",         # handler (see table above)
    checkpoint_dir="./dw_checkpoints",  # where to save files
    dashboard=True,                # Rich live UI (auto-suppressed in CI)
    max_context_tokens=200_000,    # context window for token % display
    weights={                      # override composite signal weights
        "goal_coherence": 0.50,
        "repetition_entropy": 0.30,
        "memory_delta": 0.20,
    },
    log_path=None,                 # custom JSONL log path
)
```

---

## DriftEvent schema

Every turn produces a `DriftEvent` (Pydantic model):

```python
@dataclass
class DriftEvent:
    turn: int                    # monotonically increasing (1-based)
    timestamp: datetime          # UTC
    goal_coherence: float        # Signal 1: [0.0, 1.0]
    repetition_entropy: float    # Signal 2: [0.0, 1.0]
    memory_delta: float          # Signal 3: [0.0, 1.0]
    health_score: float          # weighted composite: [0.0, 1.0]
    token_count: int             # input_tokens from API usage
    triggered_checkpoint: bool   # True if on_drift handler fired
    notes: str                   # optional annotation
```

Access the full history:

```python
for event in client.drift_history:
    print(f"T{event.turn}: {event.health_score:.3f}")
```

---

## Roadmap

- [ ] OpenAI SDK support
- [ ] LangGraph integration (`DriftWatchCallbackHandler`)
- [ ] Multi-agent drift — coordination drift signal across agent network
- [ ] GitHub Actions reporter (`driftwatch-action`)
- [ ] Prometheus metrics endpoint
- [ ] `driftwatch watch <script.py>` — subprocess injection (CLI v0.2)
- [ ] Grafana dashboard template

---

## Architecture

```
driftwatch/
├── signals.py      ← 3 drift signal classes (offline, no API key)
├── engine.py       ← composite scorer + DriftEvent schema
├── wrapper.py      ← Anthropic SDK intercept layer
├── checkpoint.py   ← save/restore + compaction API
├── dashboard.py    ← Rich live terminal UI
└── cli.py          ← Typer CLI (replay, report, watch)
```

DriftWatch is an **observer** — it never modifies the response your code
receives from the Anthropic SDK.  It intercepts only to evaluate and log.
The sole exception is `on_drift="compact"`, which updates your `messages`
list in place after compaction (your agent continues seamlessly).

---

## Contributing

```bash
git clone https://github.com/your-org/driftwatch
cd driftwatch
pip install -e ".[dev]"
python -m pytest tests/ -v
```

All signal tests run without an API key. PRs welcome!

---

## Citation

If you use DriftWatch in academic research, please cite the foundational work
this library is built on:

```bibtex
@article{rath2026agentdrift,
  title   = {Agent Drift: Quantifying Behavioral Degradation in Multi-Agent LLM Systems},
  author  = {Rath, et al.},
  journal = {arXiv preprint arXiv:2601.04170},
  year    = {2026}
}
```

**Related papers:**
- arXiv:2505.02709 — *"Technical Report: Evaluating Goal Drift in Language Model Agents"*
  — defines GD_actions and GD_inaction metrics
- arXiv:2510.00615 — *"ACON: Optimizing Context Compression for Long-horizon LLM Agents"*
  — validates 26–54% peak token reduction with smart compression

---

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  Built with ❤️ for the AI engineering community.<br/>
  If DriftWatch saved your agent, give it a ⭐
</p>
