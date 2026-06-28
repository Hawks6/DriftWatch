"""
driftwatch/dashboard.py
───────────────────────
Rich live terminal dashboard for real-time drift monitoring.

Layout:
  ┌─ DriftWatch ─────────────────────────────────────────────┐
  │ Goal: "Summarise this codebase and identify dead code"   │
  ├──────────────────────┬───────────────────────────────────┤
  │ Health Score         │ Signal Breakdown                  │
  │  ████░░░  0.72       │ Goal Coherence   ████████░  0.81  │
  │  [HEALTHY]           │ Entropy          ██████░░░  0.68  │
  │                      │ Memory Delta     ████░░░░░  0.54  │
  ├──────────────────────┼───────────────────────────────────┤
  │ Turn 12 / ?          │ Tokens: 48,230 / 200,000  (24%)  │
  ├──────────────────────┴───────────────────────────────────┤
  │ Recent events:                                           │
  │  T08 0.79✓  T09 0.76✓  T10 0.68⚠  T11 0.61⚠  T12 0.72✓ │
  └──────────────────────────────────────────────────────────┘

Color scheme:
  >= 0.70 → green  (HEALTHY)
  0.55–0.70 → yellow (WARNING)
  < 0.55  → red    (DRIFT DETECTED)
"""
from __future__ import annotations

from typing import Optional

from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

from driftwatch.engine import DriftEvent

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _health_colour(score: float) -> str:
    if score >= 0.70:
        return "green"
    if score >= 0.55:
        return "yellow"
    return "red"


def _health_label(score: float) -> str:
    if score >= 0.70:
        return "HEALTHY"
    if score >= 0.55:
        return "WARNING"
    return "DRIFT DETECTED"


def _bar(value: float, width: int = 10) -> str:
    """Simple ASCII progress bar."""
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# DriftDashboard
# ---------------------------------------------------------------------------

class DriftDashboard:
    """
    Rich live-updating dashboard for DriftWatch.

    Usage::

        dashboard = DriftDashboard(goal="...", max_tokens=200_000)
        dashboard.start()
        # … inside agent loop …
        dashboard.update(event)
        # …
        dashboard.stop()

    Args:
        goal:       Original task description (displayed in header).
        max_tokens: Context window size for the token fill percentage.
                    Defaults to 200,000 (Claude Sonnet 4.6 window).
    """

    HISTORY_SLOTS: int = 8  # how many recent events to show in footer

    def __init__(
        self,
        goal: str,
        max_tokens: int = 200_000,
    ) -> None:
        self.goal = goal
        self.max_tokens = max_tokens
        self._events: list[DriftEvent] = []
        self._live: Optional[Live] = None
        self._console = Console()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the Rich Live context."""
        if self._live is not None:
            return
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            screen=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the Rich Live context."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def update(self, event: DriftEvent) -> None:
        """
        Update the dashboard with a new DriftEvent.

        Safe to call even if start() has not been called (no-op in that case).
        """
        self._events.append(event)
        if self._live is not None:
            self._live.update(self._render())

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> Panel:
        """Build the full dashboard panel."""
        event = self._events[-1] if self._events else None

        # Header
        goal_display = self.goal if len(self.goal) <= 55 else self.goal[:52] + "..."
        header = Text(f'Goal: "{goal_display}"', style="bold cyan")

        # Main body: two columns
        left = self._render_health(event)
        right = self._render_signals(event)

        cols = Columns([left, right], equal=True, expand=True)

        # Status bar
        status = self._render_status(event)

        # Recent events footer
        footer = self._render_footer()

        content = Group(header, Text(""), cols, Text(""), status, Text(""), footer)

        return Panel(
            content,
            title="[bold white]DriftWatch[/bold white]",
            border_style="bright_blue",
            expand=True,
        )

    def _render_health(self, event: Optional[DriftEvent]) -> Panel:
        """Left panel: composite health score."""
        if event is None:
            body = Text("Waiting for first turn…", style="dim")
        else:
            score = event.health_score
            colour = _health_colour(score)
            label = _health_label(score)
            bar = _bar(score, width=12)

            score_text = Text()
            score_text.append(f"{bar}  ", style=colour)
            score_text.append(f"{score:.2f}\n", style=f"bold {colour}")
            score_text.append(f"[{label}]", style=f"bold {colour}")
            body = score_text

        return Panel(body, title="Health Score", border_style="dim")

    def _render_signals(self, event: Optional[DriftEvent]) -> Panel:
        """Right panel: per-signal breakdown."""
        if event is None:
            return Panel(Text("No data yet", style="dim"), title="Signal Breakdown", border_style="dim")

        table = Table.grid(padding=(0, 1))
        table.add_column(min_width=18)  # label
        table.add_column(min_width=14)  # bar
        table.add_column(min_width=5)   # value

        rows = [
            ("Goal Coherence", event.goal_coherence),
            ("Entropy", event.repetition_entropy),
            ("Memory Delta", event.memory_delta),
        ]
        for label, val in rows:
            colour = _health_colour(val)
            table.add_row(
                Text(label, style="dim"),
                Text(_bar(val, width=10), style=colour),
                Text(f"{val:.2f}", style=f"bold {colour}"),
            )

        return Panel(table, title="Signal Breakdown", border_style="dim")

    def _render_status(self, event: Optional[DriftEvent]) -> Text:
        """Status line: turn counter and token usage."""
        if event is None:
            return Text("Turn 0   |   Tokens: ---", style="dim")
        turn = event.turn
        tokens = event.token_count
        pct = min(tokens / max(self.max_tokens, 1) * 100, 100)
        token_colour = "green" if pct < 50 else ("yellow" if pct < 80 else "red")

        t = Text()
        t.append(f"Turn {turn}", style="bold white")
        t.append("   |   Tokens: ", style="dim")
        t.append(f"{tokens:,}", style=f"bold {token_colour}")
        t.append(f" / {self.max_tokens:,}  ({pct:.0f}%)", style=token_colour)
        return t

    def _render_footer(self) -> Text:
        """Mini history: last N events as inline badges."""
        recent = self._events[-self.HISTORY_SLOTS :]
        t = Text("Recent: ", style="dim")
        for e in recent:
            colour = _health_colour(e.health_score)
            icon = "✓" if e.health_score >= 0.70 else ("⚠" if e.health_score >= 0.55 else "✗")
            t.append(f"T{e.turn:02d} {e.health_score:.2f}{icon}  ", style=colour)
        return t

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "DriftDashboard":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
