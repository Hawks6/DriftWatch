"""
driftwatch/cli.py
─────────────────
Typer CLI entry point for DriftWatch.

Commands:
  driftwatch replay  <session.jsonl>              — turn-by-turn health table
  driftwatch report  <session.jsonl> [--format]   — summary stats report
  driftwatch watch   <script.py>                  — (coming soon)

Install:
  pip install driftwatch
  # or, from source:
  pip install -e .

Usage:
  driftwatch replay events.jsonl
  driftwatch report events.jsonl --format md
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from driftwatch.engine import DriftEvent

app = typer.Typer(
    name="driftwatch",
    help="Real-time memory health monitoring for AI agents.",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_events(path: Path) -> list[DriftEvent]:
    """Load DriftEvent records from a JSONL file."""
    events: list[DriftEvent] = []
    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(code=1)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                events.append(DriftEvent.model_validate(raw))
            except Exception as exc:
                console.print(
                    f"[yellow]Skipping line {line_no}[/yellow] — {exc}"
                )
    return events


def _health_style(score: float) -> str:
    if score >= 0.70:
        return "green"
    if score >= 0.55:
        return "yellow"
    return "bold red"


def _status_text(score: float, triggered: bool) -> str:
    if triggered:
        return "[*] compacted"
    if score >= 0.70:
        return "[OK] healthy"
    if score >= 0.55:
        return "[!!] warning"
    return "[X] DRIFT"


# ---------------------------------------------------------------------------
# driftwatch replay
# ---------------------------------------------------------------------------

@app.command("replay")
def replay(
    session: Annotated[Path, typer.Argument(help="Path to DriftEvent JSONL log")],
) -> None:
    """
    Replay a DriftWatch session log as a turn-by-turn health timeline.

    \b
    Example:
        driftwatch replay ./dw_checkpoints/events.jsonl
    """
    events = _load_events(session)
    if not events:
        console.print("[yellow]No events found in file.[/yellow]")
        raise typer.Exit()

    table = Table(
        title=f"[bold cyan]DriftWatch Replay[/bold cyan] — {session.name}",
        show_header=True,
        header_style="bold white",
        border_style="bright_blue",
    )
    table.add_column("Turn", justify="right", style="dim", width=6)
    table.add_column("Health", justify="right", width=8)
    table.add_column("GC", justify="right", width=7)
    table.add_column("Entropy", justify="right", width=9)
    table.add_column("MemDelta", justify="right", width=10)
    table.add_column("Tokens", justify="right", width=10)
    table.add_column("Status", width=16)
    table.add_column("Notes", style="dim")

    for e in events:
        style = _health_style(e.health_score)
        table.add_row(
            str(e.turn),
            Text(f"{e.health_score:.2f}", style=style),
            f"{e.goal_coherence:.2f}",
            f"{e.repetition_entropy:.2f}",
            f"{e.memory_delta:.2f}",
            f"{e.token_count:,}",
            Text(_status_text(e.health_score, e.triggered_checkpoint), style=style),
            e.notes or "",
        )

    console.print(table)

    # Summary line
    avg = statistics.mean(e.health_score for e in events)
    min_event = min(events, key=lambda e: e.health_score)
    console.print(
        f"\n[dim]Average health:[/dim] [bold]{avg:.3f}[/bold]  "
        f"[dim]Worst turn:[/dim] T{min_event.turn} ({min_event.health_score:.3f})"
    )


# ---------------------------------------------------------------------------
# driftwatch report
# ---------------------------------------------------------------------------

class ReportFormat(str, Enum):
    json = "json"
    md = "md"


@app.command("report")
def report(
    session: Annotated[Path, typer.Argument(help="Path to DriftEvent JSONL log")],
    format: Annotated[ReportFormat, typer.Option("--format", "-f")] = ReportFormat.json,
) -> None:
    """
    Generate a summary report for a DriftWatch session.

    \b
    Example:
        driftwatch report ./dw_checkpoints/events.jsonl --format md
    """
    events = _load_events(session)
    if not events:
        console.print("[yellow]No events found.[/yellow]")
        raise typer.Exit()

    scores = [e.health_score for e in events]
    total_turns = len(events)
    drift_events = [e for e in events if e.health_score < 0.55]
    warning_events = [e for e in events if 0.55 <= e.health_score < 0.70]
    compaction_events = [e for e in events if e.triggered_checkpoint]
    first_drift = drift_events[0] if drift_events else None
    worst = min(events, key=lambda e: e.health_score)

    # Quartile averages
    q_size = max(total_turns // 4, 1)
    quartiles = [
        statistics.mean(scores[i * q_size : (i + 1) * q_size])
        for i in range(4)
    ]
    # Fill any missing quartiles
    while len(quartiles) < 4:
        quartiles.append(quartiles[-1] if quartiles else 0.0)

    if format == ReportFormat.json:
        data = {
            "session_file": str(session),
            "total_turns": total_turns,
            "first_drift_turn": first_drift.turn if first_drift else None,
            "worst_health_turn": worst.turn,
            "worst_health_score": worst.health_score,
            "drift_events_count": len(drift_events),
            "warning_events_count": len(warning_events),
            "compaction_events_count": len(compaction_events),
            "average_health": round(statistics.mean(scores), 4),
            "quartile_averages": [round(q, 4) for q in quartiles],
        }
        console.print_json(json.dumps(data, indent=2))

    else:  # markdown
        lines = [
            f"# DriftWatch Session Report",
            f"",
            f"**Session file:** `{session}`",
            f"",
            f"## Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total turns | {total_turns} |",
            f"| Average health | {statistics.mean(scores):.3f} |",
            f"| First drift turn | {first_drift.turn if first_drift else 'N/A'} |",
            f"| Worst health turn | T{worst.turn} ({worst.health_score:.3f}) |",
            f"| Drift events (< 0.55) | {len(drift_events)} |",
            f"| Warning events (0.55–0.70) | {len(warning_events)} |",
            f"| Compaction events | {len(compaction_events)} |",
            f"",
            f"## Health by Session Quartile",
            f"",
            f"| Quartile | Avg Health |",
            f"|----------|------------|",
        ]
        for i, q in enumerate(quartiles, start=1):
            bar = "█" * int(q * 10) + "░" * (10 - int(q * 10))
            lines.append(f"| Q{i} (turns {(i-1)*q_size+1}–{min(i*q_size, total_turns)}) | {bar} {q:.3f} |")

        console.print("\n".join(lines))


# ---------------------------------------------------------------------------
# driftwatch watch (coming soon)
# ---------------------------------------------------------------------------

@app.command("watch")
def watch(
    script: Annotated[Path, typer.Argument(help="Python script to run with DriftWatch")],
) -> None:
    """
    [bold yellow]Coming soon![/bold yellow]

    Run a Python script with DriftWatch auto-injected.

    \b
    This will:
      1. Set DRIFTWATCH_ENABLED=1 in the environment
      2. Launch the script as a subprocess
      3. Stream the live dashboard while it runs

    For now, import driftwatch directly in your script using driftwatch.wrap().
    See: https://github.com/your-org/driftwatch#getting-started
    """
    console.print(
        "[bold yellow]driftwatch watch[/bold yellow] is coming in v0.2.0!\n\n"
        "In the meantime, use [cyan]driftwatch.wrap()[/cyan] directly:\n\n"
        "  [dim]import driftwatch[/dim]\n"
        "  [dim]client = driftwatch.wrap(anthropic.Anthropic(), goal='...')\n[/dim]"
    )
    raise typer.Exit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()
