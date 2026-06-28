"""
driftwatch/checkpoint.py
────────────────────────
Checkpoint save/restore and Anthropic compaction API integration.

CheckpointManager handles three operations:
  1. save()                — persist messages + DriftEvents to JSONL files
  2. save_with_compaction() — trigger Anthropic compact-2026-01-12 beta, then save
  3. restore()             — reload messages + DriftEvents from a checkpoint

Compaction API reference:
  betas=["compact-2026-01-12"]
  context_management.edits[0].type = "compact_20260112"
  Response stop_reason == "compaction" → extract compaction block
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import jsonlines

from driftwatch.engine import DriftEvent

if TYPE_CHECKING:
    import anthropic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso_filename_stem() -> str:
    """Return a filesystem-safe ISO-8601 timestamp string."""
    now = datetime.now(tz=timezone.utc)
    return now.strftime("%Y-%m-%dT%H-%M-%S-%f")


def _serialise_message(msg: dict) -> dict:
    """
    Convert a message dict (which may contain Anthropic SDK objects) to a
    plain JSON-serialisable dict.
    """
    if isinstance(msg, dict):
        content = msg.get("content", "")
        if isinstance(content, str):
            return {"role": msg["role"], "content": content}
        if isinstance(content, list):
            blocks = []
            for block in content:
                if isinstance(block, dict):
                    blocks.append(block)
                elif hasattr(block, "model_dump"):
                    blocks.append(block.model_dump())
                elif hasattr(block, "__dict__"):
                    blocks.append(vars(block))
                else:
                    blocks.append(str(block))
            return {"role": msg["role"], "content": blocks}
    return msg


def _deserialise_event(raw: dict) -> DriftEvent:
    return DriftEvent.model_validate(raw)


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Manages persisting and restoring DriftWatch session state.

    Each checkpoint creates two files in checkpoint_dir:
      {stem}_messages.jsonl  — one JSON object per line (the message history)
      {stem}_events.jsonl    — one DriftEvent JSON per line

    Args:
        checkpoint_dir: Directory where checkpoint files are written.
                        Created automatically if it does not exist.
    """

    def __init__(self, checkpoint_dir: str | os.PathLike = "./dw_checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        messages: list[dict],
        drift_events: list[DriftEvent],
        metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        """
        Persist the current message history and DriftEvent log.

        Args:
            messages:     Current conversation message list.
            drift_events: The engine's history of DriftEvents.
            metadata:     Optional free-form metadata (goal, turn, etc.).

        Returns:
            Path to the saved messages JSONL file.
        """
        stem = _iso_filename_stem()
        msg_path = self.checkpoint_dir / f"{stem}_messages.jsonl"
        events_path = self.checkpoint_dir / f"{stem}_events.jsonl"

        # Write messages
        with jsonlines.open(msg_path, mode="w") as writer:
            if metadata:
                writer.write({"__metadata__": metadata})
            for msg in messages:
                writer.write(_serialise_message(msg))

        # Write events
        with jsonlines.open(events_path, mode="w") as writer:
            for event in drift_events:
                writer.write(event.model_dump(mode="json"))

        return msg_path

    # ------------------------------------------------------------------
    # Compact-then-save
    # ------------------------------------------------------------------

    def save_with_compaction(
        self,
        client: "anthropic.Anthropic",
        messages: list[dict],
        drift_events: list[DriftEvent],
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        compaction_instructions: str = (
            "Preserve: original goal, all tool call results, decisions made, "
            "files modified. Discard: repeated tool outputs, exploratory tangents."
        ),
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[Path, list[dict]]:
        """
        Trigger Anthropic's compact-2026-01-12 API, append the compaction block
        to the message list, then save the compacted history to disk.

        Uses `pause_after_compaction=True` so we receive the compaction block
        and can inspect / save it before resuming.

        Args:
            client:                  The raw anthropic.Anthropic client.
            messages:                Current message history (mutated in place).
            drift_events:            DriftEvent history to persist alongside.
            model:                   Anthropic model string.
            max_tokens:              Max tokens for the compaction request.
            compaction_instructions: Preservation instructions for the summary.
            metadata:                Optional metadata dict.

        Returns:
            Tuple of (path_to_messages_file, updated_messages_list).
        """
        response = client.beta.messages.create(
            betas=["compact-2026-01-12"],
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            context_management={
                "edits": [
                    {
                        "type": "compact_20260112",
                        "pause_after_compaction": True,
                        "instructions": compaction_instructions,
                    }
                ]
            },
        )

        if response.stop_reason == "compaction":
            # The compaction block replaces previous history
            compaction_content = response.content
            # Build a fresh message list: the compaction block acts as a
            # synthetic assistant turn that summarises what came before
            messages.clear()
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        block.model_dump() if hasattr(block, "model_dump") else block
                        for block in compaction_content
                    ],
                }
            )

        saved_path = self.save(messages, drift_events, metadata=metadata)
        return saved_path, messages

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore(
        self, messages_path: str | os.PathLike
    ) -> tuple[list[dict], list[DriftEvent]]:
        """
        Reload a saved checkpoint from disk.

        Args:
            messages_path: Path to a *_messages.jsonl checkpoint file.

        Returns:
            (messages, drift_events) tuple.
        """
        messages_path = Path(messages_path)
        if not messages_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {messages_path}")

        # Derive events path from messages path
        events_path = Path(str(messages_path).replace("_messages.jsonl", "_events.jsonl"))

        messages: list[dict] = []
        with jsonlines.open(messages_path, mode="r") as reader:
            for obj in reader:
                if "__metadata__" in obj:
                    continue  # skip metadata line
                messages.append(obj)

        drift_events: list[DriftEvent] = []
        if events_path.exists():
            with jsonlines.open(events_path, mode="r") as reader:
                for obj in reader:
                    try:
                        drift_events.append(_deserialise_event(obj))
                    except Exception:
                        pass  # Skip malformed records

        return messages, drift_events

    # ------------------------------------------------------------------
    # List available checkpoints
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> list[Path]:
        """Return all message checkpoint files sorted by creation time."""
        return sorted(
            self.checkpoint_dir.glob("*_messages.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
