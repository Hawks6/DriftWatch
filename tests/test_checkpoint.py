"""
tests/test_checkpoint.py
─────────────────────────
Unit tests for CheckpointManager — save, restore, and list operations.

These tests run fully offline — no Anthropic API key required.
Compaction tests are marked with @pytest.mark.skip as they require
a real Anthropic client.

Run:
    python -m pytest tests/test_checkpoint.py -v
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from driftwatch.checkpoint import CheckpointManager
from driftwatch.engine import DriftEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    return tmp_path / "checkpoints"


@pytest.fixture
def manager(tmp_checkpoint_dir: Path) -> CheckpointManager:
    return CheckpointManager(checkpoint_dir=tmp_checkpoint_dir)


@pytest.fixture
def sample_messages() -> list[dict]:
    return [
        {"role": "user", "content": "What is clean code?"},
        {"role": "assistant", "content": "Clean code is readable and maintainable."},
        {"role": "user", "content": "Give me a Python example."},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Here is a simple example of a clean function."}
            ],
        },
    ]


@pytest.fixture
def sample_events() -> list[DriftEvent]:
    return [
        DriftEvent(
            turn=1,
            timestamp=datetime(2026, 6, 28, 10, 0, 0, tzinfo=timezone.utc),
            goal_coherence=0.88,
            repetition_entropy=0.75,
            memory_delta=0.90,
            health_score=0.84,
            token_count=1200,
            triggered_checkpoint=False,
        ),
        DriftEvent(
            turn=2,
            timestamp=datetime(2026, 6, 28, 10, 1, 0, tzinfo=timezone.utc),
            goal_coherence=0.52,
            repetition_entropy=0.40,
            memory_delta=0.35,
            health_score=0.44,
            token_count=3800,
            triggered_checkpoint=True,
            notes="threshold breached",
        ),
    ]


# ---------------------------------------------------------------------------
# CheckpointManager initialisation
# ---------------------------------------------------------------------------

class TestCheckpointManagerInit:
    def test_creates_directory(self, tmp_checkpoint_dir: Path) -> None:
        assert not tmp_checkpoint_dir.exists()
        CheckpointManager(checkpoint_dir=tmp_checkpoint_dir)
        assert tmp_checkpoint_dir.exists()

    def test_existing_directory_ok(self, tmp_checkpoint_dir: Path) -> None:
        tmp_checkpoint_dir.mkdir(parents=True)
        # Should not raise
        CheckpointManager(checkpoint_dir=tmp_checkpoint_dir)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

class TestSave:
    def test_save_creates_two_files(
        self,
        manager: CheckpointManager,
        sample_messages: list[dict],
        sample_events: list[DriftEvent],
    ) -> None:
        path = manager.save(sample_messages, sample_events)
        assert path.exists()

        events_path = Path(str(path).replace("_messages.jsonl", "_events.jsonl"))
        assert events_path.exists()

    def test_saved_messages_count(
        self,
        manager: CheckpointManager,
        sample_messages: list[dict],
        sample_events: list[DriftEvent],
    ) -> None:
        path = manager.save(sample_messages, sample_events)
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) == len(sample_messages)

    def test_saved_events_count(
        self,
        manager: CheckpointManager,
        sample_messages: list[dict],
        sample_events: list[DriftEvent],
    ) -> None:
        path = manager.save(sample_messages, sample_events)
        events_path = Path(str(path).replace("_messages.jsonl", "_events.jsonl"))
        lines = [l for l in events_path.read_text().splitlines() if l.strip()]
        assert len(lines) == len(sample_events)

    def test_metadata_written(
        self,
        manager: CheckpointManager,
        sample_messages: list[dict],
        sample_events: list[DriftEvent],
    ) -> None:
        path = manager.save(
            sample_messages, sample_events, metadata={"goal": "test", "turn": 2}
        )
        first_line = json.loads(path.read_text().splitlines()[0])
        assert "__metadata__" in first_line
        assert first_line["__metadata__"]["goal"] == "test"

    def test_save_returns_path(
        self,
        manager: CheckpointManager,
        sample_messages: list[dict],
        sample_events: list[DriftEvent],
    ) -> None:
        path = manager.save(sample_messages, sample_events)
        assert isinstance(path, Path)
        assert "_messages.jsonl" in path.name

    def test_save_empty_messages(
        self, manager: CheckpointManager, sample_events: list[DriftEvent]
    ) -> None:
        path = manager.save([], sample_events)
        assert path.exists()

    def test_save_empty_events(
        self, manager: CheckpointManager, sample_messages: list[dict]
    ) -> None:
        path = manager.save(sample_messages, [])
        events_path = Path(str(path).replace("_messages.jsonl", "_events.jsonl"))
        assert events_path.exists()
        lines = [l for l in events_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 0


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------

class TestRestore:
    def test_restore_messages(
        self,
        manager: CheckpointManager,
        sample_messages: list[dict],
        sample_events: list[DriftEvent],
    ) -> None:
        path = manager.save(sample_messages, sample_events)
        restored_msgs, _ = manager.restore(path)
        assert len(restored_msgs) == len(sample_messages)

    def test_restore_message_content(
        self,
        manager: CheckpointManager,
        sample_messages: list[dict],
        sample_events: list[DriftEvent],
    ) -> None:
        path = manager.save(sample_messages, sample_events)
        restored, _ = manager.restore(path)
        for original, restored_msg in zip(sample_messages, restored):
            assert restored_msg["role"] == original["role"]

    def test_restore_events(
        self,
        manager: CheckpointManager,
        sample_messages: list[dict],
        sample_events: list[DriftEvent],
    ) -> None:
        path = manager.save(sample_messages, sample_events)
        _, restored_events = manager.restore(path)
        assert len(restored_events) == len(sample_events)

    def test_restore_event_fields(
        self,
        manager: CheckpointManager,
        sample_messages: list[dict],
        sample_events: list[DriftEvent],
    ) -> None:
        path = manager.save(sample_messages, sample_events)
        _, restored = manager.restore(path)
        assert restored[0].turn == 1
        assert restored[0].health_score == pytest.approx(0.84, abs=1e-5)
        assert restored[1].triggered_checkpoint is True
        assert restored[1].notes == "threshold breached"

    def test_restore_with_metadata(
        self,
        manager: CheckpointManager,
        sample_messages: list[dict],
        sample_events: list[DriftEvent],
    ) -> None:
        path = manager.save(sample_messages, sample_events, metadata={"goal": "test"})
        restored_msgs, _ = manager.restore(path)
        # Metadata line should be skipped; message count unchanged
        assert len(restored_msgs) == len(sample_messages)

    def test_restore_nonexistent_raises(self, manager: CheckpointManager) -> None:
        with pytest.raises(FileNotFoundError):
            manager.restore("./nonexistent_messages.jsonl")

    def test_restore_without_events_file(
        self,
        manager: CheckpointManager,
        sample_messages: list[dict],
    ) -> None:
        """If events file is missing, restore still returns messages with empty events."""
        path = manager.save(sample_messages, [])
        events_path = Path(str(path).replace("_messages.jsonl", "_events.jsonl"))
        events_path.unlink()  # delete events file

        restored_msgs, restored_events = manager.restore(path)
        assert len(restored_msgs) == len(sample_messages)
        assert restored_events == []


# ---------------------------------------------------------------------------
# List checkpoints
# ---------------------------------------------------------------------------

class TestListCheckpoints:
    def test_empty_dir(self, manager: CheckpointManager) -> None:
        assert manager.list_checkpoints() == []

    def test_lists_message_files_only(
        self,
        manager: CheckpointManager,
        sample_messages: list[dict],
        sample_events: list[DriftEvent],
    ) -> None:
        manager.save(sample_messages, sample_events)
        manager.save(sample_messages, sample_events)
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 2
        assert all("_messages.jsonl" in str(p) for p in checkpoints)


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_full_round_trip(
        self,
        manager: CheckpointManager,
        sample_messages: list[dict],
        sample_events: list[DriftEvent],
    ) -> None:
        """Save → restore → verify all data survives intact."""
        path = manager.save(sample_messages, sample_events, metadata={"turn": 2})
        msgs, events = manager.restore(path)

        assert len(msgs) == len(sample_messages)
        assert len(events) == len(sample_events)
        assert events[0].goal_coherence == pytest.approx(0.88, abs=1e-5)
        assert events[1].token_count == 3800
