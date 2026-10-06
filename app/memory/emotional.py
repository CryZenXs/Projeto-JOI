"""Projeto JOI - Emotional Memory layer.

Tracks the emotional tone associated with each topic for each user.
Allows the JOI to recall "we had a sad conversation about your dog"
and respond with appropriate sensitivity.

Backend: SQLite (shared database file with semantic memory).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.memory.base import EmotionalMemoryEntry, EmotionalTone, MemoryClient

logger = get_logger(__name__)


class EmotionalMemoryStore(MemoryClient):
    """Abstract interface for emotional memory backends."""

    async def record(
        self,
        user_id: str,
        topic: str,
        tone: EmotionalTone,
        intensity: float = 0.5,
    ) -> EmotionalMemoryEntry | None:
        """Record an emotional association with a topic."""
        raise NotImplementedError

    async def get_topic_tone(
        self,
        user_id: str,
        topic: str,
    ) -> EmotionalMemoryEntry | None:
        """Get the dominant emotional tone for a topic."""
        raise NotImplementedError

    async def get_all(
        self,
        user_id: str,
    ) -> list[EmotionalMemoryEntry]:
        """Get all emotional associations for a user."""
        raise NotImplementedError


class SQLiteEmotionalStore(EmotionalMemoryStore):
    """SQLite-backed emotional memory.

    Uses (user_id, topic) as a unique key and aggregates occurrences:
    each time the same tone is seen for a topic, occurrence_count is
    incremented and intensity is updated as a running average.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS emotional_memory (
        entry_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        topic TEXT NOT NULL,
        tone TEXT NOT NULL,
        intensity REAL DEFAULT 0.5,
        timestamp REAL NOT NULL,
        occurrence_count INTEGER DEFAULT 1,
        UNIQUE(user_id, topic)
    );

    CREATE INDEX IF NOT EXISTS idx_emotional_user ON emotional_memory(user_id);
    CREATE INDEX IF NOT EXISTS idx_emotional_user_topic ON emotional_memory(user_id, topic);
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            persist_dir = Path(settings.chroma_persist_dir)
            try:
                persist_dir.mkdir(parents=True, exist_ok=True)
                db_path = persist_dir / "memory.db"
            except (OSError, PermissionError):
                db_path = Path("/tmp/joi_memory.db")

        self._db_path = str(db_path)

        with self._get_conn() as conn:
            conn.executescript(self.SCHEMA)

        logger.info(
            "memory.emotional.initialized",
            backend="sqlite",
            db_path=self._db_path,
        )

    @contextmanager
    def _get_conn(self) -> Any:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    async def record(
        self,
        user_id: str,
        topic: str,
        tone: EmotionalTone,
        intensity: float = 0.5,
    ) -> EmotionalMemoryEntry | None:
        """Record or update an emotional association."""
        try:
            import time

            with self._get_conn() as conn:
                # Check if entry exists for this (user, topic)
                existing = conn.execute(
                    "SELECT * FROM emotional_memory WHERE user_id = ? AND topic = ?",
                    (user_id, topic),
                ).fetchone()

                if existing:
                    # Update: increment count, update running average intensity
                    new_count = existing["occurrence_count"] + 1
                    # Running average: (old_avg * old_count + new_value) / new_count
                    new_intensity = (
                        (existing["intensity"] * existing["occurrence_count"] + intensity)
                        / new_count
                    )
                    conn.execute(
                        """UPDATE emotional_memory
                           SET tone = ?, intensity = ?, timestamp = ?,
                               occurrence_count = ?
                           WHERE user_id = ? AND topic = ?""",
                        (
                            tone.value,
                            new_intensity,
                            time.time(),
                            new_count,
                            user_id,
                            topic,
                        ),
                    )
                    return EmotionalMemoryEntry(
                        entry_id=existing["entry_id"],
                        user_id=user_id,
                        topic=topic,
                        tone=tone,
                        intensity=new_intensity,
                        timestamp=time.time(),
                        occurrence_count=new_count,
                    )
                else:
                    # Insert new
                    entry_id = str(uuid.uuid4())
                    conn.execute(
                        """INSERT INTO emotional_memory
                           (entry_id, user_id, topic, tone, intensity,
                            timestamp, occurrence_count)
                           VALUES (?, ?, ?, ?, ?, ?, 1)""",
                        (
                            entry_id,
                            user_id,
                            topic,
                            tone.value,
                            intensity,
                            time.time(),
                        ),
                    )
                    return EmotionalMemoryEntry(
                        entry_id=entry_id,
                        user_id=user_id,
                        topic=topic,
                        tone=tone,
                        intensity=intensity,
                        timestamp=time.time(),
                        occurrence_count=1,
                    )

        except Exception as exc:
            logger.error("memory.emotional.record_failed", error=str(exc))
            return None

    async def get_topic_tone(
        self,
        user_id: str,
        topic: str,
    ) -> EmotionalMemoryEntry | None:
        """Get the dominant tone for a topic."""
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM emotional_memory WHERE user_id = ? AND topic = ?",
                    (user_id, topic),
                ).fetchone()

                if not row:
                    return None

                return EmotionalMemoryEntry(
                    entry_id=row["entry_id"],
                    user_id=row["user_id"],
                    topic=row["topic"],
                    tone=EmotionalTone(row["tone"]),
                    intensity=row["intensity"],
                    timestamp=row["timestamp"],
                    occurrence_count=row["occurrence_count"],
                )
        except Exception as exc:
            logger.warning("memory.emotional.get_failed", error=str(exc))
            return None

    async def get_all(self, user_id: str) -> list[EmotionalMemoryEntry]:
        """Get all emotional entries for a user."""
        try:
            with self._get_conn() as conn:
                results = conn.execute(
                    "SELECT * FROM emotional_memory WHERE user_id = ? ORDER BY timestamp DESC",
                    (user_id,),
                ).fetchall()

                return [
                    EmotionalMemoryEntry(
                        entry_id=row["entry_id"],
                        user_id=row["user_id"],
                        topic=row["topic"],
                        tone=EmotionalTone(row["tone"]),
                        intensity=row["intensity"],
                        timestamp=row["timestamp"],
                        occurrence_count=row["occurrence_count"],
                    )
                    for row in results
                ]
        except Exception:
            return []

    async def health_check(self) -> bool:
        try:
            with self._get_conn() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        pass


class InMemoryEmotionalStore(EmotionalMemoryStore):
    """In-memory emotional store for testing."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], EmotionalMemoryEntry] = {}

    async def record(
        self,
        user_id: str,
        topic: str,
        tone: EmotionalTone,
        intensity: float = 0.5,
    ) -> EmotionalMemoryEntry | None:
        import time

        key = (user_id, topic)
        if key in self._store:
            existing = self._store[key]
            new_count = existing.occurrence_count + 1
            new_intensity = (
                (existing.intensity * existing.occurrence_count + intensity)
                / new_count
            )
            entry = EmotionalMemoryEntry(
                entry_id=existing.entry_id,
                user_id=user_id,
                topic=topic,
                tone=tone,
                intensity=new_intensity,
                timestamp=time.time(),
                occurrence_count=new_count,
            )
        else:
            entry = EmotionalMemoryEntry(
                entry_id=str(uuid.uuid4()),
                user_id=user_id,
                topic=topic,
                tone=tone,
                intensity=intensity,
                timestamp=time.time(),
                occurrence_count=1,
            )
        self._store[key] = entry
        return entry

    async def get_topic_tone(
        self,
        user_id: str,
        topic: str,
    ) -> EmotionalMemoryEntry | None:
        return self._store.get((user_id, topic))

    async def get_all(self, user_id: str) -> list[EmotionalMemoryEntry]:
        return [
            e for (uid, _) in self._store if uid == user_id
            for e in [self._store[(uid, _)]]  # type: ignore[index]
        ]

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        self._store.clear()


def create_emotional_store(backend: str | None = None) -> EmotionalMemoryStore:
    """Factory: create an emotional memory backend."""
    if backend == "in_memory":
        return InMemoryEmotionalStore()
    return SQLiteEmotionalStore()
