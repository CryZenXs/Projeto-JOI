"""Projeto JOI - Semantic Memory layer.

Structured key-value storage for facts about the user.
Unlike episodic memory (which appends), semantic memory updates in place
— if the user moves to a new city, the old 'location' fact is replaced.

Backend: SQLite (no external dependencies, works everywhere).
Postgres support is planned but not yet implemented.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.memory.base import MemoryClient, SemanticMemoryEntry

logger = get_logger(__name__)


class SemanticMemoryStore(MemoryClient):
    """Abstract interface for semantic memory backends."""

    async def upsert(self, entry: SemanticMemoryEntry) -> SemanticMemoryEntry | None:
        """Insert or update a fact."""
        raise NotImplementedError

    async def get(
        self,
        user_id: str,
        keys: list[str] | None = None,
    ) -> list[SemanticMemoryEntry]:
        """Retrieve facts for a user, optionally filtered by keys."""
        raise NotImplementedError

    async def delete(self, user_id: str, key: str) -> bool:
        """Delete a specific fact. Returns True if deleted."""
        raise NotImplementedError

    async def count(self, user_id: str) -> int:
        """Count facts for a user."""
        raise NotImplementedError


class SQLiteSemanticStore(SemanticMemoryStore):
    """SQLite-backed semantic memory.

    Uses a simple table with UNIQUE constraint on (user_id, key) to
    enforce one fact per key per user. Upsert via INSERT OR REPLACE.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS semantic_memory (
        entry_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        confidence REAL DEFAULT 1.0,
        timestamp REAL NOT NULL,
        source TEXT DEFAULT 'user_statement',
        metadata_json TEXT DEFAULT '{}',
        UNIQUE(user_id, key)
    );

    CREATE INDEX IF NOT EXISTS idx_semantic_user ON semantic_memory(user_id);
    CREATE INDEX IF NOT EXISTS idx_semantic_user_key ON semantic_memory(user_id, key);
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
            "memory.semantic.initialized",
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

    async def upsert(self, entry: SemanticMemoryEntry) -> SemanticMemoryEntry | None:
        """Insert or update a fact."""
        try:
            with self._get_conn() as conn:
                # Check if entry exists (to preserve entry_id or update)
                existing = conn.execute(
                    "SELECT entry_id FROM semantic_memory WHERE user_id = ? AND key = ?",
                    (entry.user_id, entry.key),
                ).fetchone()

                if existing:
                    # Update existing entry
                    conn.execute(
                        """UPDATE semantic_memory
                           SET value = ?, confidence = ?, timestamp = ?,
                               source = ?, metadata_json = ?
                           WHERE user_id = ? AND key = ?""",
                        (
                            entry.value,
                            entry.confidence,
                            entry.timestamp,
                            entry.source,
                            json.dumps(entry.metadata),
                            entry.user_id,
                            entry.key,
                        ),
                    )
                else:
                    # Insert new entry
                    conn.execute(
                        """INSERT OR REPLACE INTO semantic_memory
                           (entry_id, user_id, key, value, confidence,
                            timestamp, source, metadata_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            entry.entry_id,
                            entry.user_id,
                            entry.key,
                            entry.value,
                            entry.confidence,
                            entry.timestamp,
                            entry.source,
                            json.dumps(entry.metadata),
                        ),
                    )

            logger.debug(
                "memory.semantic.upserted",
                user_id=entry.user_id,
                key=entry.key,
                value_length=len(entry.value),
            )
            return entry
        except Exception as exc:
            logger.error("memory.semantic.upsert_failed", error=str(exc))
            return None

    async def get(
        self,
        user_id: str,
        keys: list[str] | None = None,
    ) -> list[SemanticMemoryEntry]:
        """Retrieve facts for a user."""
        try:
            with self._get_conn() as conn:
                if keys:
                    placeholders = ",".join("?" * len(keys))
                    results = conn.execute(
                        f"""SELECT * FROM semantic_memory
                            WHERE user_id = ? AND key IN ({placeholders})
                            ORDER BY timestamp DESC""",
                        (user_id, *keys),
                    ).fetchall()
                else:
                    results = conn.execute(
                        "SELECT * FROM semantic_memory WHERE user_id = ? ORDER BY timestamp DESC",
                        (user_id,),
                    ).fetchall()

            return [
                SemanticMemoryEntry(
                    entry_id=row["entry_id"],
                    user_id=row["user_id"],
                    key=row["key"],
                    value=row["value"],
                    confidence=row["confidence"],
                    timestamp=row["timestamp"],
                    source=row["source"],
                    metadata=json.loads(row["metadata_json"]),
                )
                for row in results
            ]
        except Exception as exc:
            logger.warning("memory.semantic.get_failed", error=str(exc))
            return []

    async def delete(self, user_id: str, key: str) -> bool:
        """Delete a fact."""
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "DELETE FROM semantic_memory WHERE user_id = ? AND key = ?",
                    (user_id, key),
                )
                return cursor.rowcount > 0
        except Exception as exc:
            logger.warning("memory.semantic.delete_failed", error=str(exc))
            return False

    async def count(self, user_id: str) -> int:
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) as count FROM semantic_memory WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                return int(row["count"]) if row else 0
        except Exception:
            return 0

    async def health_check(self) -> bool:
        try:
            with self._get_conn() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        pass


class InMemorySemanticStore(SemanticMemoryStore):
    """In-memory semantic store for testing."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], SemanticMemoryEntry] = {}

    async def upsert(self, entry: SemanticMemoryEntry) -> SemanticMemoryEntry | None:
        self._store[(entry.user_id, entry.key)] = entry
        return entry

    async def get(
        self,
        user_id: str,
        keys: list[str] | None = None,
    ) -> list[SemanticMemoryEntry]:
        results = [
            e for (uid, _), e in self._store.items() if uid == user_id
        ]
        if keys:
            results = [e for e in results if e.key in keys]
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results

    async def delete(self, user_id: str, key: str) -> bool:
        if (user_id, key) in self._store:
            del self._store[(user_id, key)]
            return True
        return False

    async def count(self, user_id: str) -> int:
        return sum(1 for (uid, _) in self._store if uid == user_id)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        self._store.clear()


def create_semantic_store(backend: str | None = None) -> SemanticMemoryStore:
    """Factory: create a semantic memory backend."""
    if backend == "in_memory":
        return InMemorySemanticStore()
    return SQLiteSemanticStore()
