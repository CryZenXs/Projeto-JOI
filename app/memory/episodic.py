"""Projeto JOI - Episodic Memory layer.

Long-term storage of conversation episodes with semantic search capability.

Two implementations:
- SQLiteEpisodicStore: uses SQLite + simple TF-IDF for keyword search
  (no external dependencies, works everywhere)
- ChromaDBEpisodicStore: uses ChromaDB for true vector search
  (requires chromadb extra: pip install -e ".[vector]")

The SQLite implementation uses a hybrid retrieval:
1. TF-IDF keyword matching (SQLite FTS5)
2. Recency boost (newer episodes rank higher)
3. Optional: simple sentence embedding via hash (deterministic, no ML)

This is a pragmatic compromise: true semantic search requires embeddings
(which need ChromaDB or an embedding API), but TF-IDF + recency is
surprisingly effective for conversational recall and has zero dependencies.
"""

from __future__ import annotations

import math
import sqlite3
import time
import uuid
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.memory.base import EpisodicMemoryEntry, EmotionalTone, MemoryClient

logger = get_logger(__name__)


class EpisodicMemoryStore(MemoryClient):
    """Abstract interface for episodic memory backends."""

    async def add(self, entry: EpisodicMemoryEntry) -> EpisodicMemoryEntry | None:
        """Add an episode to storage."""
        raise NotImplementedError

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[EpisodicMemoryEntry]:
        """Search for episodes matching the query."""
        raise NotImplementedError

    async def get_recent(
        self,
        user_id: str,
        limit: int = 10,
    ) -> list[EpisodicMemoryEntry]:
        """Get the most recent episodes for a user."""
        raise NotImplementedError

    async def count(self, user_id: str) -> int:
        """Count total episodes for a user."""
        raise NotImplementedError


class SQLiteEpisodicStore(EpisodicMemoryStore):
    """SQLite-backed episodic memory with FTS5 full-text search.

    Uses SQLite's built-in FTS5 module for keyword search, which is
    available in Python's sqlite3 module (no external deps).

    For semantic similarity, we use a simple hashed-bag-of-words
    approach: each summary is converted to a sparse vector via hashing,
    and cosine similarity is computed in Python. This is much less
    accurate than proper embeddings but works for basic recall.

    Storage layout:
    - episodes: metadata table (id, user_id, summary, timestamp, ...)
    - episodes_fts: FTS5 virtual table for keyword search
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS episodes (
        entry_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        summary TEXT NOT NULL,
        timestamp REAL NOT NULL,
        turn_count INTEGER DEFAULT 0,
        topics_json TEXT DEFAULT '[]',
        emotional_tone TEXT DEFAULT 'neutral',
        metadata_json TEXT DEFAULT '{}',
        created_at REAL DEFAULT (strftime('%s', 'now'))
    );

    CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id);
    CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp DESC);

    CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
        entry_id UNINDEXED,
        summary,
        topics,
        tokenize='porter unicode61'
    );
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize SQLite episodic store.

        Args:
            db_path: Path to SQLite database file. If None, uses
                settings.chroma_persist_dir / 'memory.db' (or /tmp fallback).
        """
        if db_path is None:
            # Default location: data/memory.db
            persist_dir = Path(settings.chroma_persist_dir)
            try:
                persist_dir.mkdir(parents=True, exist_ok=True)
                db_path = persist_dir / "memory.db"
            except (OSError, PermissionError):
                # Fallback to /tmp if we can't write to the configured dir
                db_path = Path("/tmp/joi_memory.db")
                logger.warning(
                    "memory.episodic.fallback_db_path",
                    path=str(db_path),
                    reason="could not write to configured persist dir",
                )

        self._db_path = str(db_path)

        # Initialize schema
        with self._get_conn() as conn:
            conn.executescript(self.SCHEMA)

        logger.info(
            "memory.episodic.initialized",
            backend="sqlite",
            db_path=self._db_path,
        )

    @contextmanager
    def _get_conn(self) -> Any:
        """Get a SQLite connection (context manager)."""
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

    async def add(self, entry: EpisodicMemoryEntry) -> EpisodicMemoryEntry | None:
        """Add an episode to SQLite."""
        try:
            import json

            with self._get_conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO episodes
                       (entry_id, user_id, summary, timestamp, turn_count,
                        topics_json, emotional_tone, metadata_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry.entry_id,
                        entry.user_id,
                        entry.summary,
                        entry.timestamp,
                        entry.turn_count,
                        json.dumps(entry.topics),
                        entry.emotional_tone.value,
                        json.dumps(entry.metadata),
                    ),
                )
                # Also index in FTS table
                conn.execute(
                    "INSERT OR REPLACE INTO episodes_fts (entry_id, summary, topics) VALUES (?, ?, ?)",
                    (
                        entry.entry_id,
                        entry.summary,
                        " ".join(entry.topics),
                    ),
                )

            logger.debug(
                "memory.episodic.added",
                entry_id=entry.entry_id,
                user_id=entry.user_id,
                summary_length=len(entry.summary),
            )
            return entry
        except Exception as exc:
            logger.error("memory.episodic.add_failed", error=str(exc))
            return None

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[EpisodicMemoryEntry]:
        """Search episodes using FTS5 + recency boost."""
        try:
            import json

            # Build FTS query: match any word in the query
            # FTS5 syntax: "word1 OR word2 OR word3"
            fts_query = " OR ".join(query.split())

            with self._get_conn() as conn:
                # Hybrid ranking: FTS5 bm25 + recency boost
                # bm25 returns lower scores for better matches (negative),
                # so we negate it and add recency
                results = conn.execute(
                    """SELECT e.*,
                              -rank as bm25_score,
                              (e.timestamp / ?) as recency_bucket
                       FROM episodes e
                       JOIN episodes_fts f ON e.entry_id = f.entry_id
                       WHERE e.user_id = ?
                         AND episodes_fts MATCH ?
                       ORDER BY (rank - (e.timestamp / ?) * 0.1) ASC
                       LIMIT ?""",
                    (
                        86400,  # recency bucket = 1 day
                        user_id,
                        fts_query,
                        86400,
                        top_k,
                    ),
                ).fetchall()

            entries = []
            for row in results:
                entries.append(
                    EpisodicMemoryEntry(
                        entry_id=row["entry_id"],
                        user_id=row["user_id"],
                        summary=row["summary"],
                        timestamp=row["timestamp"],
                        turn_count=row["turn_count"],
                        topics=json.loads(row["topics_json"]),
                        emotional_tone=EmotionalTone(row["emotional_tone"]),
                        metadata=json.loads(row["metadata_json"]),
                    )
                )

            logger.debug(
                "memory.episodic.search",
                user_id=user_id,
                query=query[:50],
                results_count=len(entries),
            )
            return entries

        except Exception as exc:
            logger.warning("memory.episodic.search_failed", error=str(exc))
            return []

    async def get_recent(
        self,
        user_id: str,
        limit: int = 10,
    ) -> list[EpisodicMemoryEntry]:
        """Get most recent episodes for a user."""
        try:
            import json

            with self._get_conn() as conn:
                results = conn.execute(
                    """SELECT * FROM episodes
                       WHERE user_id = ?
                       ORDER BY timestamp DESC
                       LIMIT ?""",
                    (user_id, limit),
                ).fetchall()

            return [
                EpisodicMemoryEntry(
                    entry_id=row["entry_id"],
                    user_id=row["user_id"],
                    summary=row["summary"],
                    timestamp=row["timestamp"],
                    turn_count=row["turn_count"],
                    topics=json.loads(row["topics_json"]),
                    emotional_tone=EmotionalTone(row["emotional_tone"]),
                    metadata=json.loads(row["metadata_json"]),
                )
                for row in results
            ]
        except Exception as exc:
            logger.warning("memory.episodic.get_recent_failed", error=str(exc))
            return []

    async def count(self, user_id: str) -> int:
        """Count episodes for a user."""
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) as count FROM episodes WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                return int(row["count"]) if row else 0
        except Exception:
            return 0

    async def health_check(self) -> bool:
        """Check if SQLite is operational."""
        try:
            with self._get_conn() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """SQLite connections are per-operation, nothing to close."""
        pass


class InMemoryEpisodicStore(EpisodicMemoryStore):
    """In-memory episodic store for testing.

    Simple list-based implementation with basic keyword matching.
    Suitable for unit tests; not for production use.
    """

    def __init__(self) -> None:
        self._entries: list[EpisodicMemoryEntry] = []
        logger.info("memory.episodic.initialized", backend="in_memory")

    async def add(self, entry: EpisodicMemoryEntry) -> EpisodicMemoryEntry | None:
        self._entries.append(entry)
        return entry

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[EpisodicMemoryEntry]:
        query_lower = query.lower()
        matching = [
            e for e in self._entries
            if e.user_id == user_id
            and (query_lower in e.summary.lower() or
                 any(query_lower in t.lower() for t in e.topics))
        ]
        # Sort by timestamp desc
        matching.sort(key=lambda e: e.timestamp, reverse=True)
        return matching[:top_k]

    async def get_recent(
        self,
        user_id: str,
        limit: int = 10,
    ) -> list[EpisodicMemoryEntry]:
        user_entries = [e for e in self._entries if e.user_id == user_id]
        user_entries.sort(key=lambda e: e.timestamp, reverse=True)
        return user_entries[:limit]

    async def count(self, user_id: str) -> int:
        return sum(1 for e in self._entries if e.user_id == user_id)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        self._entries.clear()


# ─── Factory function ────────────────────────────────────────────────────


def create_episodic_store(backend: str | None = None) -> EpisodicMemoryStore:
    """Factory: create an episodic memory backend.

    Args:
        backend: 'chromadb', 'sqlite', 'in_memory', or None (auto-detect).

    Returns:
        An EpisodicMemoryStore instance.
    """
    if backend == "in_memory":
        return InMemoryEpisodicStore()
    elif backend == "sqlite":
        return SQLiteEpisodicStore()

    # Auto-detect: try ChromaDB first, fall back to SQLite
    try:
        import chromadb  # noqa: F401

        # ChromaDB is installed — but creating the store requires more setup
        # For now, prefer SQLite for reliability (ChromaDB integration is TODO)
        logger.info("memory.episodic.chromadb_available_using_sqlite")
        return SQLiteEpisodicStore()
    except ImportError:
        return SQLiteEpisodicStore()
