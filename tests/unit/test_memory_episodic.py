"""Tests for app.memory.episodic (Episodic Memory layer)."""

from __future__ import annotations

import pytest

from app.memory.base import EmotionalTone, EpisodicMemoryEntry
from app.memory.episodic import InMemoryEpisodicStore, SQLiteEpisodicStore


@pytest.fixture
def in_memory_store() -> InMemoryEpisodicStore:
    """Fresh in-memory episodic store."""
    return InMemoryEpisodicStore()


@pytest.fixture
def sqlite_store(tmp_path) -> SQLiteEpisodicStore:
    """Fresh SQLite episodic store with temp database."""
    db_path = tmp_path / "test_memory.db"
    return SQLiteEpisodicStore(db_path=db_path)


def _make_entry(
    user_id: str = "user1",
    summary: str = "Discussed pizza preferences",
    topics: list[str] | None = None,
) -> EpisodicMemoryEntry:
    """Create a test episodic entry."""
    import uuid

    return EpisodicMemoryEntry(
        entry_id=str(uuid.uuid4()),
        user_id=user_id,
        summary=summary,
        topics=topics or ["food"],
        emotional_tone=EmotionalTone.JOY,
    )


# ─── InMemoryEpisodicStore tests ─────────────────────────────────────────


class TestInMemoryEpisodicStore:
    """Test the in-memory episodic store."""

    @pytest.mark.asyncio
    async def test_add_and_count(self, in_memory_store: InMemoryEpisodicStore) -> None:
        """Adding entries should increment the count."""
        assert await in_memory_store.count("user1") == 0

        await in_memory_store.add(_make_entry(user_id="user1", summary="First"))
        assert await in_memory_store.count("user1") == 1

        await in_memory_store.add(_make_entry(user_id="user1", summary="Second"))
        assert await in_memory_store.count("user1") == 2

    @pytest.mark.asyncio
    async def test_search_by_keyword(self, in_memory_store: InMemoryEpisodicStore) -> None:
        """Search should find entries matching the query."""
        await in_memory_store.add(_make_entry(summary="I love pizza and pasta"))
        await in_memory_store.add(_make_entry(summary="Went hiking yesterday"))

        results = await in_memory_store.search("user1", "pizza")
        assert len(results) == 1
        assert "pizza" in results[0].summary

    @pytest.mark.asyncio
    async def test_search_by_topic(self, in_memory_store: InMemoryEpisodicStore) -> None:
        """Search should also match topics."""
        await in_memory_store.add(
            _make_entry(summary="Random text", topics=["cooking", "recipes"])
        )

        results = await in_memory_store.search("user1", "cooking")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_empty_for_no_match(
        self,
        in_memory_store: InMemoryEpisodicStore,
    ) -> None:
        """Search with no matches should return empty list."""
        await in_memory_store.add(_make_entry(summary="Hello world"))

        results = await in_memory_store.search("user1", "nonexistent")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_recent(self, in_memory_store: InMemoryEpisodicStore) -> None:
        """get_recent should return entries sorted by timestamp desc."""
        import time

        e1 = _make_entry(summary="First")
        e1.timestamp = 1000.0
        e2 = _make_entry(summary="Second")
        e2.timestamp = 2000.0

        await in_memory_store.add(e1)
        await in_memory_store.add(e2)

        recent = await in_memory_store.get_recent("user1", limit=10)
        assert len(recent) == 2
        assert recent[0].summary == "Second"  # most recent first
        assert recent[1].summary == "First"

    @pytest.mark.asyncio
    async def test_user_isolation(self, in_memory_store: InMemoryEpisodicStore) -> None:
        """Different users should have isolated memory."""
        await in_memory_store.add(_make_entry(user_id="user1", summary="User 1 memory"))
        await in_memory_store.add(_make_entry(user_id="user2", summary="User 2 memory"))

        assert await in_memory_store.count("user1") == 1
        assert await in_memory_store.count("user2") == 1

        results = await in_memory_store.search("user1", "memory")
        assert len(results) == 1
        assert "User 1" in results[0].summary

    @pytest.mark.asyncio
    async def test_health_check(self, in_memory_store: InMemoryEpisodicStore) -> None:
        """In-memory store should always be healthy."""
        assert await in_memory_store.health_check() is True


# ─── SQLiteEpisodicStore tests ───────────────────────────────────────────


class TestSQLiteEpisodicStore:
    """Test the SQLite episodic store."""

    @pytest.mark.asyncio
    async def test_add_and_count(self, sqlite_store: SQLiteEpisodicStore) -> None:
        """Adding entries should persist to SQLite."""
        await sqlite_store.add(_make_entry(user_id="user1", summary="First"))
        assert await sqlite_store.count("user1") == 1

    @pytest.mark.asyncio
    async def test_search_by_keyword(self, sqlite_store: SQLiteEpisodicStore) -> None:
        """FTS5 search should find keyword matches."""
        await sqlite_store.add(_make_entry(summary="I love pizza"))
        await sqlite_store.add(_make_entry(summary="Went to the beach"))

        results = await sqlite_store.search("user1", "pizza")
        assert len(results) == 1
        assert "pizza" in results[0].summary

    @pytest.mark.asyncio
    async def test_persistence_across_instances(
        self,
        sqlite_store: SQLiteEpisodicStore,
        tmp_path,
    ) -> None:
        """Data should persist when a new store instance opens the same DB."""
        await sqlite_store.add(_make_entry(user_id="user1", summary="Persistent memory"))

        # Create a new store pointing to the same database file
        new_store = SQLiteEpisodicStore(db_path=sqlite_store._db_path)
        count = await new_store.count("user1")
        assert count == 1

    @pytest.mark.asyncio
    async def test_get_recent(self, sqlite_store: SQLiteEpisodicStore) -> None:
        """get_recent should return entries sorted by timestamp."""
        e1 = _make_entry(summary="Older")
        e1.timestamp = 1000.0
        e2 = _make_entry(summary="Newer")
        e2.timestamp = 2000.0

        await sqlite_store.add(e1)
        await sqlite_store.add(e2)

        recent = await sqlite_store.get_recent("user1")
        assert len(recent) == 2
        assert recent[0].summary == "Newer"

    @pytest.mark.asyncio
    async def test_health_check(self, sqlite_store: SQLiteEpisodicStore) -> None:
        """SQLite store should report healthy."""
        assert await sqlite_store.health_check() is True

    @pytest.mark.asyncio
    async def test_user_isolation(self, sqlite_store: SQLiteEpisodicStore) -> None:
        """Different users should be isolated."""
        await sqlite_store.add(_make_entry(user_id="user1", summary="User 1"))
        await sqlite_store.add(_make_entry(user_id="user2", summary="User 2"))

        assert await sqlite_store.count("user1") == 1
        assert await sqlite_store.count("user2") == 1

    @pytest.mark.asyncio
    async def test_emotional_tone_persisted(
        self,
        sqlite_store: SQLiteEpisodicStore,
    ) -> None:
        """Emotional tone should survive round-trip through SQLite."""
        entry = _make_entry(summary="Happy conversation")
        entry.emotional_tone = EmotionalTone.JOY
        await sqlite_store.add(entry)

        results = await sqlite_store.search("user1", "Happy")
        assert len(results) == 1
        assert results[0].emotional_tone == EmotionalTone.JOY

    @pytest.mark.asyncio
    async def test_topics_persisted(self, sqlite_store: SQLiteEpisodicStore) -> None:
        """Topics list should survive round-trip through SQLite."""
        entry = _make_entry(summary="Topic test", topics=["python", "coding", "ai"])
        await sqlite_store.add(entry)

        results = await sqlite_store.search("user1", "Topic")
        assert len(results) == 1
        assert results[0].topics == ["python", "coding", "ai"]
