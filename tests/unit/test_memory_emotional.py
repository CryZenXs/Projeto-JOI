"""Tests for app.memory.emotional (Emotional Memory layer)."""

from __future__ import annotations

import pytest

from app.memory.base import EmotionalTone
from app.memory.emotional import InMemoryEmotionalStore, SQLiteEmotionalStore


@pytest.fixture
def in_memory_store() -> InMemoryEmotionalStore:
    return InMemoryEmotionalStore()


@pytest.fixture
def sqlite_store(tmp_path) -> SQLiteEmotionalStore:
    return SQLiteEmotionalStore(db_path=tmp_path / "test.db")


# ─── InMemoryEmotionalStore ──────────────────────────────────────────────


class TestInMemoryEmotionalStore:
    @pytest.mark.asyncio
    async def test_record_new_emotion(self, in_memory_store: InMemoryEmotionalStore) -> None:
        """Recording a new emotion should create an entry."""
        entry = await in_memory_store.record(
            user_id="user1",
            topic="work",
            tone=EmotionalTone.SADNESS,
            intensity=0.7,
        )

        assert entry is not None
        assert entry.topic == "work"
        assert entry.tone == EmotionalTone.SADNESS
        assert entry.intensity == 0.7
        assert entry.occurrence_count == 1

    @pytest.mark.asyncio
    async def test_record_increments_count(
        self,
        in_memory_store: InMemoryEmotionalStore,
    ) -> None:
        """Recording the same topic+tone should increment occurrence_count."""
        await in_memory_store.record("user1", "work", EmotionalTone.SADNESS, 0.8)
        entry = await in_memory_store.record("user1", "work", EmotionalTone.SADNESS, 0.6)

        assert entry is not None
        assert entry.occurrence_count == 2
        # Intensity should be the average: (0.8 + 0.6) / 2 = 0.7
        assert abs(entry.intensity - 0.7) < 0.01

    @pytest.mark.asyncio
    async def test_get_topic_tone(
        self,
        in_memory_store: InMemoryEmotionalStore,
    ) -> None:
        """get_topic_tone should return the recorded entry."""
        await in_memory_store.record("user1", "family", EmotionalTone.JOY, 0.9)

        entry = await in_memory_store.get_topic_tone("user1", "family")
        assert entry is not None
        assert entry.tone == EmotionalTone.JOY

    @pytest.mark.asyncio
    async def test_get_topic_tone_nonexistent(
        self,
        in_memory_store: InMemoryEmotionalStore,
    ) -> None:
        """get_topic_tone for unknown topic should return None."""
        entry = await in_memory_store.get_topic_tone("user1", "nonexistent")
        assert entry is None

    @pytest.mark.asyncio
    async def test_get_all(self, in_memory_store: InMemoryEmotionalStore) -> None:
        """get_all should return all entries for a user."""
        await in_memory_store.record("user1", "work", EmotionalTone.SADNESS)
        await in_memory_store.record("user1", "family", EmotionalTone.JOY)
        await in_memory_store.record("user2", "work", EmotionalTone.ANGER)

        entries = await in_memory_store.get_all("user1")
        assert len(entries) == 2  # only user1's entries

    @pytest.mark.asyncio
    async def test_health_check(self, in_memory_store: InMemoryEmotionalStore) -> None:
        assert await in_memory_store.health_check() is True


# ─── SQLiteEmotionalStore ────────────────────────────────────────────────


class TestSQLiteEmotionalStore:
    @pytest.mark.asyncio
    async def test_record_and_get(self, sqlite_store: SQLiteEmotionalStore) -> None:
        """Record should persist and be retrievable."""
        await sqlite_store.record("user1", "work", EmotionalTone.SADNESS, 0.8)

        entry = await sqlite_store.get_topic_tone("user1", "work")
        assert entry is not None
        assert entry.tone == EmotionalTone.SADNESS
        assert entry.intensity == 0.8
        assert entry.occurrence_count == 1

    @pytest.mark.asyncio
    async def test_record_increments_count(
        self,
        sqlite_store: SQLiteEmotionalStore,
    ) -> None:
        """Multiple records for same topic should increment count."""
        await sqlite_store.record("user1", "work", EmotionalTone.SADNESS, 0.8)
        await sqlite_store.record("user1", "work", EmotionalTone.SADNESS, 0.6)

        entry = await sqlite_store.get_topic_tone("user1", "work")
        assert entry is not None
        assert entry.occurrence_count == 2
        assert abs(entry.intensity - 0.7) < 0.01

    @pytest.mark.asyncio
    async def test_persistence(self, sqlite_store: SQLiteEmotionalStore) -> None:
        """Emotional entries should persist across instances."""
        await sqlite_store.record("user1", "family", EmotionalTone.JOY, 0.9)

        new_store = SQLiteEmotionalStore(db_path=sqlite_store._db_path)
        entry = await new_store.get_topic_tone("user1", "family")
        assert entry is not None
        assert entry.tone == EmotionalTone.JOY

    @pytest.mark.asyncio
    async def test_get_all(self, sqlite_store: SQLiteEmotionalStore) -> None:
        await sqlite_store.record("user1", "work", EmotionalTone.SADNESS)
        await sqlite_store.record("user1", "family", EmotionalTone.JOY)

        entries = await sqlite_store.get_all("user1")
        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_health_check(self, sqlite_store: SQLiteEmotionalStore) -> None:
        assert await sqlite_store.health_check() is True

    @pytest.mark.asyncio
    async def test_user_isolation(self, sqlite_store: SQLiteEmotionalStore) -> None:
        """Different users should have isolated emotional memory."""
        await sqlite_store.record("user1", "work", EmotionalTone.SADNESS)
        await sqlite_store.record("user2", "work", EmotionalTone.JOY)

        e1 = await sqlite_store.get_topic_tone("user1", "work")
        e2 = await sqlite_store.get_topic_tone("user2", "work")

        assert e1 is not None
        assert e2 is not None
        assert e1.tone == EmotionalTone.SADNESS
        assert e2.tone == EmotionalTone.JOY
