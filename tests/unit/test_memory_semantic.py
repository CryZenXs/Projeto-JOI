"""Tests for app.memory.semantic (Semantic Memory layer)."""

from __future__ import annotations

import pytest

from app.memory.base import SemanticMemoryEntry
from app.memory.semantic import InMemorySemanticStore, SQLiteSemanticStore


@pytest.fixture
def in_memory_store() -> InMemorySemanticStore:
    return InMemorySemanticStore()


@pytest.fixture
def sqlite_store(tmp_path) -> SQLiteSemanticStore:
    return SQLiteSemanticStore(db_path=tmp_path / "test.db")


def _make_fact(
    user_id: str = "user1",
    key: str = "name",
    value: str = "Alice",
) -> SemanticMemoryEntry:
    import uuid
    return SemanticMemoryEntry(
        entry_id=str(uuid.uuid4()),
        user_id=user_id,
        key=key,
        value=value,
        confidence=0.9,
        source="user_statement",
    )


# ─── InMemorySemanticStore ───────────────────────────────────────────────


class TestInMemorySemanticStore:
    @pytest.mark.asyncio
    async def test_upsert_and_get(self, in_memory_store: InMemorySemanticStore) -> None:
        """upsert should store a fact that can be retrieved."""
        await in_memory_store.upsert(_make_fact(key="name", value="Alice"))

        facts = await in_memory_store.get("user1")
        assert len(facts) == 1
        assert facts[0].key == "name"
        assert facts[0].value == "Alice"

    @pytest.mark.asyncio
    async def test_upsert_updates_existing(
        self,
        in_memory_store: InMemorySemanticStore,
    ) -> None:
        """Upserting the same key should update, not duplicate."""
        await in_memory_store.upsert(_make_fact(key="location", value="Sao Paulo"))
        await in_memory_store.upsert(_make_fact(key="location", value="Rio de Janeiro"))

        facts = await in_memory_store.get("user1")
        assert len(facts) == 1  # still only one fact for this key
        assert facts[0].value == "Rio de Janeiro"

    @pytest.mark.asyncio
    async def test_get_filtered_by_keys(
        self,
        in_memory_store: InMemorySemanticStore,
    ) -> None:
        """get with keys filter should only return matching facts."""
        await in_memory_store.upsert(_make_fact(key="name", value="Alice"))
        await in_memory_store.upsert(_make_fact(key="age", value="30"))
        await in_memory_store.upsert(_make_fact(key="location", value="Sao Paulo"))

        facts = await in_memory_store.get("user1", keys=["name", "location"])
        assert len(facts) == 2
        keys = {f.key for f in facts}
        assert keys == {"name", "location"}

    @pytest.mark.asyncio
    async def test_delete(self, in_memory_store: InMemorySemanticStore) -> None:
        """delete should remove a fact."""
        await in_memory_store.upsert(_make_fact(key="name", value="Alice"))
        assert await in_memory_store.count("user1") == 1

        deleted = await in_memory_store.delete("user1", "name")
        assert deleted is True
        assert await in_memory_store.count("user1") == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(
        self,
        in_memory_store: InMemorySemanticStore,
    ) -> None:
        """Deleting a non-existent fact should return False."""
        deleted = await in_memory_store.delete("user1", "nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_user_isolation(
        self,
        in_memory_store: InMemorySemanticStore,
    ) -> None:
        """Different users should have isolated facts."""
        await in_memory_store.upsert(_make_fact(user_id="user1", key="name", value="Alice"))
        await in_memory_store.upsert(_make_fact(user_id="user2", key="name", value="Bob"))

        assert await in_memory_store.count("user1") == 1
        assert await in_memory_store.count("user2") == 1

        facts1 = await in_memory_store.get("user1")
        assert facts1[0].value == "Alice"

        facts2 = await in_memory_store.get("user2")
        assert facts2[0].value == "Bob"


# ─── SQLiteSemanticStore ─────────────────────────────────────────────────


class TestSQLiteSemanticStore:
    @pytest.mark.asyncio
    async def test_upsert_and_get(self, sqlite_store: SQLiteSemanticStore) -> None:
        await sqlite_store.upsert(_make_fact(key="name", value="Alice"))
        facts = await sqlite_store.get("user1")
        assert len(facts) == 1
        assert facts[0].value == "Alice"

    @pytest.mark.asyncio
    async def test_upsert_updates_existing(
        self,
        sqlite_store: SQLiteSemanticStore,
    ) -> None:
        await sqlite_store.upsert(_make_fact(key="location", value="Sao Paulo"))
        await sqlite_store.upsert(_make_fact(key="location", value="Rio"))

        facts = await sqlite_store.get("user1")
        assert len(facts) == 1
        assert facts[0].value == "Rio"

    @pytest.mark.asyncio
    async def test_persistence(
        self,
        sqlite_store: SQLiteSemanticStore,
    ) -> None:
        """Facts should persist across store instances."""
        await sqlite_store.upsert(_make_fact(key="name", value="Alice"))

        new_store = SQLiteSemanticStore(db_path=sqlite_store._db_path)
        facts = await new_store.get("user1")
        assert len(facts) == 1
        assert facts[0].value == "Alice"

    @pytest.mark.asyncio
    async def test_delete(self, sqlite_store: SQLiteSemanticStore) -> None:
        await sqlite_store.upsert(_make_fact(key="name", value="Alice"))
        deleted = await sqlite_store.delete("user1", "name")
        assert deleted is True
        assert await sqlite_store.count("user1") == 0

    @pytest.mark.asyncio
    async def test_health_check(self, sqlite_store: SQLiteSemanticStore) -> None:
        assert await sqlite_store.health_check() is True
