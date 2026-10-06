"""Tests for app.memory.working (Working Memory layer)."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.memory.working import InMemoryWorkingMemory


@pytest.fixture
def working_memory() -> InMemoryWorkingMemory:
    """Fresh in-memory working memory for each test."""
    return InMemoryWorkingMemory(ttl_seconds=60, max_turns_per_user=10)


class TestInMemoryWorkingMemory:
    """Test the InMemoryWorkingMemory implementation."""

    @pytest.mark.asyncio
    async def test_add_turn_returns_entry(self, working_memory: InMemoryWorkingMemory) -> None:
        """add_turn should return a WorkingMemoryEntry."""
        entry = await working_memory.add_turn(
            user_id="user1",
            role="user",
            content="Hello",
        )

        assert entry.user_id == "user1"
        assert entry.role == "user"
        assert entry.content == "Hello"
        assert entry.turn_id == 1
        assert entry.timestamp > 0

    @pytest.mark.asyncio
    async def test_turn_ids_increment_per_user(
        self,
        working_memory: InMemoryWorkingMemory,
    ) -> None:
        """Each user should have their own incrementing turn counter."""
        e1 = await working_memory.add_turn("user1", "user", "msg1")
        e2 = await working_memory.add_turn("user1", "user", "msg2")
        e3 = await working_memory.add_turn("user2", "user", "other user")

        assert e1.turn_id == 1
        assert e2.turn_id == 2
        assert e3.turn_id == 1  # different user, separate counter

    @pytest.mark.asyncio
    async def test_get_context_returns_recent_entries(
        self,
        working_memory: InMemoryWorkingMemory,
    ) -> None:
        """get_context should return entries in chronological order."""
        for i in range(5):
            await working_memory.add_turn("user1", "user", f"msg{i}")

        context = await working_memory.get_context("user1", max_entries=3)

        assert len(context) == 3
        # Should be the last 3 entries, in chronological order
        assert context[0].content == "msg2"
        assert context[1].content == "msg3"
        assert context[2].content == "msg4"

    @pytest.mark.asyncio
    async def test_get_context_empty_for_unknown_user(
        self,
        working_memory: InMemoryWorkingMemory,
    ) -> None:
        """Unknown user should return empty list."""
        context = await working_memory.get_context("unknown")
        assert context == []

    @pytest.mark.asyncio
    async def test_max_turns_enforcement(
        self,
        working_memory: InMemoryWorkingMemory,
    ) -> None:
        """Older entries should be evicted when max_turns is exceeded."""
        wm = InMemoryWorkingMemory(ttl_seconds=60, max_turns_per_user=3)

        for i in range(5):
            await wm.add_turn("user1", "user", f"msg{i}")

        context = await wm.get_context("user1", max_entries=10)
        # Should only keep the last 3
        assert len(context) == 3
        assert context[0].content == "msg2"
        assert context[2].content == "msg4"

    @pytest.mark.asyncio
    async def test_clear_removes_all_entries(
        self,
        working_memory: InMemoryWorkingMemory,
    ) -> None:
        """clear should remove all entries for a user."""
        await working_memory.add_turn("user1", "user", "msg1")
        await working_memory.add_turn("user1", "user", "msg2")

        count = await working_memory.clear("user1")
        assert count == 2

        context = await working_memory.get_context("user1")
        assert context == []

    @pytest.mark.asyncio
    async def test_ttl_expiration(
        self,
        working_memory: InMemoryWorkingMemory,
    ) -> None:
        """Entries older than TTL should be filtered out on access."""
        wm = InMemoryWorkingMemory(ttl_seconds=0.1, max_turns_per_user=10)

        await wm.add_turn("user1", "user", "old message")
        # Wait for TTL to expire
        await asyncio.sleep(0.15)

        context = await wm.get_context("user1")
        assert context == []  # expired entry removed

    @pytest.mark.asyncio
    async def test_health_check_always_true(
        self,
        working_memory: InMemoryWorkingMemory,
    ) -> None:
        """In-memory backend should always be healthy."""
        assert await working_memory.health_check() is True

    @pytest.mark.asyncio
    async def test_metadata_preserved(
        self,
        working_memory: InMemoryWorkingMemory,
    ) -> None:
        """Metadata should be stored and retrievable."""
        entry = await working_memory.add_turn(
            user_id="user1",
            role="user",
            content="Hello",
            metadata={"timestamp": "2024-01-01", "source": "web"},
        )

        assert entry.metadata["timestamp"] == "2024-01-01"
        assert entry.metadata["source"] == "web"
