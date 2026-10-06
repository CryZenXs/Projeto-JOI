"""Tests for app.memory.manager (MemoryManager coordination layer)."""

from __future__ import annotations

import pytest

from app.memory.base import EmotionalTone
from app.memory.emotional import InMemoryEmotionalStore
from app.memory.episodic import InMemoryEpisodicStore
from app.memory.manager import MemoryManager
from app.memory.semantic import InMemorySemanticStore
from app.memory.working import InMemoryWorkingMemory


@pytest.fixture
def manager() -> MemoryManager:
    """Fresh MemoryManager with all in-memory backends."""
    return MemoryManager(
        working=InMemoryWorkingMemory(ttl_seconds=60),
        episodic=InMemoryEpisodicStore(),
        semantic=InMemorySemanticStore(),
        emotional=InMemoryEmotionalStore(),
    )


class TestMemoryManager:
    """Test the coordinated memory operations."""

    @pytest.mark.asyncio
    async def test_add_turn_stores_in_working_memory(
        self,
        manager: MemoryManager,
    ) -> None:
        """add_turn should store the turn in working memory."""
        entry = await manager.add_turn("user1", "user", "Hello")

        assert entry.role == "user"
        assert entry.content == "Hello"

        context = await manager.get_context("user1")
        assert len(context) == 1
        assert context[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_add_turn_extracts_facts(
        self,
        manager: MemoryManager,
    ) -> None:
        """User messages should trigger fact extraction."""
        await manager.add_turn(
            "user1",
            "user",
            "My name is Alice and I live in Sao Paulo",
        )

        facts = await manager.get_facts("user1")
        # Should have extracted at least name and location
        keys = {f.key for f in facts}
        assert "name" in keys
        assert "location" in keys

        name_fact = next(f for f in facts if f.key == "name")
        assert "Alice" in name_fact.value

    @pytest.mark.asyncio
    async def test_add_turn_does_not_extract_for_assistant(
        self,
        manager: MemoryManager,
    ) -> None:
        """Assistant messages should NOT trigger fact extraction."""
        await manager.add_turn(
            "user1",
            "assistant",
            "My name is JOI and I live in the cloud",
        )

        facts = await manager.get_facts("user1")
        # Should be empty because assistant messages don't extract facts
        assert facts == []

    @pytest.mark.asyncio
    async def test_consolidate_episode_stores_in_episodic(
        self,
        manager: MemoryManager,
    ) -> None:
        """consolidate_episode should persist to episodic memory."""
        entry = await manager.consolidate_episode(
            user_id="user1",
            summary="User discussed their love for pizza",
            topics=["food", "pizza"],
            emotional_tone=EmotionalTone.JOY,
        )

        assert entry is not None
        assert entry.summary == "User discussed their love for pizza"
        assert entry.topics == ["food", "pizza"]
        assert entry.emotional_tone == EmotionalTone.JOY

    @pytest.mark.asyncio
    async def test_recall_searches_episodic(
        self,
        manager: MemoryManager,
    ) -> None:
        """recall should find matching episodes."""
        await manager.consolidate_episode(
            user_id="user1",
            summary="Talked about pizza preferences",
            topics=["food"],
        )
        await manager.consolidate_episode(
            user_id="user1",
            summary="Discussed machine learning",
            topics=["tech"],
        )

        results = await manager.recall("user1", "pizza")
        assert len(results) >= 1
        assert "pizza" in results[0].summary

    @pytest.mark.asyncio
    async def test_upsert_and_get_fact(
        self,
        manager: MemoryManager,
    ) -> None:
        """upsert_fact should store and update facts."""
        await manager.upsert_fact("user1", "name", "Alice")
        await manager.upsert_fact("user1", "age", "30")

        facts = await manager.get_facts("user1")
        assert len(facts) == 2

        # Update existing fact
        await manager.upsert_fact("user1", "name", "Bob")
        facts = await manager.get_facts("user1")
        name_fact = next(f for f in facts if f.key == "name")
        assert name_fact.value == "Bob"
        assert len(facts) == 2  # still only 2 facts, not 3

    @pytest.mark.asyncio
    async def test_record_emotion(
        self,
        manager: MemoryManager,
    ) -> None:
        """record_emotion should store emotional context."""
        entry = await manager.record_emotion(
            user_id="user1",
            topic="work",
            tone=EmotionalTone.SADNESS,
            intensity=0.8,
        )

        assert entry is not None
        assert entry.tone == EmotionalTone.SADNESS
        assert entry.intensity == 0.8
        assert entry.occurrence_count == 1

        # Record again — should increment count
        entry2 = await manager.record_emotion(
            user_id="user1",
            topic="work",
            tone=EmotionalTone.SADNESS,
            intensity=0.6,
        )
        assert entry2 is not None
        assert entry2.occurrence_count == 2

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(
        self,
        manager: MemoryManager,
    ) -> None:
        """All in-memory backends should report healthy."""
        health = await manager.health_check()

        assert health["working"] is True
        assert health["episodic"] is True
        assert health["semantic"] is True
        assert health["emotional"] is True

    @pytest.mark.asyncio
    async def test_close_does_not_raise(
        self,
        manager: MemoryManager,
    ) -> None:
        """close should clean up without raising."""
        await manager.add_turn("user1", "user", "test")
        await manager.close()  # should not raise


class TestFactExtraction:
    """Test the regex-based fact extraction specifically."""

    @pytest.mark.asyncio
    async def test_extract_name(self, manager: MemoryManager) -> None:
        await manager.add_turn("user1", "user", "Hi, my name is Alice")
        facts = await manager.get_facts("user1", keys=["name"])
        assert len(facts) == 1
        assert "Alice" in facts[0].value

    @pytest.mark.asyncio
    async def test_extract_location(self, manager: MemoryManager) -> None:
        await manager.add_turn("user1", "user", "I live in Sao Paulo")
        facts = await manager.get_facts("user1", keys=["location"])
        assert len(facts) == 1
        assert "Sao Paulo" in facts[0].value

    @pytest.mark.asyncio
    async def test_extract_likes(self, manager: MemoryManager) -> None:
        await manager.add_turn("user1", "user", "I love pizza and pasta")
        facts = await manager.get_facts("user1", keys=["likes"])
        assert len(facts) >= 1

    @pytest.mark.asyncio
    async def test_extract_profession(self, manager: MemoryManager) -> None:
        await manager.add_turn("user1", "user", "I work as a developer")
        facts = await manager.get_facts("user1", keys=["profession"])
        assert len(facts) == 1
        assert "developer" in facts[0].value.lower()

    @pytest.mark.asyncio
    async def test_no_extraction_for_irrelevant_text(
        self,
        manager: MemoryManager,
    ) -> None:
        """Text without recognizable patterns should not extract facts."""
        await manager.add_turn("user1", "user", "The weather is nice today")
        facts = await manager.get_facts("user1")
        assert facts == []

    @pytest.mark.asyncio
    async def test_multiple_extractions_in_one_message(
        self,
        manager: MemoryManager,
    ) -> None:
        """A single message with multiple patterns should extract multiple facts."""
        await manager.add_turn(
            "user1",
            "user",
            "My name is Alice and I live in Sao Paulo. I love pizza.",
        )
        facts = await manager.get_facts("user1")
        keys = {f.key for f in facts}
        assert "name" in keys
        assert "location" in keys
        assert "likes" in keys
