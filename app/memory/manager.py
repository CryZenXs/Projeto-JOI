"""Projeto JOI - Memory Manager.

Coordinates all four memory layers and provides a unified API for
the rest of the application (chat endpoint, persona engine).

The manager handles:
- Routing operations to the appropriate backend
- Graceful degradation (if a backend is unavailable, return empty results)
- Fact extraction from user messages (simple keyword-based, LLM-enhanced later)
- Episode consolidation (summarize and persist conversations)
- Retrieval coordination (working + episodic + semantic + emotional)

Usage:
    manager = MemoryManager()
    await manager.add_turn("user123", "user", "I love pizza")
    context = await manager.get_context("user123")
    # context contains recent turns + relevant facts + emotional state
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.memory.base import (
    EpisodicMemoryEntry,
    EmotionalMemoryEntry,
    EmotionalTone,
    MemoryManager as MemoryManagerBase,
    SemanticMemoryEntry,
    WorkingMemoryEntry,
)
from app.memory.emotional import (
    EmotionalMemoryStore,
    SQLiteEmotionalStore,
    create_emotional_store,
)
from app.memory.episodic import (
    EpisodicMemoryStore,
    SQLiteEpisodicStore,
    create_episodic_store,
)
from app.memory.semantic import (
    SemanticMemoryStore,
    create_semantic_store,
)
from app.memory.working import (
    WorkingMemory,
    create_working_memory,
)

logger = get_logger(__name__)


# ─── Fact extraction patterns ────────────────────────────────────────────
# Simple regex-based fact extraction. This is intentionally basic —
# the LLM-based extraction (using a smaller model via Ollama) is planned
# for Part 1.5 (Persona). These patterns catch the most common cases.

FACT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # "My name is X" / "I am called X" / "I'm X"
    ("name", re.compile(r"\b(?:my name is|i am called|i'm called|call me)\s+([A-Z][a-z]+)", re.IGNORECASE)),
    # "I live in X" / "I'm from X"
    ("location", re.compile(r"\b(?:i live in|i'm from|i am from|my city is)\s+([A-Z][a-zA-Z\s]+?)(?:[.,!?]|$)", re.IGNORECASE)),
    # "I work as a X" / "I'm a X" (profession)
    ("profession", re.compile(r"\b(?:i work as a|i work as an|i'm a|i am a)\s+([a-z][a-z\s]+?)(?:[.,!?]|$)", re.IGNORECASE)),
    # "I like X" / "I love X" / "I enjoy X"
    ("likes", re.compile(r"\b(?:i like|i love|i enjoy|i'm into|i am into)\s+([a-z][a-z\s]+?)(?:[.,!?]|$)", re.IGNORECASE)),
    # "I hate X" / "I don't like X"
    ("dislikes", re.compile(r"\b(?:i hate|i don't like|i dislike|can't stand)\s+([a-z][a-z\s]+?)(?:[.,!?]|$)", re.IGNORECASE)),
    # "I have a X" / "My X is called Y" (possessions/pets)
    ("has", re.compile(r"\b(?:i have a|i have an|my)\s+([a-z][a-z\s]+?)(?:\s+is\s+called|\s+named|[,.!?:]|$)", re.IGNORECASE)),
]


class MemoryManager(MemoryManagerBase):
    """Coordinates all memory layers.

    This is the main entry point for memory operations. The chat endpoint
    and persona engine interact only with this class — they never touch
    individual backends directly.

    The manager handles graceful degradation: if any backend fails to
    initialize or becomes unavailable, operations that would use it
    return empty results instead of raising. This ensures the LLM can
    still respond even when memory is partially unavailable.
    """

    def __init__(
        self,
        working: WorkingMemory | None = None,
        episodic: EpisodicMemoryStore | None = None,
        semantic: SemanticMemoryStore | None = None,
        emotional: EmotionalMemoryStore | None = None,
    ) -> None:
        """Initialize the memory manager with backends.

        If any backend is None, it's created via the factory functions.
        If creation fails, the backend is set to None and operations
        that would use it return empty results.
        """
        self._working = working or create_working_memory()
        self._episodic = episodic or create_episodic_store()
        self._semantic = semantic or create_semantic_store()
        self._emotional = emotional or create_emotional_store()

        logger.info(
            "memory.manager.initialized",
            working=type(self._working).__name__,
            episodic=type(self._episodic).__name__,
            semantic=type(self._semantic).__name__,
            emotional=type(self._emotional).__name__,
        )

    # ─── Working Memory operations ───────────────────────────────────────

    async def add_turn(
        self,
        user_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> WorkingMemoryEntry:
        """Add a turn to working memory and extract facts if user message."""
        entry = await self._working.add_turn(
            user_id=user_id,
            role=role,
            content=content,
            metadata=metadata,
        )

        # If this is a user message, try to extract facts
        if role == "user":
            await self._extract_facts(user_id, content)

        return entry

    async def get_context(
        self,
        user_id: str,
        query: str | None = None,
        max_entries: int = 12,
    ) -> list[WorkingMemoryEntry]:
        """Get recent conversation context."""
        return await self._working.get_context(user_id, max_entries)

    # ─── Episodic Memory operations ──────────────────────────────────────

    async def consolidate_episode(
        self,
        user_id: str,
        summary: str,
        topics: list[str] | None = None,
        emotional_tone: EmotionalTone = EmotionalTone.NEUTRAL,
    ) -> EpisodicMemoryEntry | None:
        """Promote a conversation summary to episodic memory."""
        entry = EpisodicMemoryEntry(
            entry_id=str(uuid.uuid4()),
            user_id=user_id,
            summary=summary,
            topics=topics or [],
            emotional_tone=emotional_tone,
            turn_count=0,  # Set by caller if known
        )

        result = await self._episodic.add(entry)
        if result is None:
            logger.warning("memory.episodic.consolidate_failed", user_id=user_id)
        return result

    async def recall(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[EpisodicMemoryEntry]:
        """Search episodic memory for relevant past conversations."""
        return await self._episodic.search(user_id, query, top_k)

    # ─── Semantic Memory operations ──────────────────────────────────────

    async def get_facts(
        self,
        user_id: str,
        keys: list[str] | None = None,
    ) -> list[SemanticMemoryEntry]:
        """Retrieve facts about the user."""
        return await self._semantic.get(user_id, keys)

    async def upsert_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        confidence: float = 1.0,
        source: str = "user_statement",
    ) -> SemanticMemoryEntry | None:
        """Insert or update a fact."""
        entry = SemanticMemoryEntry(
            entry_id=str(uuid.uuid4()),
            user_id=user_id,
            key=key,
            value=value,
            confidence=confidence,
            source=source,
        )
        return await self._semantic.upsert(entry)

    async def _extract_facts(self, user_id: str, content: str) -> None:
        """Extract facts from a user message using regex patterns.

        This is a simple heuristic approach. A more sophisticated
        extraction (using an LLM) is planned for Part 1.5.
        """
        facts_extracted = 0
        for key, pattern in FACT_PATTERNS:
            matches = pattern.findall(content)
            for match in matches:
                value = match.strip()
                if len(value) < 2 or len(value) > 100:
                    continue  # skip implausible extractions
                await self.upsert_fact(
                    user_id=user_id,
                    key=key,
                    value=value,
                    confidence=0.7,  # lower confidence for regex-extracted facts
                    source="regex_extracted",
                )
                facts_extracted += 1

        if facts_extracted > 0:
            logger.debug(
                "memory.facts.extracted",
                user_id=user_id,
                count=facts_extracted,
            )

    # ─── Emotional Memory operations ─────────────────────────────────────

    async def record_emotion(
        self,
        user_id: str,
        topic: str,
        tone: EmotionalTone,
        intensity: float = 0.5,
    ) -> EmotionalMemoryEntry | None:
        """Record an emotional association with a topic."""
        return await self._emotional.record(user_id, topic, tone, intensity)

    async def get_emotional_context(
        self,
        user_id: str,
        topic: str | None = None,
    ) -> EmotionalMemoryEntry | list[EmotionalMemoryEntry]:
        """Get emotional context for a topic or all topics."""
        if topic:
            return await self._emotional.get_topic_tone(user_id, topic)
        return await self._emotional.get_all(user_id)

    # ─── Health and lifecycle ────────────────────────────────────────────

    async def health_check(self) -> dict[str, bool]:
        """Check health of all backends."""
        results: dict[str, bool] = {}
        try:
            results["working"] = await self._working.health_check()
        except Exception:
            results["working"] = False
        try:
            results["episodic"] = await self._episodic.health_check()
        except Exception:
            results["episodic"] = False
        try:
            results["semantic"] = await self._semantic.health_check()
        except Exception:
            results["semantic"] = False
        try:
            results["emotional"] = await self._emotional.health_check()
        except Exception:
            results["emotional"] = False
        return results

    async def close(self) -> None:
        """Close all backend connections."""
        for backend in [self._working, self._episodic, self._semantic, self._emotional]:
            try:
                await backend.close()
            except Exception as exc:
                logger.warning("memory.close_failed", error=str(exc))


# ─── Singleton instance ─────────────────────────────────────────────────

_manager_instance: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """Get or create the singleton MemoryManager instance.

    The singleton pattern is important because:
    1. Backends maintain state (connections, caches) that should persist
    2. SQLite connections should be reused (not reopened per request)
    3. Turn counters need to persist across requests
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = MemoryManager()
    return _manager_instance


def reset_memory_manager() -> None:
    """Reset the singleton (for testing)."""
    global _manager_instance
    if _manager_instance is not None:
        import asyncio
        try:
            asyncio.get_event_loop().create_task(_manager_instance.close())
        except Exception:
            pass
    _manager_instance = None
