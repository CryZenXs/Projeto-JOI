"""Projeto JOI - Memory base types and interfaces.

Defines the data models for each memory layer and the abstract MemoryClient
interface that all storage backends must implement.

Design principles:
- Each memory layer has its own data model (entries with appropriate fields)
- Storage backends are pluggable (Redis, ChromaDB, SQLite, Postgres, in-memory)
- The MemoryManager provides a unified API that coordinates all layers
- All operations are async (compatible with FastAPI's event loop)
- Failures degrade gracefully (memory unavailable → empty results, not crashes)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────


class MemoryLayer(str, Enum):
    """The four layers of memory, each with distinct persistence and retrieval."""

    WORKING = "working"  # Current conversation context (TTL 30min)
    EPISODIC = "episodic"  # Conversation history (permanent, searchable)
    SEMANTIC = "semantic"  # Facts about the user (permanent, structured)
    EMOTIONAL = "emotional"  # Affective context per topic (permanent, aggregated)


class EmotionalTone(str, Enum):
    """Standard emotional tones tracked in Emotional Memory.

    These 8 categories are deliberately coarse-grained — fine-grained
    emotion classification (e.g., "wistful nostalgia") is left to the
    LLM's interpretation via the persona engine.
    """

    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"  # fallback when no clear emotion


# ─── Data Models ─────────────────────────────────────────────────────────


class WorkingMemoryEntry(BaseModel):
    """A message in the current conversation context.

    Working Memory holds the last N turns of conversation (configurable,
    default 12). Entries expire after a TTL (default 30 minutes) of
    inactivity. This is the "short-term memory" that the LLM sees directly.
    """

    user_id: str
    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str
    timestamp: float = Field(default_factory=time.time)
    turn_id: int = Field(..., description="Sequential turn number for this user")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodicMemoryEntry(BaseModel):
    """A summarized conversation episode in long-term memory.

    Each entry represents a discrete conversation (or significant segment)
    that was important enough to persist. Includes an embedding for
    semantic retrieval ("what did we talk about last week?").
    """

    entry_id: str = Field(..., description="Unique ID (UUID)")
    user_id: str
    summary: str = Field(..., description="2-3 sentence summary of the episode")
    timestamp: float = Field(default_factory=time.time)
    turn_count: int = Field(default=0, ge=0, description="Number of turns in this episode")
    topics: list[str] = Field(default_factory=list, description="Key topics discussed")
    emotional_tone: EmotionalTone = Field(default=EmotionalTone.NEUTRAL)
    embedding: list[float] | None = Field(
        default=None,
        description="Vector embedding of the summary (for semantic search)",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticMemoryEntry(BaseModel):
    """A structured fact about the user.

    Semantic Memory stores key-value pairs of user attributes:
    - name, age, location (personal facts)
    - preferences (likes pizza, hates mondays)
    - relationships (sister named Maria)
    - work (works as a developer at ACME)

    Unlike episodic memory, these are updatable — if the user moves to
    a new city, the old "location" fact is updated, not appended.
    """

    entry_id: str = Field(..., description="Unique ID (UUID)")
    user_id: str
    key: str = Field(..., description="Fact key, e.g., 'location', 'job', 'sister_name'")
    value: str = Field(..., description="Fact value, e.g., 'Sao Paulo', 'developer'")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How confident we are in this fact (0-1)",
    )
    timestamp: float = Field(default_factory=time.time)
    source: str = Field(
        default="user_statement",
        description="How we learned this: 'user_statement', 'inferred', 'system'",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmotionalMemoryEntry(BaseModel):
    """An emotional context entry for a topic.

    Tracks the dominant emotional tone associated with each topic for
    each user. Allows the JOI to recall "we had a sad conversation about
    your dog last month" and respond with appropriate sensitivity.
    """

    entry_id: str = Field(..., description="Unique ID (UUID)")
    user_id: str
    topic: str = Field(..., description="Topic keyword, e.g., 'work', 'family', 'health'")
    tone: EmotionalTone
    intensity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How strong the emotion was (0-1)",
    )
    timestamp: float = Field(default_factory=time.time)
    occurrence_count: int = Field(
        default=1,
        ge=1,
        description="How many times this tone has been seen for this topic",
    )


# ─── Abstract storage interfaces ─────────────────────────────────────────


class MemoryClient(ABC):
    """Abstract base class for memory storage backends.

    Each layer has its own backend implementation:
    - WorkingMemory: RedisWorkingMemory or InMemoryWorkingMemory
    - EpisodicMemoryStore: ChromaDBEpisodicStore or InMemoryEpisodicStore
    - SemanticMemoryStore: PostgresSemanticStore or SQLiteSemanticStore
    - EmotionalMemoryStore: PostgresEmotionalStore or SQLiteEmotionalStore

    All methods are async. Implementations should handle their own
    connection management and degrade gracefully on failure.
    """

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is reachable and operational."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (connections, etc.). Call on app shutdown."""
        ...


# ─── MemoryManager interface ─────────────────────────────────────────────


class MemoryManager(ABC):
    """Coordinates all memory layers and provides a unified API.

    The MemoryManager is what the rest of the app (chat endpoint, persona
    engine) interacts with. It hides the complexity of multiple backends
    and provides high-level operations like:
    - add_turn(): record a new conversation turn
    - get_context(): retrieve relevant context for the next LLM call
    - extract_facts(): parse user messages for semantic facts
    - recall(): search episodic memory by semantic similarity

    If any backend is unavailable, the manager degrades gracefully —
    operations that would touch that backend return empty results
    instead of raising.
    """

    @abstractmethod
    async def add_turn(
        self,
        user_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> WorkingMemoryEntry:
        """Record a new turn in working memory."""
        ...

    @abstractmethod
    async def get_context(
        self,
        user_id: str,
        query: str | None = None,
        max_entries: int = 12,
    ) -> list[WorkingMemoryEntry]:
        """Retrieve conversation context for the next LLM call.

        Args:
            user_id: The user whose context to retrieve.
            query: Optional current query for relevance-based retrieval.
            max_entries: Maximum number of entries to return.

        Returns:
            List of working memory entries (most recent first).
        """
        ...

    @abstractmethod
    async def consolidate_episode(
        self,
        user_id: str,
        summary: str,
        topics: list[str] | None = None,
        emotional_tone: EmotionalTone = EmotionalTone.NEUTRAL,
    ) -> EpisodicMemoryEntry | None:
        """Promote a conversation summary to episodic memory.

        Called when a conversation ends or reaches a natural breakpoint.
        Returns None if episodic memory is unavailable.
        """
        ...

    @abstractmethod
    async def recall(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[EpisodicMemoryEntry]:
        """Search episodic memory for relevant past conversations.

        Uses semantic similarity (vector search) to find past episodes
        related to the query. Returns empty list if episodic memory
        is unavailable.
        """
        ...

    @abstractmethod
    async def get_facts(
        self,
        user_id: str,
        keys: list[str] | None = None,
    ) -> list[SemanticMemoryEntry]:
        """Retrieve structured facts about the user.

        Args:
            user_id: The user whose facts to retrieve.
            keys: Optional filter — only return facts with these keys.
                If None, return all facts.

        Returns:
            List of semantic memory entries (facts).
        """
        ...

    @abstractmethod
    async def upsert_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        confidence: float = 1.0,
        source: str = "user_statement",
    ) -> SemanticMemoryEntry | None:
        """Insert or update a fact about the user.

        If a fact with the same (user_id, key) exists, it's updated.
        Otherwise, a new fact is created. Returns None if semantic
        memory is unavailable.
        """
        ...

    @abstractmethod
    async def health_check(self) -> dict[str, bool]:
        """Check health of all memory backends.

        Returns a dict mapping backend names to their health status.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close all backend connections."""
        ...
