"""Projeto JOI - Memory system.

Implements a multi-layer memory architecture inspired by human cognition:

    Working Memory  →  Episodic Memory  →  Semantic Memory
    (current turn)     (event timeline)     (facts about user)
                                                ↓
                                         Emotional Memory
                                         (affective context)

Each layer has a distinct storage backend and retrieval strategy:
- Working Memory: Redis (TTL 30min) - fast, ephemeral
- Episodic Memory: Vector store (ChromaDB or in-memory fallback) - semantic search
- Semantic Memory: SQLite/PostgreSQL - structured key-value facts
- Emotional Memory: SQLite - aggregated affective state per topic

The MemoryManager coordinates all layers and provides a unified API.
When memory is not available (e.g., no Redis/DB), the system degrades
gracefully — LLM calls work without memory, just with less context.
"""

from __future__ import annotations

from app.memory.base import (
    EpisodicMemoryEntry,
    EmotionalMemoryEntry,
    MemoryClient,
    MemoryManager,
    SemanticMemoryEntry,
    WorkingMemoryEntry,
)
from app.memory.manager import MemoryManager as MemoryManagerImpl
from app.memory.working import InMemoryWorkingMemory, RedisWorkingMemory, WorkingMemory

__all__ = [
    # Base types
    "MemoryClient",
    "MemoryManager",
    "WorkingMemoryEntry",
    "EpisodicMemoryEntry",
    "SemanticMemoryEntry",
    "EmotionalMemoryEntry",
    # Working memory implementations
    "WorkingMemory",
    "InMemoryWorkingMemory",
    "RedisWorkingMemory",
    # Manager
    "MemoryManagerImpl",
]
