"""Projeto JOI - Working Memory layer.

Short-term conversation context with TTL expiration. Stores the last N
turns per user, automatically expiring after 30 minutes of inactivity.

Two implementations:
- RedisWorkingMemory: production, uses Redis for TTL and persistence
- InMemoryWorkingMemory: development/testing, no external dependencies

Both implement the same WorkingMemory protocol, so the MemoryManager
can use either transparently.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.memory.base import MemoryClient, WorkingMemoryEntry

logger = get_logger(__name__)


class WorkingMemory(MemoryClient):
    """Abstract interface for working memory backends."""

    async def add_turn(
        self,
        user_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> WorkingMemoryEntry:
        """Add a new conversation turn to working memory."""
        raise NotImplementedError

    async def get_context(
        self,
        user_id: str,
        max_entries: int = 12,
    ) -> list[WorkingMemoryEntry]:
        """Retrieve the recent conversation context for a user."""
        raise NotImplementedError

    async def get_next_turn_id(self, user_id: str) -> int:
        """Get the next sequential turn ID for a user."""
        raise NotImplementedError

    async def clear(self, user_id: str) -> int:
        """Clear all working memory for a user. Returns count cleared."""
        raise NotImplementedError


class InMemoryWorkingMemory(WorkingMemory):
    """In-memory working memory implementation.

    Uses a simple dict to store turns per user. Entries expire based on
    TTL (checked lazily on access). Suitable for development and testing
    where Redis is not available.

    Limitations:
    - Not persistent (lost on restart)
    - Not shared across processes
    - TTL checked lazily (expired entries removed on next access)
    """

    def __init__(self, ttl_seconds: int = 1800, max_turns_per_user: int = 50) -> None:
        """Initialize in-memory working memory.

        Args:
            ttl_seconds: Time-to-live for entries (default 30 min).
            max_turns_per_user: Maximum entries kept per user (FIFO eviction).
        """
        self._ttl = ttl_seconds
        self._max_turns = max_turns_per_user
        self._store: dict[str, list[WorkingMemoryEntry]] = defaultdict(list)
        self._turn_counters: dict[str, int] = defaultdict(int)

        logger.info(
            "memory.working.initialized",
            backend="in_memory",
            ttl_seconds=ttl_seconds,
            max_turns=max_turns_per_user,
        )

    async def add_turn(
        self,
        user_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> WorkingMemoryEntry:
        """Add a new turn."""
        turn_id = await self.get_next_turn_id(user_id)
        entry = WorkingMemoryEntry(
            user_id=user_id,
            role=role,
            content=content,
            turn_id=turn_id,
            metadata=metadata or {},
        )

        self._store[user_id].append(entry)

        # Enforce max turns (FIFO eviction)
        if len(self._store[user_id]) > self._max_turns:
            self._store[user_id] = self._store[user_id][-self._max_turns :]

        logger.debug(
            "memory.working.added",
            user_id=user_id,
            turn_id=turn_id,
            role=role,
            content_length=len(content),
        )

        return entry

    async def get_context(
        self,
        user_id: str,
        max_entries: int = 12,
    ) -> list[WorkingMemoryEntry]:
        """Get recent context, filtering out expired entries."""
        self._cleanup_expired(user_id)
        entries = self._store.get(user_id, [])
        # Return most recent N entries, in chronological order
        recent = entries[-max_entries:] if len(entries) > max_entries else entries
        return list(recent)

    async def get_next_turn_id(self, user_id: str) -> int:
        """Increment and return the turn counter for this user."""
        self._turn_counters[user_id] += 1
        return self._turn_counters[user_id]

    async def clear(self, user_id: str) -> int:
        """Clear all entries for a user."""
        count = len(self._store.get(user_id, []))
        self._store.pop(user_id, None)
        self._turn_counters.pop(user_id, None)
        logger.info("memory.working.cleared", user_id=user_id, count=count)
        return count

    async def health_check(self) -> bool:
        """In-memory backend is always healthy."""
        return True

    async def close(self) -> None:
        """Clear the store."""
        self._store.clear()
        self._turn_counters.clear()

    def _cleanup_expired(self, user_id: str) -> None:
        """Remove entries older than TTL for a user."""
        now = time.time()
        entries = self._store.get(user_id, [])
        if not entries:
            return

        # Keep only non-expired entries
        fresh = [e for e in entries if (now - e.timestamp) < self._ttl]
        if len(fresh) != len(entries):
            expired_count = len(entries) - len(fresh)
            logger.debug(
                "memory.working.expired_entries_removed",
                user_id=user_id,
                expired_count=expired_count,
            )
            self._store[user_id] = fresh


class RedisWorkingMemory(WorkingMemory):
    """Redis-backed working memory implementation.

    Uses Redis sorted sets and hashes for efficient TTL management.
    Each user's turns are stored in a sorted set keyed by timestamp,
    with the entry data stored as JSON in a hash.

    Benefits over InMemoryWorkingMemory:
    - Persistent across restarts
    - Shared across processes (useful for multi-worker deployments)
    - Native TTL expiration (no lazy cleanup needed)
    """

    REDIS_KEY_PREFIX = "joi:working"
    REDIS_TURNS_KEY = "joi:working:{user_id}:turns"  # sorted set
    REDIS_TURN_KEY = "joi:working:{user_id}:turn:{turn_id}"  # hash
    REDIS_COUNTER_KEY = "joi:working:{user_id}:counter"

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int | None = None,
        max_turns_per_user: int = 50,
    ) -> None:
        """Initialize Redis working memory.

        Args:
            redis_url: Redis connection URL. Defaults to settings.redis_url.
            ttl_seconds: TTL for entries. Defaults to settings.memory_working_ttl_seconds.
            max_turns_per_user: Maximum entries per user.
        """
        self._redis_url = redis_url or settings.redis_url
        self._ttl = ttl_seconds or settings.memory_working_ttl_seconds
        self._max_turns = max_turns_per_user
        self._client: Any = None  # lazy import

        logger.info(
            "memory.working.initialized",
            backend="redis",
            redis_url=self._redis_url,
            ttl_seconds=self._ttl,
        )

    async def _get_client(self) -> Any:
        """Lazy-load the Redis client (avoids import error if redis not installed)."""
        if self._client is None:
            try:
                import redis.asyncio as aioredis

                self._client = aioredis.from_url(
                    self._redis_url,
                    socket_connect_timeout=2.0,
                    socket_timeout=5.0,
                    decode_responses=True,
                )
            except ImportError:
                logger.error("memory.working.redis_not_installed")
                raise
        return self._client

    async def add_turn(
        self,
        user_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> WorkingMemoryEntry:
        """Add a turn to Redis."""
        client = await self._get_client()
        turn_id = await self.get_next_turn_id(user_id)
        timestamp = time.time()

        entry = WorkingMemoryEntry(
            user_id=user_id,
            role=role,
            content=content,
            turn_id=turn_id,
            metadata=metadata or {},
        )

        # Store entry data as JSON
        turn_key = self.REDIS_TURN_KEY.format(user_id=user_id, turn_id=turn_id)
        entry_json = entry.model_dump_json()

        # Use pipeline for atomicity
        pipe = client.pipeline()
        pipe.setex(turn_key, self._ttl, entry_json)
        # Add to sorted set with timestamp as score
        turns_key = self.REDIS_TURNS_KEY.format(user_id=user_id)
        pipe.zadd(turns_key, {str(turn_id): timestamp})
        pipe.expire(turns_key, self._ttl)
        # Trim to max_turns
        pipe.zremrangebyrank(turns_key, 0, -(self._max_turns + 1))
        await pipe.execute()

        logger.debug(
            "memory.working.added",
            backend="redis",
            user_id=user_id,
            turn_id=turn_id,
            role=role,
        )

        return entry

    async def get_context(
        self,
        user_id: str,
        max_entries: int = 12,
    ) -> list[WorkingMemoryEntry]:
        """Get recent context from Redis."""
        try:
            client = await self._get_client()
            turns_key = self.REDIS_TURNS_KEY.format(user_id=user_id)

            # Get most recent turn IDs from sorted set
            turn_ids = await client.zrevrange(turns_key, 0, max_entries - 1)
            if not turn_ids:
                return []

            # Fetch each turn's data
            entries: list[WorkingMemoryEntry] = []
            for turn_id in reversed(turn_ids):  # chronological order
                turn_key = self.REDIS_TURN_KEY.format(user_id=user_id, turn_id=turn_id)
                data = await client.get(turn_key)
                if data:
                    entries.append(WorkingMemoryEntry.model_validate_json(data))

            return entries

        except Exception as exc:
            logger.warning(
                "memory.working.get_context_failed",
                user_id=user_id,
                error=str(exc),
            )
            return []

    async def get_next_turn_id(self, user_id: str) -> int:
        """Increment and return the turn counter."""
        try:
            client = await self._get_client()
            counter_key = self.REDIS_COUNTER_KEY.format(user_id=user_id)
            turn_id = await client.incr(counter_key)
            await client.expire(counter_key, self._ttl)
            return int(turn_id)
        except Exception as exc:
            logger.warning("memory.working.counter_failed", error=str(exc))
            # Fallback: use timestamp as turn ID
            return int(time.time() * 1000) % 1000000

    async def clear(self, user_id: str) -> int:
        """Clear all entries for a user."""
        try:
            client = await self._get_client()
            turns_key = self.REDIS_TURNS_KEY.format(user_id=user_id)
            counter_key = self.REDIS_COUNTER_KEY.format(user_id=user_id)

            # Get all turn IDs
            turn_ids = await client.zrange(turns_key, 0, -1)
            count = len(turn_ids)

            # Delete all turn data and the sorted set
            pipe = client.pipeline()
            for turn_id in turn_ids:
                turn_key = self.REDIS_TURN_KEY.format(user_id=user_id, turn_id=turn_id)
                pipe.delete(turn_key)
            pipe.delete(turns_key)
            pipe.delete(counter_key)
            await pipe.execute()

            logger.info("memory.working.cleared", backend="redis", user_id=user_id, count=count)
            return count
        except Exception as exc:
            logger.warning("memory.working.clear_failed", error=str(exc))
            return 0

    async def health_check(self) -> bool:
        """Check if Redis is reachable."""
        try:
            client = await self._get_client()
            return bool(await client.ping())
        except Exception:
            return False

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ─── Factory function ────────────────────────────────────────────────────


def create_working_memory(
    backend: str | None = None,
    ttl_seconds: int | None = None,
) -> WorkingMemory:
    """Factory: create a working memory backend.

    Args:
        backend: 'redis', 'in_memory', or None (auto-detect).
            If None, tries Redis first, falls back to in-memory.
        ttl_seconds: TTL for entries. Defaults to settings.

    Returns:
        A WorkingMemory instance.
    """
    ttl = ttl_seconds or settings.memory_working_ttl_seconds

    if backend == "redis":
        return RedisWorkingMemory(ttl_seconds=ttl)
    elif backend == "in_memory":
        return InMemoryWorkingMemory(ttl_seconds=ttl)

    # Auto-detect: prefer Redis, fall back to in-memory
    # (decided at creation time; if Redis fails later, operations return empty)
    try:
        # Try to import redis to see if it's available
        import redis.asyncio  # noqa: F401

        # Try to connect (will be validated on first use via health_check)
        return RedisWorkingMemory(ttl_seconds=ttl)
    except ImportError:
        logger.info("memory.working.redis_unavailable_using_in_memory")
        return InMemoryWorkingMemory(ttl_seconds=ttl)
