"""Tests for app.llm.mock_client.MockLLMClient.

The mock client is critical for testing — these tests verify it behaves
correctly so we can trust it as a test fixture elsewhere.
"""

from __future__ import annotations

import pytest

from app.llm.base import LLMMessage, LLMRequest, LLMRole, LLMResponse, TokenChunk
from app.llm.errors import LLMError, LLMRateLimitError
from app.llm.mock_client import MockLLMClient


@pytest.fixture
def mock_client() -> MockLLMClient:
    """Return a fresh MockLLMClient for each test."""
    return MockLLMClient(default_model="mock-test-model")


@pytest.fixture
def simple_request() -> LLMRequest:
    """Return a minimal valid LLMRequest."""
    return LLMRequest(
        messages=[LLMMessage(role=LLMRole.USER, content="Hello, JOI!")],
        stream=False,
    )


class TestMockClientComplete:
    """Test the non-streaming complete() method."""

    @pytest.mark.asyncio
    async def test_returns_scripted_response(
        self, mock_client: MockLLMClient, simple_request: LLMRequest
    ) -> None:
        mock_client.queue_response("Hi there!")
        response = await mock_client.complete(simple_request)

        assert response.content == "Hi there!"
        assert response.provider == "mock"
        assert response.model == "mock-test-model"
        assert response.finish_reason == "stop"
        assert response.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_returns_multiple_responses_in_order(
        self, mock_client: MockLLMClient, simple_request: LLMRequest
    ) -> None:
        """Sequential calls should return queued responses in FIFO order."""
        mock_client.queue_response("first")
        mock_client.queue_response("second")
        mock_client.queue_response("third")

        r1 = await mock_client.complete(simple_request)
        r2 = await mock_client.complete(simple_request)
        r3 = await mock_client.complete(simple_request)

        assert r1.content == "first"
        assert r2.content == "second"
        assert r3.content == "third"

    @pytest.mark.asyncio
    async def test_echo_mode_when_no_responses_queued(
        self, mock_client: MockLLMClient, simple_request: LLMRequest
    ) -> None:
        """Without scripted responses, mock echoes the user's last message."""
        response = await mock_client.complete(simple_request)

        assert "[mock echo]" in response.content
        assert "Hello, JOI!" in response.content

    @pytest.mark.asyncio
    async def test_raises_queued_error(
        self, mock_client: MockLLMClient, simple_request: LLMRequest
    ) -> None:
        """If an error is queued, the next call should raise it (one-shot)."""
        mock_client.queue_error(LLMRateLimitError(provider="mock", retry_after_seconds=30))

        with pytest.raises(LLMRateLimitError) as exc_info:
            await mock_client.complete(simple_request)

        assert exc_info.value.retry_after_seconds == 30

        # Error should be one-shot — subsequent call should succeed
        response = await mock_client.complete(simple_request)
        assert "[mock echo]" in response.content

    @pytest.mark.asyncio
    async def test_call_count_increments(
        self, mock_client: MockLLMClient, simple_request: LLMRequest
    ) -> None:
        assert mock_client.call_count == 0
        await mock_client.complete(simple_request)
        assert mock_client.call_count == 1
        await mock_client.complete(simple_request)
        assert mock_client.call_count == 2

    @pytest.mark.asyncio
    async def test_last_request_recorded(
        self, mock_client: MockLLMClient, simple_request: LLMRequest
    ) -> None:
        assert mock_client.last_request is None
        await mock_client.complete(simple_request)
        assert mock_client.last_request is simple_request

    @pytest.mark.asyncio
    async def test_simulated_latency(
        self, simple_request: LLMRequest
    ) -> None:
        """Latency should be simulated when configured."""
        import time

        mock = MockLLMClient(latency_ms=50)
        start = time.monotonic()
        await mock.complete(simple_request)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms >= 45  # allow small timing variance


class TestMockClientStream:
    """Test the streaming stream() method."""

    @pytest.mark.asyncio
    async def test_yields_chunks_then_done(
        self, simple_request: LLMRequest
    ) -> None:
        """Stream should yield content chunks followed by a final done chunk."""
        mock = MockLLMClient(
            stream_chunks=[["Hello", "world", "!"]],
        )
        simple_request.stream = True

        chunks: list[TokenChunk] = []
        async for chunk in mock.stream(simple_request):
            chunks.append(chunk)

        # Last chunk should have finish_reason
        assert chunks[-1].finish_reason == "stop"
        assert chunks[-1].usage is not None
        assert chunks[-1].usage.total_tokens > 0

        # Earlier chunks should have content but no finish_reason
        content_chunks = chunks[:-1]
        assert len(content_chunks) == 3
        assert all(c.finish_reason is None for c in content_chunks)

    @pytest.mark.asyncio
    async def test_stream_splits_scripted_response_into_words(
        self, simple_request: LLMRequest
    ) -> None:
        """Without explicit stream_chunks, splits a queued response."""
        mock = MockLLMClient(responses=["Hello world from mock"])
        simple_request.stream = True

        chunks: list[TokenChunk] = []
        async for chunk in mock.stream(simple_request):
            chunks.append(chunk)

        content = "".join(c.content for c in chunks)
        assert "Hello" in content
        assert "world" in content
        assert "mock" in content


class TestMockClientHealthCheck:
    """Test the health_check() method."""

    @pytest.mark.asyncio
    async def test_healthy_by_default(self, mock_client: MockLLMClient) -> None:
        assert await mock_client.health_check() is True

    @pytest.mark.asyncio
    async def test_unhealthy_when_error_queued(self, mock_client: MockLLMClient) -> None:
        mock_client.queue_error(LLMError("oops", provider="mock"))
        assert await mock_client.health_check() is False
