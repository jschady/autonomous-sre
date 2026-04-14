"""Unit tests for app.utils.semantic_cache — Redis cosine similarity cache."""
from __future__ import annotations

import json
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.utils.semantic_cache import (
    _make_cache_key,
    cache_lookup,
    cache_store,
    compute_embedding,
    cosine_similarity,
)


class TestCosineSimilarity:
    """Pure function — no mocking needed."""

    def test_identical_vectors_return_one(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_return_minus_one(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_returns_zero_for_mismatched_lengths(self):
        assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_returns_zero_for_empty_vectors(self):
        assert cosine_similarity([], []) == 0.0

    def test_returns_zero_for_zero_magnitude(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_similarity_between_similar_vectors(self):
        a = [1.0, 2.0, 3.0]
        b = [1.1, 2.1, 2.9]
        sim = cosine_similarity(a, b)
        assert sim > 0.99  # Very similar vectors


class TestMakeCacheKey:
    def test_returns_string_with_prefix(self):
        embedding = [0.1, 0.2, 0.3]
        key = _make_cache_key(embedding)
        assert key.startswith("cache:incident:")
        assert len(key) > len("cache:incident:")

    def test_deterministic(self):
        embedding = [0.1, 0.2, 0.3]
        assert _make_cache_key(embedding) == _make_cache_key(embedding)

    def test_different_embeddings_produce_different_keys(self):
        a = [0.1, 0.2, 0.3]
        b = [0.4, 0.5, 0.6]
        assert _make_cache_key(a) != _make_cache_key(b)


class TestCacheLookup:
    """cache_lookup — tests against mocked Redis and OpenAI."""

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_redis_url(self):
        result = await cache_lookup(redis_url="", error_summary="OOMKilled in checkout")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_error_summary(self):
        result = await cache_lookup(redis_url="redis://localhost:6379", error_summary="")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_cached_result_on_high_similarity(self):
        query_embedding = [1.0, 0.0, 0.0]
        stored_embedding = [0.999, 0.001, 0.0]  # cosine similarity ~0.9999

        stored_entry = {
            "error_summary": "OOMKilled",
            "recommended_action": "Increase memory limit to 1Gi",
            "embedding": stored_embedding,
            "cached_at": "2026-04-13T00:00:00Z",
        }

        mock_redis = AsyncMock()
        mock_redis.scan = AsyncMock(return_value=(0, ["cache:incident:abc123"]))
        mock_redis.get = AsyncMock(return_value=json.dumps(stored_entry))
        mock_redis.aclose = AsyncMock()

        with patch("app.utils.semantic_cache.compute_embedding", return_value=query_embedding), \
             patch("redis.asyncio.from_url", return_value=mock_redis):
            result = await cache_lookup(
                redis_url="redis://localhost:6379",
                error_summary="OOMKilled in checkout",
                threshold=0.95,
            )

        assert result is not None
        assert result["recommended_action"] == "Increase memory limit to 1Gi"

    @pytest.mark.asyncio
    async def test_returns_none_on_low_similarity(self):
        query_embedding = [1.0, 0.0, 0.0]
        stored_embedding = [0.0, 1.0, 0.0]  # orthogonal — similarity = 0.0

        stored_entry = {
            "error_summary": "Completely different error",
            "recommended_action": "Some other action",
            "embedding": stored_embedding,
        }

        mock_redis = AsyncMock()
        mock_redis.scan = AsyncMock(return_value=(0, ["cache:incident:xyz"]))
        mock_redis.get = AsyncMock(return_value=json.dumps(stored_entry))
        mock_redis.aclose = AsyncMock()

        with patch("app.utils.semantic_cache.compute_embedding", return_value=query_embedding), \
             patch("redis.asyncio.from_url", return_value=mock_redis):
            result = await cache_lookup(
                redis_url="redis://localhost:6379",
                error_summary="Pod crash looping",
                threshold=0.95,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_redis_error(self):
        with patch("app.utils.semantic_cache.compute_embedding", return_value=[1.0, 0.0]), \
             patch("redis.asyncio.from_url", side_effect=ConnectionError("Redis down")):
            result = await cache_lookup(
                redis_url="redis://localhost:6379",
                error_summary="OOMKilled",
            )
        assert result is None


class TestCacheStore:
    @pytest.mark.asyncio
    async def test_no_op_for_empty_redis_url(self):
        # Should not raise, no side effects
        await cache_store(redis_url="", error_summary="test", recommended_action="test action")

    @pytest.mark.asyncio
    async def test_stores_entry_in_redis(self):
        embedding = [0.1, 0.2, 0.3]
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock()
        mock_redis.aclose = AsyncMock()

        with patch("app.utils.semantic_cache.compute_embedding", return_value=embedding), \
             patch("redis.asyncio.from_url", return_value=mock_redis):
            await cache_store(
                redis_url="redis://localhost:6379",
                error_summary="OOMKilled in checkout",
                recommended_action="Increase memory limit to 1Gi",
                ttl_seconds=3600,
            )

        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args.kwargs.get("ex") == 3600 or call_args.args[2] == 3600

    @pytest.mark.asyncio
    async def test_swallows_redis_error(self):
        with patch("app.utils.semantic_cache.compute_embedding", return_value=[0.1, 0.2]), \
             patch("redis.asyncio.from_url", side_effect=ConnectionError("Redis down")):
            # Should not raise
            await cache_store(
                redis_url="redis://localhost:6379",
                error_summary="test",
                recommended_action="test action",
            )
