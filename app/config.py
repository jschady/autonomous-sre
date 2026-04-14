"""Application configuration loaded from environment variables."""
from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str = "placeholder"
    triage_model: str = "claude-sonnet-4-6"
    processor_model: str = "claude-haiku-4-5-20251001"
    max_retries: int = 3
    verification_wait_seconds: int = 30

    # LangSmith observability
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "true"
    langchain_project: str = "autonomous-sre"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # Postgres (pgvector + checkpoint)
    postgres_dsn: str = ""

    # Redis (alert state store)
    redis_url: str = "redis://localhost:6379"

    # Feature flags
    use_vector_db: str = "false"

    # RunPod / Local inference (Phase 3A)
    runpod_base_url: str = ""
    runpod_api_key: str = ""
    local_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    local_model_enabled: bool = True

    # OpenAI (used for embeddings in semantic cache / incident store)
    openai_api_key: str = ""

    # Semantic caching (Phase 3C)
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 86400
    semantic_cache_enabled: bool = False

    # Prompt directory (Phase 3D)
    prompt_dir: str = "prompts"

    # Local model pricing (USD per 1M tokens, compute-cost estimate)
    local_model_input_price_per_m: float = 0.10
    local_model_output_price_per_m: float = 0.10

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    # LangSmith reads directly from os.environ, not pydantic-settings.
    # Propagate tracing vars so traces are emitted regardless of provider.
    _propagate_env = {
        "LANGCHAIN_TRACING_V2": settings.langchain_tracing_v2,
        "LANGCHAIN_API_KEY": settings.langchain_api_key,
        "LANGCHAIN_PROJECT": settings.langchain_project,
        "LANGCHAIN_ENDPOINT": settings.langchain_endpoint,
        "OPENAI_API_KEY": settings.openai_api_key,
    }
    for key, value in _propagate_env.items():
        if value and not os.environ.get(key):
            os.environ[key] = value
    return settings


def clear_settings_cache() -> None:
    """Invalidate the settings cache (useful in tests)."""
    get_settings.cache_clear()
