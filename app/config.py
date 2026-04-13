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

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
