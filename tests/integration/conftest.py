"""Integration test configuration.

Loads .env into os.environ so POSTGRES_DSN, OPENAI_API_KEY, etc. are accessible.
Also overrides the parent conftest's mock_k8s_apis autouse fixture so integration
tests hit real Kubernetes APIs instead of mocks.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env so environment variables are available to integration tests
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


@pytest.fixture(autouse=True)
def mock_k8s_apis():
    """Override parent conftest fixture — integration tests use real APIs."""
    yield
