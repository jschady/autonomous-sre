"""Tests for SREState schema and create_initial_state factory.

TDD: These tests are written BEFORE the implementation.
"""
import uuid
import pytest
from unittest.mock import patch

from app.agents.state import create_initial_state, SREState


class TestCreateInitialStateDefaults:
    """Verify default field values set by create_initial_state."""

    def test_create_initial_state_defaults(self):
        payload = {"alertname": "TestAlert", "status": "firing"}
        state = create_initial_state(payload)

        assert state["retry_count"] == 0
        assert state["status"] == "in_progress"
        assert state["error_log"] == []
        assert state["human_approved"] is False
        assert state["resolved"] is False
        assert state["metrics_healthy"] is False

    def test_default_max_retries_is_three(self):
        payload = {"alertname": "TestAlert", "status": "firing"}
        state = create_initial_state(payload)
        assert state["max_retries"] == 3

    def test_alert_payload_stored(self):
        payload = {"alertname": "PodCrash", "status": "firing"}
        state = create_initial_state(payload)
        assert state["alert_payload"] == payload


class TestAlertId:
    """Verify alert_id generation."""

    def test_alert_id_is_uuid(self):
        payload = {"alertname": "TestAlert"}
        state = create_initial_state(payload)
        # Should not raise
        parsed = uuid.UUID(state["alert_id"])
        assert str(parsed) == state["alert_id"]

    def test_alert_id_is_unique_each_call(self):
        payload = {"alertname": "TestAlert"}
        state1 = create_initial_state(payload)
        state2 = create_initial_state(payload)
        assert state1["alert_id"] != state2["alert_id"]


class TestMetadataExtraction:
    """Verify metadata is extracted from webhook labels."""

    def test_metadata_extracted_from_labels(self):
        payload = {
            "alertname": "PodCrash",
            "labels": {
                "region": "us-east-1",
                "env": "prod",
                "cluster_id": "k8s-prod-1",
                "namespace": "checkout",
            },
        }
        state = create_initial_state(payload)
        assert state["metadata"]["region"] == "us-east-1"
        assert state["metadata"]["env"] == "prod"
        assert state["metadata"]["cluster_id"] == "k8s-prod-1"
        assert state["metadata"]["namespace"] == "checkout"

    def test_metadata_handles_missing_labels(self):
        payload = {"alertname": "TestAlert"}
        # Must not raise; metadata must be an empty-ish dict with defaults
        state = create_initial_state(payload)
        assert isinstance(state["metadata"], dict)
        # Defaults for missing standard keys
        assert state["metadata"]["region"] == "unknown"
        assert state["metadata"]["env"] == "unknown"
        assert state["metadata"]["cluster_id"] == "unknown"
        assert state["metadata"]["namespace"] == "default"

    def test_metadata_includes_extra_labels(self):
        payload = {
            "alertname": "PodCrash",
            "labels": {
                "region": "eu-west-1",
                "env": "staging",
                "cluster_id": "k8s-stage",
                "namespace": "payments",
                "service": "payment-api",
                "pod": "payment-api-abc123",
            },
        }
        state = create_initial_state(payload)
        assert state["metadata"]["service"] == "payment-api"
        assert state["metadata"]["pod"] == "payment-api-abc123"

    def test_metadata_handles_empty_labels_dict(self):
        payload = {"alertname": "TestAlert", "labels": {}}
        state = create_initial_state(payload)
        assert isinstance(state["metadata"], dict)
        assert state["metadata"]["region"] == "unknown"


class TestMaxRetriesCap:
    """Verify max_retries is capped at 5."""

    def test_max_retries_capped_at_5(self):
        payload = {"alertname": "TestAlert"}
        with patch.dict("os.environ", {"MAX_RETRIES": "10"}):
            state = create_initial_state(payload)
        assert state["max_retries"] <= 5

    def test_max_retries_respects_env_below_cap(self):
        payload = {"alertname": "TestAlert"}
        with patch.dict("os.environ", {"MAX_RETRIES": "2"}):
            state = create_initial_state(payload)
        assert state["max_retries"] == 2

    def test_max_retries_default_without_env(self):
        payload = {"alertname": "TestAlert"}
        # Remove env var if present
        import os
        env_backup = os.environ.pop("MAX_RETRIES", None)
        try:
            state = create_initial_state(payload)
            assert state["max_retries"] == 3
        finally:
            if env_backup is not None:
                os.environ["MAX_RETRIES"] = env_backup


class TestStateIsImmutablePattern:
    """Verify create_initial_state returns a fresh dict each call (no shared state)."""

    def test_returned_state_is_independent(self):
        payload = {"alertname": "TestAlert"}
        state1 = create_initial_state(payload)
        state2 = create_initial_state(payload)
        state1["error_log"].append("some error")
        assert state2["error_log"] == []
