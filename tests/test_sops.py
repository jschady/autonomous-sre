"""Tests for mock SOP data and search_sops function.

TDD: Written BEFORE implementation.
"""
import pytest

from data.sops.mock_sops import SOPS, search_sops


class TestSopData:
    """Verify SOP catalogue structure."""

    def test_sops_has_at_least_eight_entries(self):
        assert len(SOPS) >= 8

    def test_each_sop_has_required_fields(self):
        required = {"title", "content", "tags", "recommended_tool"}
        for sop in SOPS:
            assert required.issubset(sop.keys()), f"SOP missing fields: {sop.get('title')}"

    def test_each_sop_content_is_non_empty(self):
        for sop in SOPS:
            assert len(sop["content"]) > 20, f"SOP content too short: {sop['title']}"

    def test_each_sop_tags_is_list(self):
        for sop in SOPS:
            assert isinstance(sop["tags"], list)
            assert len(sop["tags"]) >= 1


class TestSearchSops:
    """Verify SOP search behaviour."""

    def test_crashloop_sop_found(self):
        results = search_sops("CrashLoopBackOff")
        assert len(results) >= 1
        titles = [r["title"] for r in results]
        assert any("CrashLoop" in t or "crash" in t.lower() for t in titles)

    def test_crashloop_result_has_restart_instructions(self):
        results = search_sops("CrashLoopBackOff")
        assert len(results) >= 1
        content_combined = " ".join(r["content"] for r in results).lower()
        assert "restart" in content_combined or "pod" in content_combined

    def test_oom_sop_found(self):
        results = search_sops("OOMKilled memory")
        assert len(results) >= 1
        titles = [r["title"] for r in results]
        assert any("OOM" in t or "memory" in t.lower() or "oom" in t.lower() for t in titles)

    def test_no_match_returns_empty(self):
        results = search_sops("totally unknown error xyz_q9z9")
        assert results == []

    def test_returns_max_two_matches(self):
        # A broad query that could match many SOPs
        results = search_sops("error")
        assert len(results) <= 2

    def test_search_is_case_insensitive(self):
        lower = search_sops("crashloopbackoff")
        upper = search_sops("CRASHLOOPBACKOFF")
        assert len(lower) == len(upper)

    def test_image_pull_sop_found(self):
        results = search_sops("ImagePullBackOff")
        assert len(results) >= 1

    def test_high_latency_sop_found(self):
        results = search_sops("HighLatency")
        assert len(results) >= 1

    def test_disk_pressure_sop_found(self):
        results = search_sops("DiskPressure")
        assert len(results) >= 1

    def test_cert_expired_sop_found(self):
        results = search_sops("CertificateExpired")
        assert len(results) >= 1

    def test_returns_list_of_dicts(self):
        results = search_sops("CrashLoopBackOff")
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, dict)
