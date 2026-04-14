"""Unit tests for app.utils.prompt_loader — YAML prompt loading and few-shot injection."""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from app.utils.prompt_loader import (
    PromptConfig,
    build_few_shot_section,
    format_incident_story,
    load_prompt,
    render_prompt,
)

SAMPLE_YAML = textwrap.dedent("""\
    version: "1.0"
    name: "test_prompt"
    few_shot_enabled: true
    few_shot_count: 3
    variables:
      - alertname
      - error_summary
    template: |
      Alert: {alertname}
      Error: {error_summary}
      {few_shot_section}
""")


class TestLoadPrompt:
    def test_loads_valid_yaml(self, tmp_path: Path):
        (tmp_path / "test.yaml").write_text(SAMPLE_YAML)
        config = load_prompt("test", prompt_dir=str(tmp_path))
        assert config.name == "test_prompt"
        assert config.version == "1.0"
        assert config.few_shot_enabled is True
        assert config.few_shot_count == 3
        assert "alertname" in config.variables

    def test_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent", prompt_dir=str(tmp_path))

    def test_raises_value_error_for_missing_keys(self, tmp_path: Path):
        (tmp_path / "broken.yaml").write_text("name: test\n")
        with pytest.raises(ValueError, match="missing required keys"):
            load_prompt("broken", prompt_dir=str(tmp_path))

    def test_returns_frozen_config(self, tmp_path: Path):
        (tmp_path / "test.yaml").write_text(SAMPLE_YAML)
        config = load_prompt("test", prompt_dir=str(tmp_path))
        with pytest.raises((AttributeError, TypeError)):
            config.name = "modified"  # type: ignore[misc]


class TestRenderPrompt:
    def _config(self) -> PromptConfig:
        return PromptConfig(
            name="test",
            version="1.0",
            template="Alert: {alertname}\n{few_shot_section}",
            variables=["alertname"],
            few_shot_enabled=False,
            few_shot_count=0,
        )

    def test_renders_with_valid_kwargs(self):
        config = self._config()
        result = render_prompt(config, alertname="PodCrashLooping")
        assert "PodCrashLooping" in result

    def test_raises_for_missing_required_variable(self):
        config = self._config()
        with pytest.raises(ValueError, match="missing required variables"):
            render_prompt(config)  # alertname missing

    def test_few_shot_section_defaults_to_empty(self):
        config = self._config()
        result = render_prompt(config, alertname="Test")
        assert "{few_shot_section}" not in result

    def test_renders_few_shot_section_when_provided(self):
        config = self._config()
        result = render_prompt(
            config,
            alertname="Test",
            few_shot_section="--- Past Incidents ---\nIncident: X",
        )
        assert "Past Incidents" in result


class TestFormatIncidentStory:
    """format_incident_story must produce narrative text, NOT JSON."""

    def test_produces_narrative_format(self):
        incident = {
            "alertname": "OOMKilled",
            "namespace": "checkout",
            "triage_summary": "Memory limit too low, container consumed 512Mi.",
            "recommended_action": "Increase memory limit to 1Gi.",
            "action_result": "Resolved",
        }
        story = format_incident_story(incident)
        assert "Incident:" in story
        assert "Investigation:" in story
        assert "Action:" in story
        assert "Result:" in story
        assert "OOMKilled" in story
        assert "checkout" in story
        assert "1Gi" in story

    def test_does_not_contain_json_syntax(self):
        incident = {
            "alertname": "PodCrashLooping",
            "namespace": "payments",
            "triage_summary": "App startup failed.",
            "recommended_action": "Restart pod.",
            "action_result": "Resolved",
        }
        story = format_incident_story(incident)
        assert "{" not in story
        assert "}" not in story
        assert '"alertname"' not in story

    def test_handles_missing_fields_gracefully(self):
        story = format_incident_story({})
        assert "Incident:" in story
        assert "Investigation:" in story
        assert "Action:" in story


class TestBuildFewShotSection:
    def test_returns_empty_string_for_no_incidents(self):
        assert build_few_shot_section([]) == ""

    def test_returns_section_with_stories(self):
        incidents = [
            {
                "alertname": "OOMKilled",
                "namespace": "checkout",
                "triage_summary": "Memory limit too low.",
                "recommended_action": "Increase limit.",
                "action_result": "Resolved",
            },
        ]
        section = build_few_shot_section(incidents)
        assert "--- Relevant Past Incidents ---" in section
        assert "OOMKilled" in section
        assert "---" in section

    def test_section_contains_no_raw_json(self):
        incidents = [
            {"alertname": "Test", "namespace": "ns", "triage_summary": "x",
             "recommended_action": "y", "action_result": "Resolved"},
        ]
        section = build_few_shot_section(incidents)
        assert '"alertname"' not in section
