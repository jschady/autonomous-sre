"""YAML-based prompt loader with narrative few-shot injection support.

Prompts live in the prompts/ directory as YAML files. Each file has:
  - version, name, template, variables, few_shot_enabled, few_shot_count

Few-shot examples are formatted as human-readable "stories" (not JSON)
to maximise performance of smaller models like Llama 8B.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptConfig:
    """Immutable prompt configuration loaded from a YAML file."""

    name: str
    version: str
    template: str
    variables: list[str]
    few_shot_enabled: bool
    few_shot_count: int


def load_prompt(name: str, prompt_dir: str = "prompts") -> PromptConfig:
    """Load a prompt configuration from a YAML file.

    Args:
        name:       Prompt name without extension (e.g. "triage").
        prompt_dir: Directory containing YAML prompt files.

    Returns:
        Frozen PromptConfig dataclass.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
        ValueError:        If required keys are missing from the YAML.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("pyyaml is required for prompt loading") from exc

    path = Path(prompt_dir) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    required = {"name", "version", "template", "variables"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Prompt file {path} is missing required keys: {missing}")

    return PromptConfig(
        name=data["name"],
        version=data["version"],
        template=data["template"],
        variables=list(data["variables"]),
        few_shot_enabled=bool(data.get("few_shot_enabled", False)),
        few_shot_count=int(data.get("few_shot_count", 3)),
    )


def render_prompt(config: PromptConfig, **kwargs: str) -> str:
    """Render a prompt template with the provided variables.

    The template may include a {few_shot_section} placeholder. If not
    provided in kwargs it is replaced with an empty string.

    Args:
        config: Loaded PromptConfig.
        **kwargs: Variable values to substitute into the template.

    Returns:
        Rendered prompt string.

    Raises:
        ValueError: If a required variable is missing from kwargs.
    """
    required = {v for v in config.variables if v != "few_shot_section"}
    missing = required - set(kwargs.keys())
    if missing:
        raise ValueError(
            f"Prompt '{config.name}' is missing required variables: {missing}"
        )

    # Ensure few_shot_section has a default
    fill = {**kwargs, "few_shot_section": kwargs.get("few_shot_section", "")}
    return config.template.format(**fill)


def format_incident_story(incident: dict) -> str:
    """Convert a resolved incident dict into a concise narrative story.

    Story format (NOT raw JSON) significantly boosts Llama 8B performance
    by giving it a concrete pattern to follow.

    Example output:
        Incident: OOMKilled in checkout. Investigation: Memory limit too low,
        container consumed 512Mi. Action: Increased memory limit to 1Gi.
        Result: Resolved.
    """
    alertname = incident.get("alertname", "Unknown alert")
    namespace = incident.get("namespace", "unknown")
    triage_summary = incident.get("triage_summary", "").strip()
    recommended_action = incident.get("recommended_action", "").strip()
    action_result = incident.get("action_result", "Resolved").strip()

    investigation = triage_summary or "Investigation details unavailable."
    action = recommended_action or "Manual investigation performed."
    result = action_result or "Resolved."

    return (
        f"Incident: {alertname} in {namespace}. "
        f"Investigation: {investigation} "
        f"Action: {action} "
        f"Result: {result}"
    )


def build_few_shot_section(incidents: list[dict]) -> str:
    """Format a list of resolved incidents as a few-shot section.

    Returns an empty string if incidents is empty.
    Each incident is converted to a narrative story (not JSON).
    """
    if not incidents:
        return ""

    stories = [format_incident_story(inc) for inc in incidents]
    body = "\n".join(stories)
    return f"--- Relevant Past Incidents ---\n{body}\n---"
