"""Researcher node — searches SOPs and derives a recommended action."""
from __future__ import annotations

import logging
import os

from langchain_core.messages import HumanMessage
from langsmith import traceable

from app.agents.state import SREState
from app.config import get_settings
from app.utils.incident_store import fetch_similar_incidents
from app.utils.llm_cost import accumulate_cost, extract_usage
from app.utils.llm_factory import ainvoke_with_fallback
from app.utils.prompt_loader import build_few_shot_section, load_prompt, render_prompt
from app.tools.db_tools import _search as _db_search

logger = logging.getLogger(__name__)


@traceable(name="search_knowledge_base", run_type="tool", metadata={"phase": "research"})
def _search_knowledge_base(query: str) -> list[dict]:
    """Search SOPs via vector DB (when USE_VECTOR_DB=true) or keyword fallback."""
    return _db_search(query)


@traceable(name="research_node", metadata={"phase": "research"})
async def research_node(state: SREState) -> dict:
    """Search SOPs, inject few-shot examples, and produce a recommended_action."""
    settings = get_settings()
    payload = state["alert_payload"]
    error_summary = state.get("error_summary", "")
    triage_summary = state.get("triage_summary", "")

    # Build search query from error context
    alertname = payload.get("alertname", "")
    search_query = f"{alertname} {error_summary} {triage_summary}".strip()

    # Search SOPs
    sop_matches = _search_knowledge_base(search_query)

    if sop_matches:
        sop_context = "\n\n".join(
            f"SOP: {s['title']}\nSteps: {s['content']}\nRecommended Tool: {s['recommended_tool']}"
            for s in sop_matches
        )
    else:
        sop_context = "No matching SOPs found."

    # Fetch few-shot examples from resolved incidents
    few_shot_section = ""
    prompt_config = load_prompt("researcher", settings.prompt_dir)
    if prompt_config.few_shot_enabled:
        dsn = settings.postgres_dsn or os.environ.get("POSTGRES_DSN", "")
        try:
            similar_incidents = await fetch_similar_incidents(
                dsn=dsn,
                error_summary=error_summary or triage_summary,
                limit=prompt_config.few_shot_count,
            )
            few_shot_section = build_few_shot_section(similar_incidents)
        except Exception as exc:
            logger.warning("Failed to fetch few-shot examples (non-critical): %s", exc)

    try:
        prompt = render_prompt(
            prompt_config,
            alertname=alertname,
            error_summary=error_summary or triage_summary,
            sop_context=sop_context,
            few_shot_section=few_shot_section,
        )
        response = await ainvoke_with_fallback(
            provider="claude",
            messages=[HumanMessage(content=prompt)],
            model_override=settings.processor_model,
        )
        recommended_action = response.content.strip()

        if not recommended_action:
            recommended_action = (
                "Manual investigation recommended — no clear remediation identified."
            )

    except Exception as exc:
        return {
            "sop_matches": sop_matches,
            "recommended_action": "Manual investigation recommended due to research error.",
            "status": "failed",
            "error_log": state.get("error_log", []) + [f"research_node error: {exc}"],
            "current_node": "researcher",
        }

    usage = extract_usage("researcher", settings.processor_model, response)
    updated_token_usage = list(state.get("token_usage", [])) + [dict(usage)]

    sop_titles = [s["title"] for s in sop_matches]
    reasoning_entry = (
        f"[researcher] sops_matched={sop_titles} | "
        f"few_shot={'yes' if few_shot_section else 'no'} | "
        f"recommended_action={recommended_action!r} | "
        f"tokens={usage['total_tokens']} cost=${usage['cost_usd']:.6f}"
    )

    return {
        "sop_matches": sop_matches,
        "recommended_action": recommended_action,
        "proposed_action": recommended_action,
        "token_usage": updated_token_usage,
        "cost_estimate_usd": accumulate_cost(updated_token_usage),
        "reasoning_log": list(state.get("reasoning_log", [])) + [reasoning_entry],
        "current_node": "researcher",
    }
