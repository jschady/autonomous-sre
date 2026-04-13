"""Researcher node — searches SOPs and derives a recommended action.

Uses the error_summary to search the knowledge base, then asks the LLM
to synthesise a specific recommended_action from matching SOPs.
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langsmith import traceable

from app.agents.state import SREState
from app.config import get_settings
from app.tools.db_tools import query_knowledge_base
from app.utils.llm_cost import accumulate_cost, extract_usage
from data.sops.mock_sops import search_sops

_RESEARCH_PROMPT = """\
You are an expert SRE. Based on the error summary and matching SOPs below,
determine the single best remediation action to take.

Alert: {alertname}
Error Summary: {error_summary}

Matching SOPs:
{sop_context}

State the recommended action clearly in 1-2 sentences.
If no SOPs match, recommend "Manual investigation by on-call engineer".
"""


@traceable(name="research_node", metadata={"phase": "research"})
async def research_node(state: SREState) -> dict:
    """Search SOPs and produce a recommended_action."""
    settings = get_settings()
    payload = state["alert_payload"]
    error_summary = state.get("error_summary", "")
    triage_summary = state.get("triage_summary", "")

    # Build search query from error context
    alertname = payload.get("alertname", "")
    search_query = f"{alertname} {error_summary} {triage_summary}".strip()

    # Search SOPs
    sop_matches = search_sops(search_query)

    if sop_matches:
        sop_context = "\n\n".join(
            f"SOP: {s['title']}\nSteps: {s['content']}\nRecommended Tool: {s['recommended_tool']}"
            for s in sop_matches
        )
    else:
        sop_context = "No matching SOPs found."

    try:
        llm = ChatAnthropic(
            model=settings.processor_model,
            api_key=settings.anthropic_api_key,
        )
        prompt = _RESEARCH_PROMPT.format(
            alertname=alertname,
            error_summary=error_summary or triage_summary,
            sop_context=sop_context,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        recommended_action = response.content.strip()

        # Fallback if LLM returns empty content
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
