"""Eval runner for the Autonomous SRE graph.

Runs the golden dataset (tests/golden_dataset/*.json) plus a couple of
supplemental scenarios against the real graph with real LLM calls.
Prints PASS/FAIL for each scenario based on final state.status.

Usage:
    python scripts/run_evals.py

Requires:
    ANTHROPIC_API_KEY set in environment or .env file
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Skip the post-action verification delay (default is 30s per verification).
# Must be set before app.nodes.verification is imported.
os.environ.setdefault("VERIFICATION_DELAY_SECONDS", "0")

from app.agents.graph import build_graph
from app.agents.state import create_initial_state
from app.tools.k8s_tools import set_mock_healthy
from langgraph.types import Command


# ---------------------------------------------------------------------------
# Eval scenarios
# ---------------------------------------------------------------------------

GOLDEN_DATASET_DIR = PROJECT_ROOT / "tests" / "golden_dataset"


def load_golden_scenarios() -> list[dict]:
    """Load eval scenarios from the golden dataset JSON files."""
    scenarios = []
    for path in sorted(GOLDEN_DATASET_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        scenarios.append({
            "name": f"{data['id']} — {data['input']['alertname']} ({path.name})",
            "description": data["description"],
            "payload": data["input"],
            "mock_healthy_after_action": data["mock_healthy_after_action"],
            "auto_approve": data["auto_approve"],
            "expected_status": data["expected_status"],
        })
    return scenarios


# Scenarios not represented in the golden dataset.
EXTRA_SCENARIOS = [
    {
        "name": "HighErrorRate — rollback scenario",
        "description": "High error rate after deploy — should trigger rollback",
        "payload": {
            "alertname": "HighErrorRate",
            "status": "firing",
            "labels": {
                "region": "us-west-2",
                "env": "prod",
                "cluster_id": "k8s-prod-2",
                "namespace": "payments",
                "service": "payment-api",
            },
            "annotations": {
                "summary": "Error rate above 10% for 5 minutes",
                "description": "5xx error rate at 12% — possible bad deployment",
            },
        },
        "mock_healthy_after_action": True,
        "auto_approve": True,
        "expected_status": "resolved",
    },
    {
        "name": "Unknown error — graceful escalation",
        "description": "Unrecognised alert — should escalate without crashing",
        "payload": {
            "alertname": "QuantumFluxAnomaly",
            "status": "firing",
            "labels": {},
            "annotations": {
                "summary": "Quantum flux anomaly detected in sector 7G",
            },
        },
        "mock_healthy_after_action": False,
        "auto_approve": False,
        "expected_status": "escalated",
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# A failed verification loops back to triage and interrupts at human_gate again,
# so resume repeatedly; the cap guards against an unterminated retry loop.
_MAX_RESUMES = 6


async def run_scenario(graph, scenario: dict) -> tuple[str, str]:
    """Run a single eval scenario. Returns (scenario_name, 'PASS'|'FAIL')."""
    name = scenario["name"]
    payload = scenario["payload"]
    auto_approve = scenario["auto_approve"]
    mock_healthy = scenario["mock_healthy_after_action"]
    expected = scenario["expected_status"]

    set_mock_healthy(False)  # Start unhealthy
    initial_state = create_initial_state(payload)
    alert_id = initial_state["alert_id"]
    config = {"configurable": {"thread_id": alert_id}}

    try:
        # First run — may interrupt at human_gate
        final_state = await graph.ainvoke(initial_state, config=config)

        # In LangGraph >=1.x ainvoke returns state with __interrupt__ key
        # rather than raising GraphInterrupt.
        resumes = 0
        while "__interrupt__" in final_state and resumes < _MAX_RESUMES:
            if auto_approve:
                # Set healthy before the action runs
                set_mock_healthy(mock_healthy)
            final_state = await graph.ainvoke(
                Command(resume={"approved": auto_approve}),
                config=config,
            )
            resumes += 1

        actual_status = final_state.get("status", "unknown")
        result = "PASS" if actual_status == expected else "FAIL"
        return name, result, actual_status

    except Exception as exc:
        return name, "FAIL", f"exception: {exc}"


async def main() -> None:
    print("=" * 60)
    print("Autonomous SRE — Eval Runner")
    print("=" * 60)

    graph = build_graph()
    scenarios = load_golden_scenarios() + EXTRA_SCENARIOS
    results = []

    for scenario in scenarios:
        print(f"\nRunning: {scenario['name']}")
        print(f"  {scenario['description']}")
        name, verdict, actual = await run_scenario(graph, scenario)
        results.append((name, verdict, actual, scenario["expected_status"]))
        icon = "PASS" if verdict == "PASS" else "FAIL"
        print(f"  Result: {icon} (expected={scenario['expected_status']}, actual={actual})")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(1 for _, v, _, _ in results if v == "PASS")
    total = len(results)
    for name, verdict, actual, expected in results:
        print(f"  {'PASS' if verdict == 'PASS' else 'FAIL'}  {name}")
    print(f"\n{passed}/{total} scenarios passed")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
