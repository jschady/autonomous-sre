"""Eval runner for the Autonomous SRE graph.

Runs 3 predefined scenarios against the real graph with real LLM calls.
Prints PASS/FAIL for each scenario based on final state.status.

Usage:
    python scripts/run_evals.py

Requires:
    ANTHROPIC_API_KEY set in environment or .env file
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import app.tools.k8s_tools as k8s_tools
from app.agents.graph import build_graph
from app.agents.state import create_initial_state
from langgraph.types import Command


# ---------------------------------------------------------------------------
# Eval scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "name": "CrashLoopBackOff in prod",
        "description": "Pod crash looping — should trigger restart and resolve",
        "payload": {
            "alertname": "PodCrashLooping",
            "status": "firing",
            "labels": {
                "region": "us-east-1",
                "env": "prod",
                "cluster_id": "k8s-prod-1",
                "namespace": "checkout",
                "service": "checkout-api",
                "pod": "crash-api-abc123",
            },
            "annotations": {
                "summary": "Pod is crash looping — restarted 8 times",
                "description": "CrashLoopBackOff: container failed to start",
            },
        },
        "mock_healthy_after_action": True,
        "auto_approve": True,
        "expected_status": "resolved",
    },
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

async def run_scenario(graph, scenario: dict) -> tuple[str, str]:
    """Run a single eval scenario. Returns (scenario_name, 'PASS'|'FAIL')."""
    name = scenario["name"]
    payload = scenario["payload"]
    auto_approve = scenario["auto_approve"]
    mock_healthy = scenario["mock_healthy_after_action"]
    expected = scenario["expected_status"]

    k8s_tools.MOCK_HEALTHY = False  # Start unhealthy
    initial_state = create_initial_state(payload)
    alert_id = initial_state["alert_id"]
    config = {"configurable": {"thread_id": alert_id}}

    try:
        # First run — may interrupt at human_gate
        final_state = await graph.ainvoke(initial_state, config=config)

        # In LangGraph >=1.x ainvoke returns state with __interrupt__ key
        # rather than raising GraphInterrupt.
        if "__interrupt__" in final_state:
            if auto_approve:
                # Set healthy before action
                k8s_tools.MOCK_HEALTHY = mock_healthy
                final_state = await graph.ainvoke(
                    Command(resume={"approved": True}),
                    config=config,
                )
            else:
                # Reject — should escalate
                final_state = await graph.ainvoke(
                    Command(resume={"approved": False}),
                    config=config,
                )

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
    results = []

    for scenario in SCENARIOS:
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
