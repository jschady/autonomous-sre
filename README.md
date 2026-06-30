# Autonomous SRE

An AI-powered Site Reliability Engineering agent that autonomously triages, investigates, and remediates Kubernetes alerts — with human-in-the-loop approval via Slack before any destructive action.

## Overview

Autonomous SRE receives Prometheus alertmanager webhooks and runs them through a stateful LangGraph workflow. Each alert flows through triage → investigation → SOP lookup → human approval → remediation → verification, with automatic retry and escalation logic.

**Models used:**
- `claude-sonnet-4-6` — triage and complex reasoning
- `claude-haiku-4-5-20251001` — processor/tool execution (cost-efficient)

## Agent Graph

```
                    ┌─────────┐
 webhook ──────────▶│  triage  │
                    └────┬────┘
             escalate/   │   ok
             fail ───────┤
                    ┌────▼─────┐
                    │  router  │──── cache hit ───▶ notify_slack ──▶┐
                    └────┬─────┘                                     │
                    ┌────▼──────┐                                    │
                    │ processor │                                    │
                    └────┬──────┘                                    │
                    ┌────▼──────────┐                                │
                    │  researcher   │                                │
                    └────┬──────────┘                                │
                    ┌────▼──────────┐◀───────────────────────────────┘
                    │  notify_slack │
                    └────┬──────────┘
                    ┌────▼────────┐
                    │ human_gate  │  ◀─ Slack approval
                    └────┬────────┘
                    ┌────▼────────┐
                    │   action    │──── RBAC blocked / rejected ───▶ escalate
                    └────┬────────┘
                    ┌────▼────────────┐
                    │  verification   │──── resolved ──▶ END
                    └────┬────────────┘
                         │ not resolved
                    retry_count < max ──▶ triage (retry)
                    retry exhausted  ──▶ escalate ──▶ END
```

## Features

- **LangGraph stateful workflow** — durable, resumable execution via PostgreSQL checkpointing
- **Semantic caching** — pgvector-backed similarity cache; repeat/similar alerts skip full analysis
- **Human-in-the-loop** — Slack Block Kit approval messages before any k8s action
- **RBAC-aware** — detects and escalates on Kubernetes permission denials
- **SOP library** — structured runbooks for: CrashLoopBackOff, OOMKilled, ImagePullBackOff, HighLatency, DiskPressure, CertificateExpired, HighErrorRate, ConnectionRefused
- **Cost tracking** — per-node token usage and USD cost accumulated in `SREState`
- **LangSmith observability** — all LLM calls traced automatically

## Project Structure

```
app/
├── agents/
│   ├── graph.py          # LangGraph workflow definition & conditional routing
│   └── state.py          # SREState TypedDict + create_initial_state()
├── nodes/
│   ├── triage.py         # Severity classification, tool selection
│   ├── router.py         # Semantic cache lookup
│   ├── processor.py      # Tool execution (k8s read, Prometheus queries)
│   ├── researcher.py     # SOP matching, recommended action
│   ├── human_gate.py     # Slack approval request + response handling
│   ├── action.py         # k8s remediation actions
│   └── verification.py   # Post-action health check
├── tools/
│   ├── k8s_read_tools.py # Pod logs, describe, events (read-only)
│   ├── k8s_action_tools.py # Restart, scale, cordon (write, RBAC-checked)
│   ├── k8s_tools.py      # Tool registry
│   └── db_tools.py       # pgvector knowledge base queries
├── utils/
│   ├── incident_store.py # Resolved incident persistence + pgvector cache lookup
│   ├── llm_factory.py    # Anthropic client factory (model selection)
│   ├── llm_cost.py       # Token cost calculation
│   ├── slack_blocks.py   # Block Kit message builders
│   └── slack_client.py   # Slack Web API wrapper
├── config.py             # pydantic-settings configuration
├── models.py             # FastAPI request/response models
└── main.py               # FastAPI app + lifespan (checkpointer init)
data/sops/                # SOP markdown files + search index
prompts/                  # YAML prompt templates (triage, processor, researcher)
infra/vps/                # Pulumi VPS deployment
infra/migrations/         # PostgreSQL schema migrations
scripts/
├── run_evals.py          # Golden dataset evaluation harness
└── ingest_docs.py        # SOP ingestion into pgvector
tests/                    # Unit + integration tests
```

## API

### `POST /webhook`

Receives Prometheus alertmanager webhook payloads.

```json
{
  "alertname": "CrashLoopBackOff",
  "labels": {
    "namespace": "production",
    "pod": "chaos-app-7d9f4b-xxxx",
    "env": "prod",
    "region": "us-east-1"
  },
  "annotations": {
    "summary": "Pod has been restarting repeatedly"
  }
}
```

### `POST /slack/interaction`

Handles Slack interactive component callbacks (human approval/rejection).

### `GET /health`

Liveness check.

## Running Tests

```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=term-missing

# Integration tests (requires running infra)
pytest tests/integration/ -v
```

## Evaluation

Run the golden dataset eval harness against curated alert scenarios:

```bash
python scripts/run_evals.py
```

Golden dataset scenarios are in `tests/golden_dataset/`:
- `oom_killed_simple.json`
- `crash_looping_moderate.json`
- `cascading_failure_complex.json`

## Deployment

Infrastructure is managed with Pulumi (VPS target). See `infra/vps/` for deployment details.

```bash
cd infra/vps
pulumi up --stack prod
```

The `cloud-init.yaml` provisions the host with Docker, configures the systemd service, and handles kubeconfig injection.

## State Schema

All agent state is typed via `SREState` (`app/agents/state.py`). Key fields:

| Field | Type | Description |
|---|---|---|
| `alert_id` | `str` | UUID assigned at intake |
| `severity` | `str` | Triage severity: `critical`, `high`, `medium`, `low` |
| `tools_to_run` | `list[str]` | Tools selected by triage for this alert type |
| `recommended_action` | `str` | SOP-derived remediation recommendation |
| `human_approved` | `bool` | Slack approval decision |
| `resolved` | `bool` | Verification outcome |
| `retry_count` | `int` | Current retry iteration |
| `cost_estimate_usd` | `float` | Accumulated LLM cost |
| `cache_hit` | `bool` | Whether semantic cache was hit |
| `rbac_blocked` | `bool` | Whether k8s action was denied |

## Architecture Notes

- **Checkpointing**: Uses `AsyncPostgresSaver` in production, `MemorySaver` as fallback. The FastAPI lifespan initialises the checkpointer before `build_graph()` is called — the graph itself is stateless at build time.
- **Semantic cache**: On cache hit, the router short-circuits to `notify_slack`, skipping `processor` and `researcher`. Similarity threshold is configurable via `CACHE_SIMILARITY_THRESHOLD`.
- **Immutable state**: All node functions return new dicts rather than mutating `SREState` in place.
- **Prompt templates**: Loaded from `prompts/*.yaml` at startup via `prompt_loader.py`, avoiding hardcoded strings in node logic.
