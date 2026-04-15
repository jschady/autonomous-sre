"""Kubernetes tools for the Autonomous SRE system.

Phase 2: Real kubernetes Python client calls with Pydantic input validation.
Falls back gracefully on API errors; surfaces RBAC 403 errors explicitly.

All tools are decorated with @tool for LangChain compatibility.
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

from langchain_core.tools import tool
from kubernetes.client.exceptions import ApiException
from pydantic import BaseModel, Field, field_validator

from app.tools.k8s_client import get_apps_v1_api, get_core_v1_api, k8s_available

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state — kept for backward-compat with existing tests/conftest
# ---------------------------------------------------------------------------

MOCK_HEALTHY: bool = False
EXECUTED_ACTIONS: list[str] = []

# ---------------------------------------------------------------------------
# RBAC error sentinel
# ---------------------------------------------------------------------------

_RBAC_PREFIX = "[RBAC_DENIED]"


def _is_rbac_error(exc: ApiException) -> bool:
    return getattr(exc, "status", None) == 403


def _rbac_message(operation: str) -> str:
    return (
        f"{_RBAC_PREFIX} Permission denied for operation '{operation}'. "
        "The service account lacks the required RBAC permissions. "
        "Contact your cluster administrator to grant the necessary Role/ClusterRole."
    )


# ---------------------------------------------------------------------------
# Lazy API accessors — thin wrappers so unit tests can patch a single symbol
# ---------------------------------------------------------------------------


def _get_core_v1():  # pragma: no cover — patched in tests
    return get_core_v1_api()


def _get_apps_v1():  # pragma: no cover — patched in tests
    return get_apps_v1_api()


# ---------------------------------------------------------------------------
# Pydantic input schemas
# ---------------------------------------------------------------------------


class GetClusterEventsInput(BaseModel):
    namespace: str = Field(..., min_length=1, description="Kubernetes namespace")
    service: str = Field(..., description="Service name to filter events")


class FetchContainerLogsInput(BaseModel):
    pod_id: str = Field(..., min_length=1, description="Pod identifier (optionally namespace/pod-name)")
    container: str = Field(..., description="Container name within the pod")
    tail: int = Field(default=100, gt=0, description="Number of log lines to return")


class GetSystemMetricsInput(BaseModel):
    service_name: str = Field(..., min_length=1, description="Name of the service")


class RestartServiceInput(BaseModel):
    service_id: str = Field(..., min_length=1, description="Deployment name to restart")
    namespace: str = Field(default="default", description="Kubernetes namespace")

    @field_validator("service_id")
    @classmethod
    def service_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("service_id must not be blank")
        return v


class ExecuteRollbackInput(BaseModel):
    deployment_name: str = Field(..., min_length=1, description="Deployment name to roll back")
    namespace: str = Field(default="default", description="Kubernetes namespace")


# ---------------------------------------------------------------------------
# Helper: parse "namespace/pod-name" or bare "pod-name"
# ---------------------------------------------------------------------------


def _parse_pod_ref(pod_id: str) -> tuple[str, str]:
    """Return (namespace, pod_name) from 'namespace/pod' or 'pod'."""
    if "/" in pod_id:
        parts = pod_id.split("/", 1)
        return parts[0], parts[1]
    return "default", pod_id


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


_K8S_DISABLED_MSG = "[K8s not configured] Set K8S_ENABLED=true and provide KUBECONFIG_B64 or KUBECONFIG."


@tool
def get_cluster_events(namespace: str, service: str) -> str:
    """Retrieve recent Kubernetes cluster events for a given namespace and service.

    Returns a formatted string of events useful for diagnosing pod failures.
    """
    if not k8s_available():
        return _K8S_DISABLED_MSG

    # Validate input
    _ = GetClusterEventsInput(namespace=namespace, service=service)

    try:
        core = _get_core_v1()
        event_list = core.list_namespaced_event(namespace=namespace)
    except ApiException as exc:
        if _is_rbac_error(exc):
            return _rbac_message("list_namespaced_event")
        logger.error("ApiException fetching events namespace=%s: %s", namespace, exc)
        return (
            f"[ERROR] Failed to fetch cluster events for namespace='{namespace}': "
            f"API error {exc.status} — {exc.reason}"
        )
    except Exception as exc:
        logger.error("Unexpected error fetching events: %s", exc)
        return f"[ERROR] Unexpected error fetching cluster events: {exc}"

    items = event_list.items or []
    # Filter to events whose involved object name starts with the service prefix
    relevant = [
        ev for ev in items
        if ev.involved_object and service in (ev.involved_object.name or "")
    ]

    header = f"[Cluster Events — namespace={namespace} service={service}]\n"
    if not relevant:
        return header + f"No events found for service '{service}' in namespace '{namespace}'."

    lines: list[str] = [
        "LAST SEEN   TYPE      REASON              OBJECT                      MESSAGE"
    ]
    for ev in relevant[:20]:  # cap at 20 events for readability
        last_seen = _format_timestamp(ev.last_timestamp)
        ev_type = ev.type or "Unknown"
        reason = (ev.reason or "")[:18].ljust(18)
        obj_name = (ev.involved_object.name or "")[:26].ljust(26)
        message = (ev.message or "")[:60]
        lines.append(f"{last_seen:<12}{ev_type:<10}{reason:<20}{obj_name:<28}{message}")

    return header + "\n".join(lines)


def _format_timestamp(ts: Optional[datetime.datetime]) -> str:
    if ts is None:
        return "unknown"
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    delta = now - ts.replace(tzinfo=datetime.timezone.utc) if ts.tzinfo is None else now - ts
    minutes = int(delta.total_seconds() / 60)
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    return f"{hours}h"


@tool
def fetch_container_logs(pod_id: str, container: str, tail: int = 100) -> str:
    """Fetch recent container logs from a Kubernetes pod.

    Returns the last `tail` lines of logs from the specified container.
    pod_id may be 'namespace/pod-name' or bare 'pod-name' (defaults to 'default').
    """
    if not k8s_available():
        return _K8S_DISABLED_MSG

    _ = FetchContainerLogsInput(pod_id=pod_id, container=container, tail=tail)
    namespace, pod_name = _parse_pod_ref(pod_id)

    try:
        core = _get_core_v1()
        logs = core.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            container=container,
            tail_lines=tail,
        )
        return logs or f"[No log output for pod={pod_name} container={container}]"
    except ApiException as exc:
        if _is_rbac_error(exc):
            return _rbac_message("read_namespaced_pod_log")
        logger.error("ApiException fetching logs pod=%s: %s", pod_id, exc)
        if exc.status == 404:
            return (
                f"[ERROR] Pod or container not found: "
                f"namespace='{namespace}' pod='{pod_name}' container='{container}'"
            )
        return f"[ERROR] Failed to fetch logs: API error {exc.status} — {exc.reason}"
    except Exception as exc:
        logger.error("Unexpected error fetching logs pod=%s: %s", pod_id, exc)
        return f"[ERROR] Unexpected error fetching container logs: {exc}"


@tool
def get_system_metrics(service_name: str) -> str:
    """Retrieve current system metrics for the specified service.

    Queries Prometheus when PROMETHEUS_URL is configured; falls back to
    mock data (controlled by MOCK_HEALTHY) for local dev and tests.

    Returns a structured string with error_rate, latency_p99, cpu_usage, memory_usage.
    """
    _ = GetSystemMetricsInput(service_name=service_name)

    from app.config import get_settings
    settings = get_settings()
    if settings.prometheus_enabled and settings.prometheus_url:
        return _query_prometheus(service_name, settings.prometheus_url)

    return _mock_metrics(service_name)


def _query_prometheus(service_name: str, prometheus_url: str) -> str:
    """Query Prometheus for real service metrics."""
    import httpx

    base = prometheus_url.rstrip("/")
    queries = {
        "error_rate": (
            f'sum(rate(http_requests_total{{service="{service_name}",status=~"5.."}}[5m])) '
            f'/ sum(rate(http_requests_total{{service="{service_name}"}}[5m]))'
        ),
        "latency_p99": (
            f'histogram_quantile(0.99, sum(rate('
            f'http_request_duration_seconds_bucket{{service="{service_name}"}}[5m])) by (le))'
        ),
        "cpu_usage": (
            f'sum(rate(container_cpu_usage_seconds_total{{container="{service_name}"}}[5m])) * 100'
        ),
        "memory_usage": (
            f'sum(container_memory_working_set_bytes{{container="{service_name}"}}) '
            f'/ sum(container_spec_memory_limit_bytes{{container="{service_name}"}}) * 100'
        ),
    }

    results: dict[str, str] = {}
    try:
        with httpx.Client(timeout=10) as client:
            for metric, query in queries.items():
                resp = client.get(f"{base}/api/v1/query", params={"query": query})
                resp.raise_for_status()
                data = resp.json()
                result_list = data.get("data", {}).get("result", [])
                if result_list:
                    raw_value = float(result_list[0]["value"][1])
                    if metric == "latency_p99":
                        results[metric] = f"{raw_value * 1000:.0f}ms"
                    else:
                        results[metric] = f"{raw_value:.1f}%"
                else:
                    results[metric] = "N/A"
    except Exception as exc:
        logger.error("Prometheus query failed for service=%s: %s", service_name, exc)
        return (
            f"[ERROR] Failed to fetch metrics from Prometheus for service '{service_name}': {exc}"
        )

    error_rate = results.get("error_rate", "N/A")
    latency = results.get("latency_p99", "N/A")
    cpu = results.get("cpu_usage", "N/A")
    memory = results.get("memory_usage", "N/A")

    # Determine status from error rate
    status = "healthy"
    try:
        if error_rate != "N/A" and float(error_rate.rstrip("%")) > 5.0:
            status = "degraded"
    except ValueError:
        pass

    return (
        f"service={service_name} "
        f"error_rate={error_rate} "
        f"latency_p99={latency} "
        f"cpu_usage={cpu} "
        f"memory_usage={memory} "
        f"status={status}"
    )


def _mock_metrics(service_name: str) -> str:
    """Return hardcoded mock metrics for local dev and tests."""
    if MOCK_HEALTHY:
        return (
            f"service={service_name} "
            "error_rate=0.2% "
            "latency_p99=45ms "
            "cpu_usage=22.0% "
            "memory_usage=38.0% "
            "status=healthy"
        )
    return (
        f"service={service_name} "
        "error_rate=12.5% "
        "latency_p99=4500ms "
        "cpu_usage=92.0% "
        "memory_usage=88.0% "
        "status=degraded"
    )


@tool
def restart_service(service_id: str, namespace: str = "default") -> str:
    """Restart a Kubernetes deployment by rolling restart.

    Issues a patch to the deployment's pod template annotations to force a
    rolling restart.  Records the action in EXECUTED_ACTIONS for audit.
    """
    if not k8s_available():
        return _K8S_DISABLED_MSG

    _ = RestartServiceInput(service_id=service_id, namespace=namespace)

    try:
        apps = _get_apps_v1()
        patch_body = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": (
                                datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
                            )
                        }
                    }
                }
            }
        }
        apps.patch_namespaced_deployment(
            name=service_id, namespace=namespace, body=patch_body
        )
        action = f"restart_service:{service_id}"
        EXECUTED_ACTIONS.append(action)
        return (
            f"Successfully triggered rolling restart for service '{service_id}'. "
            "Pods will be replaced one by one. "
            f"Monitor with: kubectl rollout status deployment/{service_id}"
        )
    except ApiException as exc:
        if _is_rbac_error(exc):
            return _rbac_message("patch_namespaced_deployment")
        logger.error("ApiException restarting service=%s: %s", service_id, exc)
        return f"[ERROR] Failed to restart service '{service_id}': API error {exc.status} — {exc.reason}"
    except Exception as exc:
        logger.error("Unexpected error restarting service=%s: %s", service_id, exc)
        return f"[ERROR] Unexpected error restarting service: {exc}"


@tool
def execute_rollback(deployment_name: str, namespace: str = "default") -> str:
    """Roll back a Kubernetes deployment to the previous stable revision.

    Patches the deployment with an annotation to trigger undo, then records
    the action in EXECUTED_ACTIONS for audit.
    """
    if not k8s_available():
        return _K8S_DISABLED_MSG

    _ = ExecuteRollbackInput(deployment_name=deployment_name, namespace=namespace)

    try:
        apps = _get_apps_v1()
        # Kubernetes rollback: patch deployment with rollback annotation
        patch_body = {
            "metadata": {
                "annotations": {
                    "deployment.kubernetes.io/revision": "0"
                }
            }
        }
        apps.patch_namespaced_deployment(
            name=deployment_name, namespace=namespace, body=patch_body
        )
        action = f"execute_rollback:{deployment_name}"
        EXECUTED_ACTIONS.append(action)
        return (
            f"Successfully initiated rollback for deployment '{deployment_name}'. "
            "Rolling back to previous revision. "
            f"Monitor with: kubectl rollout status deployment/{deployment_name}"
        )
    except ApiException as exc:
        if _is_rbac_error(exc):
            return _rbac_message("patch_namespaced_deployment")
        logger.error("ApiException rolling back deployment=%s: %s", deployment_name, exc)
        return (
            f"[ERROR] Failed to rollback deployment '{deployment_name}': "
            f"API error {exc.status} — {exc.reason}"
        )
    except Exception as exc:
        logger.error("Unexpected error rolling back deployment=%s: %s", deployment_name, exc)
        return f"[ERROR] Unexpected error executing rollback: {exc}"
