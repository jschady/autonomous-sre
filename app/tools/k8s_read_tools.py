"""Read-only Kubernetes tools — events, pods, containers, logs, metrics.

All tools are decorated with @tool for LangChain compatibility.
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

from langchain_core.tools import tool
from kubernetes.client.exceptions import ApiException

from app.tools.k8s_client import k8s_available
from app.tools.k8s_helpers import (
    K8S_DISABLED_MSG,
    GetClusterEventsInput,
    FetchContainerLogsInput,
    GetSystemMetricsInput,
    get_core_v1,
    is_mock_healthy,
    is_rbac_error,
    parse_pod_ref,
    rbac_message,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@tool
def get_cluster_events(namespace: str, service: str) -> str:
    """Retrieve recent Kubernetes cluster events for a given namespace and service.

    Returns a formatted string of events useful for diagnosing pod failures.
    """
    if not k8s_available():
        return K8S_DISABLED_MSG

    _ = GetClusterEventsInput(namespace=namespace, service=service)

    try:
        core = get_core_v1()
        event_list = core.list_namespaced_event(namespace=namespace)
    except ApiException as exc:
        if is_rbac_error(exc):
            return rbac_message("list_namespaced_event")
        logger.error("ApiException fetching events namespace=%s: %s", namespace, exc)
        return (
            f"[ERROR] Failed to fetch cluster events for namespace='{namespace}': "
            f"API error {exc.status} — {exc.reason}"
        )
    except Exception as exc:
        logger.error("Unexpected error fetching events: %s", exc)
        return f"[ERROR] Unexpected error fetching cluster events: {exc}"

    items = event_list.items or []
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
    for ev in relevant[:20]:
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


# ---------------------------------------------------------------------------
# Pods & containers
# ---------------------------------------------------------------------------


@tool
def list_pods(namespace: str, name_prefix: str = "") -> str:
    """List pods in a namespace, optionally filtered by name prefix.

    Use this when a specific pod is not found (stale pod name from alert) to
    discover currently running pods for the same workload.
    Returns pod name, status, and restart count.
    """
    if not k8s_available():
        return K8S_DISABLED_MSG

    try:
        core = get_core_v1()
        pod_list = core.list_namespaced_pod(namespace=namespace)
    except ApiException as exc:
        if is_rbac_error(exc):
            return rbac_message("list_namespaced_pod")
        logger.error("ApiException listing pods namespace=%s: %s", namespace, exc)
        return f"[ERROR] Failed to list pods: API error {exc.status} — {exc.reason}"
    except Exception as exc:
        logger.error("Unexpected error listing pods: %s", exc)
        return f"[ERROR] Unexpected error listing pods: {exc}"

    items = pod_list.items or []
    if name_prefix:
        items = [p for p in items if (p.metadata.name or "").startswith(name_prefix)]

    if not items:
        suffix = f" matching prefix '{name_prefix}'" if name_prefix else ""
        return f"[No pods found in namespace={namespace}{suffix}]"

    header = f"[Pods in namespace={namespace}" + (f" prefix={name_prefix}]" if name_prefix else "]")
    lines = ["NAME                                          STATUS              RESTARTS"]
    for pod in items[:20]:
        name = (pod.metadata.name or "")[:44].ljust(44)
        phase = (pod.status.phase or "Unknown")[:18].ljust(18)
        restarts = sum(
            (cs.restart_count or 0)
            for cs in (pod.status.container_statuses or [])
        )
        lines.append(f"{name}  {phase}  {restarts}")
    return header + "\n" + "\n".join(lines)


@tool
def list_pod_containers(pod_id: str) -> str:
    """List all containers in a Kubernetes pod.

    Use this before fetch_container_logs to get valid container names.
    pod_id may be 'namespace/pod-name' or bare 'pod-name' (defaults to 'default').
    """
    if not k8s_available():
        return K8S_DISABLED_MSG

    namespace, pod_name = parse_pod_ref(pod_id)

    try:
        core = get_core_v1()
        pod = core.read_namespaced_pod(name=pod_name, namespace=namespace)
        containers = [c.name for c in (pod.spec.containers or [])]
        init_containers = [c.name for c in (pod.spec.init_containers or [])]
        lines = [f"[Containers in pod={pod_name} namespace={namespace}]"]
        lines.append("containers: " + ", ".join(containers) if containers else "containers: (none)")
        if init_containers:
            lines.append("init_containers: " + ", ".join(init_containers))
        return "\n".join(lines)
    except ApiException as exc:
        if is_rbac_error(exc):
            return rbac_message("read_namespaced_pod")
        logger.error("ApiException listing containers pod=%s: %s", pod_id, exc)
        if exc.status == 404:
            return f"[ERROR] Pod not found: namespace='{namespace}' pod='{pod_name}'"
        return f"[ERROR] Failed to list containers: API error {exc.status} — {exc.reason}"
    except Exception as exc:
        logger.error("Unexpected error listing containers pod=%s: %s", pod_id, exc)
        return f"[ERROR] Unexpected error listing pod containers: {exc}"


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


@tool
def fetch_container_logs(pod_id: str, container: str, tail: int = 100) -> str:
    """Fetch recent container logs from a Kubernetes pod.

    Returns the last `tail` lines of logs from the specified container.
    pod_id may be 'namespace/pod-name' or bare 'pod-name' (defaults to 'default').
    """
    if not k8s_available():
        return K8S_DISABLED_MSG

    _ = FetchContainerLogsInput(pod_id=pod_id, container=container, tail=tail)
    namespace, pod_name = parse_pod_ref(pod_id)

    try:
        core = get_core_v1()
        logs = core.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            container=container,
            tail_lines=tail,
        )
        return logs or f"[No log output for pod={pod_name} container={container}]"
    except ApiException as exc:
        if is_rbac_error(exc):
            return rbac_message("read_namespaced_pod_log")
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


# ---------------------------------------------------------------------------
# Resource specs
# ---------------------------------------------------------------------------


@tool
def get_pod_resources(pod_id: str) -> str:
    """Get current resource requests and limits for all containers in a pod.

    Critical for OOM diagnosis — shows whether a memory limit is already
    configured (and what it is) before recommending scaling actions.
    pod_id may be 'namespace/pod-name' or bare 'pod-name' (defaults to 'default').
    """
    if not k8s_available():
        return K8S_DISABLED_MSG

    namespace, pod_name = parse_pod_ref(pod_id)

    try:
        core = get_core_v1()
        pod = core.read_namespaced_pod(name=pod_name, namespace=namespace)
    except ApiException as exc:
        if is_rbac_error(exc):
            return rbac_message("read_namespaced_pod")
        if exc.status == 404:
            return f"[ERROR] Pod not found: namespace='{namespace}' pod='{pod_name}'"
        return f"[ERROR] Failed to read pod: API error {exc.status} — {exc.reason}"
    except Exception as exc:
        return f"[ERROR] Unexpected error reading pod resources: {exc}"

    lines = [f"[Resources in pod={pod_name} namespace={namespace}]"]
    for c in pod.spec.containers or []:
        res = c.resources
        limits = (res.limits or {}) if res else {}
        requests = (res.requests or {}) if res else {}
        lines.append(f"Container '{c.name}':")
        lines.append(f"  memory_limit: {limits.get('memory', 'N/A')}")
        lines.append(f"  memory_request: {requests.get('memory', 'N/A')}")
        lines.append(f"  cpu_limit: {limits.get('cpu', 'N/A')}")
        lines.append(f"  cpu_request: {requests.get('cpu', 'N/A')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@tool
def get_system_metrics(service_name: str, namespace: str = "default") -> str:
    """Retrieve current system metrics for the specified service.

    Queries Prometheus when PROMETHEUS_URL is configured; falls back to
    mock data (controlled by set_mock_healthy) for local dev and tests.

    Returns a structured string with error_rate, latency_p99, cpu_usage, memory_usage.
    """
    _ = GetSystemMetricsInput(service_name=service_name, namespace=namespace)

    from app.config import get_settings
    settings = get_settings()
    if settings.prometheus_enabled and settings.prometheus_url:
        return _query_prometheus(service_name, namespace, settings.prometheus_url)

    return _mock_metrics(service_name)


def _query_prometheus(service_name: str, namespace: str, prometheus_url: str) -> str:
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
            f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{service_name}.*",namespace="{namespace}"}}[5m])) * 100'
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
        f"status={status}"
    )


def _mock_metrics(service_name: str) -> str:
    """Return hardcoded mock metrics for local dev and tests."""
    if is_mock_healthy():
        return (
            f"service={service_name} "
            "error_rate=0.2% "
            "latency_p99=45ms "
            "cpu_usage=22.0% "
            "status=healthy"
        )
    return (
        f"service={service_name} "
        "error_rate=12.5% "
        "latency_p99=4500ms "
        "cpu_usage=92.0% "
        "status=degraded"
    )
