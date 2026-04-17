"""Shared helpers for Kubernetes tools.

RBAC detection, API accessors, controller resolution, Pydantic schemas,
memory parsing, and pod spec construction.
"""
from __future__ import annotations

import logging
import time

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

K8S_DISABLED_MSG = "[K8s not configured] Set K8S_ENABLED=true and provide KUBECONFIG_B64 or KUBECONFIG."


def is_rbac_error(exc: ApiException) -> bool:
    return getattr(exc, "status", None) == 403


def rbac_message(operation: str) -> str:
    return (
        f"{_RBAC_PREFIX} Permission denied for operation '{operation}'. "
        "The service account lacks the required RBAC permissions. "
        "Contact your cluster administrator to grant the necessary Role/ClusterRole."
    )


# ---------------------------------------------------------------------------
# Lazy API accessors — thin wrappers so unit tests can patch a single symbol
# ---------------------------------------------------------------------------


def get_core_v1():  # pragma: no cover — patched in tests
    return get_core_v1_api()


def get_apps_v1():  # pragma: no cover — patched in tests
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
    namespace: str = Field(default="default", description="Kubernetes namespace")


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


class ScaleMemoryInput(BaseModel):
    workload_name: str = Field(..., min_length=1, description="Pod or Deployment name")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    container: str = Field(default="", description="Target container (empty = first container)")
    factor: float = Field(default=2.0, gt=1.0, le=4.0, description="Multiplier for current limit (1.5–4x)")
    initial_limit: str = Field(default="256Mi", description="Limit to set when no existing limit is configured")


# ---------------------------------------------------------------------------
# Helper: parse "namespace/pod-name" or bare "pod-name"
# ---------------------------------------------------------------------------


def parse_pod_ref(pod_id: str) -> tuple[str, str]:
    """Return (namespace, pod_name) from 'namespace/pod' or 'pod'."""
    if "/" in pod_id:
        parts = pod_id.split("/", 1)
        return parts[0], parts[1]
    return "default", pod_id


# ---------------------------------------------------------------------------
# Controller resolution
# ---------------------------------------------------------------------------


def find_controller(namespace: str, name: str) -> tuple[str | None, str | None]:
    """Find the controller (Deployment/StatefulSet/DaemonSet) for a workload.

    Strategy:
      1. **Direct lookup** — try reading the controller by *name*.  Fast and
         reliable even when pods are temporarily missing (OOM crash, rolling
         restart in progress).
      2. **Fallback** — walk pod owner references for edge cases where the
         workload name doesn't exactly match a controller name.

    Returns:
        (kind, controller_name) e.g. ("Deployment", "oom-test")
        (None, None)            if no controller found
    """
    apps = get_apps_v1()

    _READS: list[tuple[str, callable]] = [
        ("Deployment", lambda: apps.read_namespaced_deployment(name=name, namespace=namespace)),
        ("StatefulSet", lambda: apps.read_namespaced_stateful_set(name=name, namespace=namespace)),
        ("DaemonSet", lambda: apps.read_namespaced_daemon_set(name=name, namespace=namespace)),
    ]

    for kind, read_fn in _READS:
        try:
            read_fn()
            return kind, name
        except ApiException as exc:
            if exc.status == 404:
                continue
            if is_rbac_error(exc):
                logger.warning("RBAC denied reading %s '%s' in namespace '%s'", kind, name, namespace)
                continue
            logger.error("ApiException reading %s '%s': %s", kind, name, exc)
            continue

    # Fallback: walk pod owner references
    return _resolve_workload_owner_via_pods(namespace, name)


def _resolve_workload_owner_via_pods(namespace: str, name: str) -> tuple[str | None, str | None]:
    """Fallback: find root controller by walking pod owner references."""
    try:
        core = get_core_v1()
        pod_list = core.list_namespaced_pod(namespace=namespace)
        pods = [p for p in (pod_list.items or []) if (p.metadata.name or "").startswith(name)]
        if not pods:
            logger.debug("No pods found with prefix '%s' in namespace '%s'", name, namespace)
            return None, None
        pod = pods[0]
    except ApiException as exc:
        logger.error("ApiException listing pods for owner resolution: %s", exc)
        return None, None

    owners = pod.metadata.owner_references or []
    if not owners:
        logger.debug("Pod '%s' has no owner references (bare pod)", pod.metadata.name)
        return None, None

    owner = owners[0]
    kind = owner.kind
    owner_name = owner.name

    if kind == "ReplicaSet":
        try:
            apps = get_apps_v1()
            rs = apps.read_namespaced_replica_set(name=owner_name, namespace=namespace)
            rs_owners = rs.metadata.owner_references or []
            if rs_owners and rs_owners[0].kind == "Deployment":
                return "Deployment", rs_owners[0].name
        except ApiException as exc:
            logger.error("ApiException reading ReplicaSet '%s': %s", owner_name, exc)

    return kind, owner_name


# ---------------------------------------------------------------------------
# Pod lifecycle helpers
# ---------------------------------------------------------------------------


def wait_for_pod_deleted(core, pod_name: str, namespace: str, timeout: int = 30) -> str | None:
    """Poll until a pod is fully gone. Returns an error string on timeout, None on success."""
    for _ in range(timeout):
        try:
            core.read_namespaced_pod(name=pod_name, namespace=namespace)
            time.sleep(1)
        except ApiException as exc:
            if exc.status == 404:
                return None
            logger.error("ApiException polling pod deletion %s/%s: %s", namespace, pod_name, exc)
            return None  # best-effort — proceed with create anyway
    return (
        f"[ERROR] Pod '{pod_name}' still terminating after {timeout}s. "
        f"Wait and retry: kubectl delete pod {pod_name} -n {namespace} --grace-period=0"
    )


# ---------------------------------------------------------------------------
# Memory parsing
# ---------------------------------------------------------------------------


_MEMORY_SUFFIXES = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4}


def parse_memory(value: str) -> int | None:
    """Parse a k8s memory string (e.g. '50Mi', '1Gi', '256000') into bytes."""
    if not value:
        return None
    for suffix, multiplier in _MEMORY_SUFFIXES.items():
        if value.endswith(suffix):
            try:
                return int(value.removesuffix(suffix)) * multiplier
            except ValueError:
                return None
    try:
        return int(value)
    except ValueError:
        return None


def format_memory(bytes_val: int) -> str:
    """Format bytes into the most readable k8s memory string."""
    if bytes_val >= 1024**3 and bytes_val % (1024**3) == 0:
        return f"{bytes_val // 1024**3}Gi"
    if bytes_val >= 1024**2:
        return f"{bytes_val // 1024**2}Mi"
    if bytes_val >= 1024:
        return f"{bytes_val // 1024}Ki"
    return str(bytes_val)


# ---------------------------------------------------------------------------
# Container helpers
# ---------------------------------------------------------------------------


def find_target_container(containers: list, container_hint: str) -> tuple[int, str] | tuple[None, None]:
    """Find the target container index and name from a container list."""
    if not containers:
        return None, None
    if container_hint:
        for i, c in enumerate(containers):
            name = c.name if hasattr(c, "name") else c.get("name", "")
            if name == container_hint:
                return i, name
        return None, None
    name = containers[0].name if hasattr(containers[0], "name") else containers[0].get("name", "")
    return 0, name


def get_container_memory_limit(container) -> str | None:
    """Extract the memory limit string from a container spec object."""
    resources = getattr(container, "resources", None)
    if resources is None:
        return None
    limits = getattr(resources, "limits", None) or {}
    return limits.get("memory")


def build_clean_pod_spec(pod, container_idx: int, new_memory_limit: str):
    """Build a new Pod manifest from an existing pod, updating one container's memory limit.

    Strips runtime-generated fields (status, uid, resourceVersion, etc.) so the
    API server accepts it as a fresh create.
    """
    from kubernetes import client as k8s_client

    containers = []
    for i, c in enumerate(pod.spec.containers):
        container = k8s_client.V1Container(
            name=c.name,
            image=c.image,
            command=c.command,
            args=c.args,
            env=c.env,
            ports=c.ports,
            volume_mounts=c.volume_mounts,
            resources=k8s_client.V1ResourceRequirements(
                limits=dict(c.resources.limits) if c.resources and c.resources.limits else {},
                requests=dict(c.resources.requests) if c.resources and c.resources.requests else None,
            ) if c.resources else None,
        )
        if i == container_idx:
            if container.resources is None:
                container.resources = k8s_client.V1ResourceRequirements(limits={}, requests=None)
            if container.resources.limits is None:
                container.resources.limits = {}
            container.resources.limits["memory"] = new_memory_limit
        containers.append(container)

    return k8s_client.V1Pod(
        api_version="v1",
        kind="Pod",
        metadata=k8s_client.V1ObjectMeta(
            name=pod.metadata.name,
            namespace=pod.metadata.namespace,
            labels=pod.metadata.labels,
            annotations=pod.metadata.annotations,
        ),
        spec=k8s_client.V1PodSpec(
            containers=containers,
            init_containers=pod.spec.init_containers,
            volumes=pod.spec.volumes,
            restart_policy=pod.spec.restart_policy,
            service_account_name=pod.spec.service_account_name,
            node_selector=pod.spec.node_selector,
            tolerations=pod.spec.tolerations,
        ),
    )
