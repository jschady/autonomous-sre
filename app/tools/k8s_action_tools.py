"""Mutating Kubernetes tools — restart, rollback, scale memory.

All tools are decorated with @tool for LangChain compatibility.
"""
from __future__ import annotations

import datetime
import logging

from langchain_core.tools import tool
from kubernetes.client.exceptions import ApiException

from app.tools.k8s_client import k8s_available
from app.tools.k8s_helpers import (
    EXECUTED_ACTIONS,
    K8S_DISABLED_MSG,
    ExecuteRollbackInput,
    RestartServiceInput,
    ScaleMemoryInput,
    build_clean_pod_spec,
    find_controller,
    find_target_container,
    format_memory,
    get_apps_v1,
    get_container_memory_limit,
    get_core_v1,
    is_rbac_error,
    parse_memory,
    rbac_message,
    wait_for_pod_deleted,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# restart_service
# ---------------------------------------------------------------------------


def _restart_bare_pod(namespace: str, pod_name: str) -> str:
    """Delete a bare pod to trigger a fresh restart.

    This is the only "restart" available for workloads not managed by a
    Deployment, StatefulSet, or DaemonSet.
    """
    try:
        core = get_core_v1()
        core.delete_namespaced_pod(name=pod_name, namespace=namespace)
        action = f"restart_service:pod/{pod_name}"
        EXECUTED_ACTIONS.append(action)
        return (
            f"No controller (Deployment/StatefulSet/DaemonSet) manages '{pod_name}'. "
            f"Deleted bare pod '{pod_name}' in namespace '{namespace}' to clear crash loop. "
            "Note: bare pods are not recreated automatically — reapply the manifest if needed. "
            f"Monitor with: kubectl get pod {pod_name} -n {namespace}"
        )
    except ApiException as exc:
        if exc.status == 404:
            return (
                f"[ERROR] No controller or pod named '{pod_name}' found in namespace '{namespace}'. "
                f"Run `kubectl get all -n {namespace}` to identify the resource."
            )
        if is_rbac_error(exc):
            return rbac_message("delete_namespaced_pod")
        logger.error("ApiException deleting bare pod %s/%s: %s", namespace, pod_name, exc)
        return f"[ERROR] Failed to delete bare pod '{pod_name}': API error {exc.status} — {exc.reason}"
    except Exception as exc:
        logger.error("Unexpected error deleting bare pod %s/%s: %s", namespace, pod_name, exc)
        return f"[ERROR] Unexpected error deleting bare pod: {exc}"


@tool
def restart_service(service_id: str, namespace: str = "default") -> str:
    """Restart a Kubernetes workload.

    For Deployments/StatefulSets/DaemonSets: issues a rolling restart via
    annotation patch.  For bare pods: deletes the pod to clear crash loops.
    Records the action in EXECUTED_ACTIONS for audit.
    """
    if not k8s_available():
        return K8S_DISABLED_MSG

    _ = RestartServiceInput(service_id=service_id, namespace=namespace)

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
    apps = get_apps_v1()

    kind, controller_name = find_controller(namespace, service_id)

    if kind is None:
        return _restart_bare_pod(namespace, service_id)

    _PATCH_FNS = {
        "Deployment": lambda: apps.patch_namespaced_deployment(name=controller_name, namespace=namespace, body=patch_body),
        "StatefulSet": lambda: apps.patch_namespaced_stateful_set(name=controller_name, namespace=namespace, body=patch_body),
        "DaemonSet": lambda: apps.patch_namespaced_daemon_set(name=controller_name, namespace=namespace, body=patch_body),
    }
    patch_fn = _PATCH_FNS.get(kind)
    if patch_fn is None:
        return f"[ERROR] Unsupported controller kind '{kind}' for '{service_id}' — manual restart required."

    rollout_kind = kind.lower().replace("set", "set")
    try:
        patch_fn()
        action = f"restart_service:{controller_name}"
        EXECUTED_ACTIONS.append(action)
        return (
            f"Successfully triggered rolling restart for {kind} '{controller_name}'. "
            "Pods will be replaced one by one. "
            f"Monitor with: kubectl rollout status {rollout_kind}/{controller_name}"
        )
    except ApiException as exc:
        if is_rbac_error(exc):
            return rbac_message(f"patch_namespaced_{kind.lower()}")
        logger.error("ApiException restarting %s=%s: %s", kind, controller_name, exc)
        return f"[ERROR] Failed to restart '{controller_name}' ({kind}): API error {exc.status} — {exc.reason}"
    except Exception as exc:
        logger.error("Unexpected error restarting service=%s: %s", service_id, exc)
        return f"[ERROR] Unexpected error restarting service: {exc}"


# ---------------------------------------------------------------------------
# execute_rollback
# ---------------------------------------------------------------------------


@tool
def execute_rollback(deployment_name: str, namespace: str = "default") -> str:
    """Roll back a Kubernetes deployment to the previous stable revision.

    Patches the deployment with an annotation to trigger undo, then records
    the action in EXECUTED_ACTIONS for audit.
    """
    if not k8s_available():
        return K8S_DISABLED_MSG

    _ = ExecuteRollbackInput(deployment_name=deployment_name, namespace=namespace)

    patch_body = {
        "metadata": {
            "annotations": {
                "deployment.kubernetes.io/revision": "0"
            }
        }
    }
    apps = get_apps_v1()

    kind, controller_name = find_controller(namespace, deployment_name)
    if kind is None:
        return (
            f"[ERROR] '{deployment_name}' in namespace '{namespace}' is a bare pod — "
            "rollback requires a Deployment/StatefulSet/DaemonSet with revision history. "
            "Use restart_service instead to delete and recreate the pod."
        )

    _PATCH_FNS = {
        "Deployment": lambda: apps.patch_namespaced_deployment(name=controller_name, namespace=namespace, body=patch_body),
        "StatefulSet": lambda: apps.patch_namespaced_stateful_set(name=controller_name, namespace=namespace, body=patch_body),
        "DaemonSet": lambda: apps.patch_namespaced_daemon_set(name=controller_name, namespace=namespace, body=patch_body),
    }
    patch_fn = _PATCH_FNS.get(kind)
    if patch_fn is None:
        return f"[ERROR] Unsupported controller kind '{kind}' for '{deployment_name}' — manual rollback required."

    rollout_kind = kind.lower()
    try:
        patch_fn()
        action = f"execute_rollback:{controller_name}"
        EXECUTED_ACTIONS.append(action)
        return (
            f"Successfully initiated rollback for {kind} '{controller_name}'. "
            "Rolling back to previous revision. "
            f"Monitor with: kubectl rollout status {rollout_kind}/{controller_name}"
        )
    except ApiException as exc:
        if is_rbac_error(exc):
            return rbac_message(f"patch_namespaced_{kind.lower()}")
        logger.error("ApiException rolling back %s=%s: %s", kind, controller_name, exc)
        return (
            f"[ERROR] Failed to rollback '{controller_name}' ({kind}): "
            f"API error {exc.status} — {exc.reason}"
        )
    except Exception as exc:
        logger.error("Unexpected error rolling back deployment=%s: %s", deployment_name, exc)
        return f"[ERROR] Unexpected error executing rollback: {exc}"


# ---------------------------------------------------------------------------
# scale_memory
# ---------------------------------------------------------------------------


@tool
def scale_memory(
    workload_name: str,
    namespace: str = "default",
    container: str = "",
    factor: float = 2.0,
    initial_limit: str = "256Mi",
) -> str:
    """Scale up memory limits for a workload to fix OOMKilled pods.

    For Deployments/StatefulSets/DaemonSets: patches the controller spec
    (rolling update creates new pods automatically).
    For bare pods: reads the spec, deletes the pod, recreates with new limits.

    When no memory limit is currently configured, sets initial_limit directly
    instead of erroring. Records the action in EXECUTED_ACTIONS for audit.
    """
    if not k8s_available():
        return K8S_DISABLED_MSG

    _ = ScaleMemoryInput(
        workload_name=workload_name,
        namespace=namespace,
        container=container,
        factor=factor,
        initial_limit=initial_limit,
    )

    kind, controller_name = find_controller(namespace, workload_name)

    if kind is not None:
        return _scale_memory_controller(kind, controller_name, namespace, container, factor, initial_limit)

    return _scale_memory_bare_pod(workload_name, namespace, container, factor, initial_limit)


def _scale_memory_controller(
    kind: str, name: str, namespace: str, container_hint: str, factor: float, initial_limit: str = "256Mi",
) -> str:
    """Patch a controller's container memory limit."""
    apps = get_apps_v1()

    _READ_FNS = {
        "Deployment": lambda: apps.read_namespaced_deployment(name=name, namespace=namespace),
        "StatefulSet": lambda: apps.read_namespaced_stateful_set(name=name, namespace=namespace),
        "DaemonSet": lambda: apps.read_namespaced_daemon_set(name=name, namespace=namespace),
    }
    _PATCH_FNS = {
        "Deployment": lambda body: apps.patch_namespaced_deployment(name=name, namespace=namespace, body=body),
        "StatefulSet": lambda body: apps.patch_namespaced_stateful_set(name=name, namespace=namespace, body=body),
        "DaemonSet": lambda body: apps.patch_namespaced_daemon_set(name=name, namespace=namespace, body=body),
    }

    try:
        obj = _READ_FNS[kind]()
    except ApiException as exc:
        if is_rbac_error(exc):
            return rbac_message(f"read_namespaced_{kind.lower()}")
        return f"[ERROR] Failed to read {kind} '{name}': API error {exc.status} — {exc.reason}"

    containers = obj.spec.template.spec.containers or []
    idx, cname = find_target_container(containers, container_hint)
    if idx is None:
        return f"[ERROR] Container '{container_hint}' not found in {kind} '{name}'."

    current_limit = get_container_memory_limit(containers[idx])
    if not current_limit:
        new_limit = initial_limit
        from_display = "(none)"
        scale_display = f"initial={new_limit}"
    else:
        current_bytes = parse_memory(current_limit)
        if current_bytes is None:
            return f"[ERROR] Could not parse memory limit '{current_limit}' on container '{cname}'."
        new_limit = format_memory(int(current_bytes * factor))
        from_display = current_limit
        scale_display = f"{current_limit} → {new_limit} ({factor}x)"

    patch_body = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {"name": cname, "resources": {"limits": {"memory": new_limit}}}
                    ]
                }
            }
        }
    }

    try:
        _PATCH_FNS[kind](patch_body)
        action = f"scale_memory:{kind}/{name}:{cname}:{from_display}->{new_limit}"
        EXECUTED_ACTIONS.append(action)
        return (
            f"Set memory limit on container '{cname}' in {kind} '{name}': {scale_display}. "
            "Rolling update will create new pods with the updated limit."
        )
    except ApiException as exc:
        if is_rbac_error(exc):
            return rbac_message(f"patch_namespaced_{kind.lower()}")
        return f"[ERROR] Failed to patch {kind} '{name}': API error {exc.status} — {exc.reason}"


def _scale_memory_bare_pod(
    pod_name: str, namespace: str, container_hint: str, factor: float, initial_limit: str = "256Mi",
) -> str:
    """Delete a bare pod and recreate it with a higher memory limit."""
    core = get_core_v1()

    try:
        pod = core.read_namespaced_pod(name=pod_name, namespace=namespace)
    except ApiException as exc:
        if exc.status == 404:
            return f"[ERROR] Pod '{pod_name}' not found in namespace '{namespace}'."
        if is_rbac_error(exc):
            return rbac_message("read_namespaced_pod")
        return f"[ERROR] Failed to read pod '{pod_name}': API error {exc.status} — {exc.reason}"

    containers = pod.spec.containers or []
    idx, cname = find_target_container(containers, container_hint)
    if idx is None:
        return f"[ERROR] Container '{container_hint}' not found in pod '{pod_name}'."

    current_limit = get_container_memory_limit(containers[idx])
    if not current_limit:
        new_limit = initial_limit
        from_display = "(none)"
    else:
        current_bytes = parse_memory(current_limit)
        if current_bytes is None:
            return f"[ERROR] Could not parse memory limit '{current_limit}' on container '{cname}'."
        new_limit = format_memory(int(current_bytes * factor))
        from_display = current_limit

    new_pod = build_clean_pod_spec(pod, idx, new_limit)

    try:
        core.delete_namespaced_pod(name=pod_name, namespace=namespace)
    except ApiException as exc:
        if is_rbac_error(exc):
            return rbac_message("delete_namespaced_pod")
        return f"[ERROR] Failed to delete pod '{pod_name}': API error {exc.status} — {exc.reason}"

    if err := wait_for_pod_deleted(core, pod_name, namespace):
        return err

    try:
        core.create_namespaced_pod(namespace=namespace, body=new_pod)
    except ApiException as exc:
        if is_rbac_error(exc):
            return rbac_message("create_namespaced_pod")
        return (
            f"[ERROR] Deleted old pod but failed to create replacement: "
            f"API error {exc.status} — {exc.reason}. "
            f"Reapply the manifest manually with memory limit={new_limit}."
        )

    action = f"scale_memory:pod/{pod_name}:{cname}:{from_display}->{new_limit}"
    EXECUTED_ACTIONS.append(action)
    return (
        f"Set memory limit on bare pod '{pod_name}' container '{cname}': "
        f"{from_display} → {new_limit}. "
        f"Pod recreated with new limit. "
        f"Monitor with: kubectl get pod {pod_name} -n {namespace}"
    )
