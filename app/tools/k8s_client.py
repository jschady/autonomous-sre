"""Singleton Kubernetes client factory.

Provides lazy-initialized, cached API client instances.  Detects whether the
code is running inside a cluster (KUBERNETES_SERVICE_HOST env var) and loads
the appropriate kubeconfig.

Kubeconfig resolution order:
  1. KUBECONFIG_B64 env var — base64-encoded kubeconfig, written to a temp file.
     Most portable for Docker: no volume mount required.
     Generate with: base64 < ~/.kube/config | tr -d '\\n'
  2. In-cluster config      — automatic when KUBERNETES_SERVICE_HOST is set.
  3. KUBECONFIG env var     — path to a kubeconfig file (kubernetes lib reads this).
  4. ~/.kube/config         — default kubeconfig location.

Set K8S_ENABLED=false to disable all k8s tooling (tools return a
"K8s not configured" message instead of raising).

Usage:
    from app.tools.k8s_client import get_core_v1_api, get_apps_v1_api, k8s_available
"""
from __future__ import annotations

import base64
import logging
import os
import tempfile
from functools import lru_cache

import kubernetes.client as k8s
import kubernetes.config as k8s_config
from kubernetes.config import ConfigException

logger = logging.getLogger(__name__)

# Temp file handle kept alive for process lifetime (GC would delete it otherwise)
_KUBECONFIG_TEMPFILE: tempfile.NamedTemporaryFile | None = None  # type: ignore[type-arg]


def _write_kubeconfig_b64(b64_value: str) -> str:
    """Decode a base64 kubeconfig string, write to a temp file, return its path."""
    global _KUBECONFIG_TEMPFILE
    try:
        content = base64.b64decode(b64_value)
    except Exception as exc:
        raise ValueError(f"KUBECONFIG_B64 is not valid base64: {exc}") from exc

    # NamedTemporaryFile with delete=False so the kubernetes lib can open it;
    # we keep the handle alive to prevent premature GC on some platforms.
    tmp = tempfile.NamedTemporaryFile(suffix=".kubeconfig", delete=False)
    tmp.write(content)
    tmp.flush()
    _KUBECONFIG_TEMPFILE = tmp
    logger.info("KUBECONFIG_B64 decoded to temp file: %s", tmp.name)
    return tmp.name


def _load_kube_config() -> None:
    """Load Kubernetes config using the resolution order described in the module docstring."""
    # 1. KUBECONFIG_B64
    b64 = os.environ.get("KUBECONFIG_B64", "").strip()
    if b64:
        path = _write_kubeconfig_b64(b64)
        k8s_config.load_kube_config(config_file=path)
        return

    # 2. In-cluster
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        logger.debug("Loading in-cluster Kubernetes config")
        k8s_config.load_incluster_config()
        return

    # 3 & 4. KUBECONFIG env var or default ~/.kube/config (handled by the lib)
    logger.debug("Loading kubeconfig from KUBECONFIG env or ~/.kube/config")
    k8s_config.load_kube_config()


@lru_cache(maxsize=1)
def _get_api_client() -> k8s.ApiClient:
    """Return a cached ApiClient, loading config on first call."""
    try:
        _load_kube_config()
    except ConfigException as exc:
        logger.warning("Could not load Kubernetes config: %s", exc)
    return k8s.ApiClient()


def k8s_available() -> bool:
    """Return True when k8s tooling is enabled via settings."""
    from app.config import get_settings
    return get_settings().k8s_enabled


def get_core_v1_api() -> k8s.CoreV1Api:
    """Return a CoreV1Api instance backed by the shared ApiClient."""
    return k8s.CoreV1Api(api_client=_get_api_client())


def get_apps_v1_api() -> k8s.AppsV1Api:
    """Return an AppsV1Api instance backed by the shared ApiClient."""
    return k8s.AppsV1Api(api_client=_get_api_client())
