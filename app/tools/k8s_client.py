"""Singleton Kubernetes client factory.

Provides lazy-initialized, cached API client instances.  Detects whether the
code is running inside a cluster (KUBERNETES_SERVICE_HOST env var) and loads
the appropriate kubeconfig.

Usage:
    from app.tools.k8s_client import get_core_v1_api, get_apps_v1_api
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import kubernetes.client as k8s
import kubernetes.config as k8s_config
from kubernetes.config import ConfigException

logger = logging.getLogger(__name__)

# Sentinel so we can reset in tests
_CONFIG_LOADED: bool = False


def _load_kube_config() -> None:
    """Load Kubernetes config from in-cluster environment or local kubeconfig.

    Prefers in-cluster config when KUBERNETES_SERVICE_HOST is present.
    Falls back to ~/.kube/config otherwise.
    """
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        logger.debug("Loading in-cluster Kubernetes config")
        k8s_config.load_incluster_config()
    else:
        logger.debug("Loading local kubeconfig")
        k8s_config.load_kube_config()


@lru_cache(maxsize=1)
def _get_api_client() -> k8s.ApiClient:
    """Return a cached ApiClient, loading config on first call."""
    try:
        _load_kube_config()
    except ConfigException as exc:
        logger.warning("Could not load Kubernetes config: %s", exc)
    return k8s.ApiClient()


def get_core_v1_api() -> k8s.CoreV1Api:
    """Return a CoreV1Api instance backed by the shared ApiClient."""
    return k8s.CoreV1Api(api_client=_get_api_client())


def get_apps_v1_api() -> k8s.AppsV1Api:
    """Return an AppsV1Api instance backed by the shared ApiClient."""
    return k8s.AppsV1Api(api_client=_get_api_client())
