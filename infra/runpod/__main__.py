"""Pulumi program: build a vLLM Docker image and deploy it to RunPod.

Strategy (per project design decision):
  - pulumi_docker: build and push the vLLM Docker image to a registry
  - RunPodResource (DynamicProvider via requests): hit the RunPod GraphQL API
    to create/destroy the GPU pod. This is more reliable than waiting for a
    third-party Pulumi provider to keep up with RunPod API changes.

Required env vars / Pulumi config:
  RUNPOD_API_KEY          RunPod API key
  DOCKER_REGISTRY         e.g. "docker.io/youruser" or "ghcr.io/yourorg"
  DOCKER_USERNAME         Registry username
  DOCKER_PASSWORD         Registry password / token
  HF_TOKEN                (optional) HuggingFace token for gated model access
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import pulumi
import pulumi_docker as docker
import pulumi.dynamic as dynamic
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

config = pulumi.Config()

# Use get() (not get_secret()) so it's a plain str, not Output[str].
# The value is still stored encrypted in the Pulumi state.
RUNPOD_API_KEY: str = config.get("runpodApiKey") or os.environ.get("RUNPOD_API_KEY", "")
DOCKER_REGISTRY: str = config.get("dockerRegistry") or os.environ.get("DOCKER_REGISTRY", "")
DOCKER_USERNAME: str = config.get("dockerUsername") or os.environ.get("DOCKER_USERNAME", "")
DOCKER_PASSWORD: str = config.get("dockerPassword") or os.environ.get("DOCKER_PASSWORD", "")
HF_TOKEN: str = config.get("hfToken") or os.environ.get("HF_TOKEN", "")

_image_tag = config.get("imageTag") or "latest"
IMAGE_NAME = f"{DOCKER_REGISTRY}/autonomous-sre-vllm:{_image_tag}" if DOCKER_REGISTRY else f"autonomous-sre-vllm:{_image_tag}"

RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"

# ---------------------------------------------------------------------------
# Step 1: Build and push Docker image via pulumi_docker
# ---------------------------------------------------------------------------

registry_info: docker.RegistryArgs | None = None
if DOCKER_USERNAME and DOCKER_PASSWORD and DOCKER_REGISTRY:
    _registry_server = DOCKER_REGISTRY.split("/")[0]
    # pulumi_docker requires the canonical Docker Hub host for auth
    if _registry_server == "docker.io":
        _registry_server = "https://index.docker.io/v1/"
    registry_info = docker.RegistryArgs(
        server=_registry_server,
        username=DOCKER_USERNAME,
        password=DOCKER_PASSWORD,
    )

vllm_image = docker.Image(
    "vllm-llama-image",
    image_name=IMAGE_NAME,
    build=docker.DockerBuildArgs(
        context=".",
        dockerfile="Dockerfile",
        platform="linux/amd64",
    ),
    registry=registry_info,
    skip_push=not bool(DOCKER_REGISTRY),
)


# ---------------------------------------------------------------------------
# Step 2: RunPod DynamicProvider — hit the RunPod GraphQL API via requests
# ---------------------------------------------------------------------------

class RunPodProvider(dynamic.ResourceProvider):
    """DynamicProvider that manages a RunPod GPU pod via GraphQL API."""

    def _graphql(self, query: str, variables: dict) -> dict:
        api_key = os.environ.get("RUNPOD_API_KEY") or RUNPOD_API_KEY
        if not api_key:
            raise ValueError(
                "RUNPOD_API_KEY is required. Set via: "
                "pulumi config set runpodApiKey <key>"
            )
        response = requests.post(
            RUNPOD_GRAPHQL_URL,
            json={"query": query, "variables": variables},
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        if "errors" in data:
            raise RuntimeError(f"RunPod GraphQL error: {data['errors']}")
        return data

    def create(self, props: dict) -> dynamic.CreateResult:
        image_name = props["image_name"]
        pod_name = props.get("pod_name", "autonomous-sre-llm")
        model = props.get("model", "meta-llama/Llama-3.1-8B-Instruct")
        gpu_util = props.get("gpu_util", "0.9")
        hf_token = props.get("hf_token", "")

        env_vars: list[dict] = [
            {"key": "MODEL", "value": model},
            {"key": "GPU_UTIL", "value": gpu_util},
            {"key": "MAX_MODEL_LEN", "value": "32768"},
        ]
        if hf_token:
            env_vars.append({"key": "HF_TOKEN", "value": hf_token})

        mutation = """
        mutation DeployPod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                runtime {
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                }
            }
        }
        """
        variables = {
            "input": {
                "name": pod_name,
                "imageName": image_name,
                "gpuTypeId": "NVIDIA L4",
                "gpuCount": 1,
                "containerDiskInGb": 50,
                "volumeInGb": 0,
                "ports": "8000/http",
                "env": env_vars,
                "startSsh": False,
                "supportPublicIp": True,
            }
        }

        result = self._graphql(mutation, variables)
        pod = result["data"]["podFindAndDeployOnDemand"]
        pod_id: str = pod["id"]
        endpoint_url = f"https://{pod_id}-8000.proxy.runpod.net/v1"

        pulumi.log.info(f"RunPod pod created: id={pod_id}, endpoint={endpoint_url}")

        return dynamic.CreateResult(
            id_=pod_id,
            outs={
                **props,
                "pod_id": pod_id,
                "endpoint_url": endpoint_url,
            },
        )

    def delete(self, id: str, props: dict) -> None:
        mutation = """
        mutation TerminatePod($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }
        """
        try:
            self._graphql(mutation, {"input": {"podId": id}})
            pulumi.log.info(f"RunPod pod terminated: id={id}")
        except Exception as exc:
            pulumi.log.warn(f"Failed to terminate RunPod pod {id}: {exc}")

    def diff(self, id: str, olds: dict, news: dict) -> dynamic.DiffResult:
        changes = olds.get("image_name") != news.get("image_name")
        return dynamic.DiffResult(changes=changes)


class RunPodResource(dynamic.Resource):
    """A Pulumi dynamic resource representing a RunPod GPU pod."""

    pod_id: pulumi.Output[str]
    endpoint_url: pulumi.Output[str]

    def __init__(
        self,
        name: str,
        image_name: pulumi.Input[str],
        pod_name: str = "autonomous-sre-llm",
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        gpu_util: str = "0.9",
        hf_token: str = "",
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__(
            RunPodProvider(),
            name,
            {
                "image_name": image_name,
                "pod_name": pod_name,
                "model": model,
                "gpu_util": gpu_util,
                "hf_token": hf_token,
                "pod_id": None,
                "endpoint_url": None,
            },
            opts,
        )


# ---------------------------------------------------------------------------
# Step 3: Deploy the pod
# ---------------------------------------------------------------------------

runpod_pod = RunPodResource(
    "autonomous-sre-pod",
    image_name=vllm_image.image_name,
    pod_name="autonomous-sre-llm",
    model="meta-llama/Llama-3.1-8B-Instruct",
    gpu_util="0.9",
    hf_token=HF_TOKEN,
    opts=pulumi.ResourceOptions(depends_on=[vllm_image]),
)

# ---------------------------------------------------------------------------
# Stack outputs
# ---------------------------------------------------------------------------

pulumi.export("pod_id", runpod_pod.pod_id)
pulumi.export("endpoint_url", runpod_pod.endpoint_url)
pulumi.export("image_name", vllm_image.image_name)
