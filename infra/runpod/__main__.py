"""Pulumi program: build the vLLM Serverless image and deploy to RunPod Serverless.

Phase 4C replaces the always-on GPU pod with a RunPod Serverless endpoint:
  - idle_timeout: 30s  (GPU released after 30s of no traffic)
  - min_workers:  0    (scale to zero when idle)
  - Billed per-second only when inference is running (~$0.0002/s for RTX 4090)

The Serverless endpoint exposes an OpenAI-compatible proxy so the existing
ChatOpenAI-style client code works without modification when using the
always-on path.  The new RunPodServerlessChat client in
app/utils/runpod_client.py uses the /run + /status polling pattern for
the serverless path (handles cold-start delays gracefully).

Required env vars / Pulumi config:
  RUNPOD_API_KEY     RunPod API key
  DOCKER_REGISTRY    e.g. "ghcr.io/yourorg"
  DOCKER_USERNAME    Registry username
  DOCKER_PASSWORD    Registry password / token
  HF_TOKEN           (optional) HuggingFace token for gated model access
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

RUNPOD_API_KEY: str = config.get("runpodApiKey") or os.environ.get("RUNPOD_API_KEY", "")
DOCKER_REGISTRY: str = config.get("dockerRegistry") or os.environ.get("DOCKER_REGISTRY", "")
DOCKER_USERNAME: str = config.get("dockerUsername") or os.environ.get("DOCKER_USERNAME", "")
DOCKER_PASSWORD: str = config.get("dockerPassword") or os.environ.get("DOCKER_PASSWORD", "")
HF_TOKEN: str = config.get("hfToken") or os.environ.get("HF_TOKEN", "")

_image_tag = config.get("imageTag") or os.environ.get("IMAGE_TAG", "")
IMAGE_NAME = (
    f"{DOCKER_REGISTRY}/autonomous-sre-vllm-serverless:{_image_tag or 'latest'}"
    if DOCKER_REGISTRY
    else f"autonomous-sre-vllm-serverless:{_image_tag or 'latest'}"
)

RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"

# ---------------------------------------------------------------------------
# Step 1: Build and push the Serverless Docker image (weights baked in)
#
# If IMAGE_TAG / imageTag config is set, CI has already built and pushed the
# image — skip the local docker build entirely.  This is the normal path in
# production.  The local build path is retained for one-off developer use.
# ---------------------------------------------------------------------------

registry_info: docker.RegistryArgs | None = None
if DOCKER_USERNAME and DOCKER_PASSWORD and DOCKER_REGISTRY:
    _registry_server = DOCKER_REGISTRY.split("/")[0]
    if _registry_server == "docker.io":
        _registry_server = "https://index.docker.io/v1/"
    registry_info = docker.RegistryArgs(
        server=_registry_server,
        username=DOCKER_USERNAME,
        password=DOCKER_PASSWORD,
    )

if _image_tag:
    # Pre-built by CI — reference the image directly, no local build needed.
    pulumi.log.info(f"Using pre-built image: {IMAGE_NAME}")
    final_image_name: pulumi.Input[str] = IMAGE_NAME
else:
    # Local build path (slow — downloads 15 GB of model weights into the layer).
    pulumi.log.warn(
        "No imageTag set — building image locally. "
        "Set IMAGE_TAG env var or 'pulumi config set imageTag <sha>' to skip."
    )
    vllm_serverless_image = docker.Image(
        "vllm-serverless-image",
        image_name=IMAGE_NAME,
        build=docker.DockerBuildArgs(
            context=".",
            dockerfile="Dockerfile.serverless",
            platform="linux/amd64",
        ),
        registry=registry_info,
        skip_push=not bool(DOCKER_REGISTRY),
    )
    final_image_name = vllm_serverless_image.image_name


# ---------------------------------------------------------------------------
# Step 2: RunPod Serverless DynamicProvider
# ---------------------------------------------------------------------------

class RunPodServerlessProvider(dynamic.ResourceProvider):
    """DynamicProvider that manages a RunPod Serverless endpoint via GraphQL."""

    def _graphql(self, query: str, variables: dict) -> dict:
        api_key = os.environ.get("RUNPOD_API_KEY") or RUNPOD_API_KEY
        if not api_key:
            raise ValueError(
                "RUNPOD_API_KEY is required. Set via: pulumi config set runpodApiKey <key>"
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
        endpoint_name = props.get("endpoint_name", "autonomous-sre-serverless")
        idle_timeout = props.get("idle_timeout", 30)
        min_workers = props.get("min_workers", 0)
        max_workers = props.get("max_workers", 3)
        gpu_ids = props.get("gpu_ids", "AMPERE_16")

        mutation = """
        mutation SaveTemplate($input: SaveTemplateInput!) {
            saveTemplate(input: $input) {
                id
                name
            }
        }
        """
        template_vars = {
            "input": {
                "name": endpoint_name,
                "imageName": image_name,
                "containerDiskInGb": 50,
                "volumeInGb": 0,
                "dockerArgs": "",
                "env": [],
                "isServerless": True,
            }
        }
        template_result = self._graphql(mutation, template_vars)
        template_id = template_result["data"]["saveTemplate"]["id"]

        endpoint_mutation = """
        mutation SaveEndpoint($input: EndpointInput!) {
            saveEndpoint(input: $input) {
                id
                name
            }
        }
        """
        endpoint_vars = {
            "input": {
                "name": endpoint_name,
                "templateId": template_id,
                "gpuIds": gpu_ids,
                "idleTimeout": idle_timeout,
                "workersMin": min_workers,
                "workersMax": max_workers,
                "scalerType": "QUEUE_DELAY",
                "scalerValue": 4,
            }
        }
        endpoint_result = self._graphql(endpoint_mutation, endpoint_vars)
        endpoint_id = endpoint_result["data"]["saveEndpoint"]["id"]
        endpoint_url = f"https://api.runpod.ai/v2/{endpoint_id}"

        pulumi.log.info(
            f"RunPod Serverless endpoint created: id={endpoint_id}, url={endpoint_url}"
        )

        return dynamic.CreateResult(
            id_=endpoint_id,
            outs={
                **props,
                "template_id": template_id,
                "endpoint_id": endpoint_id,
                "endpoint_url": endpoint_url,
            },
        )

    def delete(self, id: str, props: dict) -> None:
        mutation = """
        mutation DeleteEndpoint($id: String!) {
            deleteEndpoint(id: $id)
        }
        """
        try:
            self._graphql(mutation, {"id": id})
            pulumi.log.info(f"RunPod Serverless endpoint deleted: id={id}")
        except Exception as exc:
            pulumi.log.warn(f"Failed to delete RunPod endpoint {id}: {exc}")

    def diff(self, id: str, olds: dict, news: dict) -> dynamic.DiffResult:
        changes = olds.get("image_name") != news.get("image_name")
        return dynamic.DiffResult(changes=changes)


class RunPodServerlessResource(dynamic.Resource):
    """Pulumi dynamic resource representing a RunPod Serverless endpoint."""

    endpoint_id: pulumi.Output[str]
    endpoint_url: pulumi.Output[str]
    template_id: pulumi.Output[str]

    def __init__(
        self,
        name: str,
        image_name: pulumi.Input[str],
        endpoint_name: str = "autonomous-sre-serverless",
        idle_timeout: int = 30,
        min_workers: int = 0,
        max_workers: int = 3,
        gpu_ids: str = "AMPERE_16",
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__(
            RunPodServerlessProvider(),
            name,
            {
                "image_name": image_name,
                "endpoint_name": endpoint_name,
                "idle_timeout": idle_timeout,
                "min_workers": min_workers,
                "max_workers": max_workers,
                "gpu_ids": gpu_ids,
                "endpoint_id": None,
                "endpoint_url": None,
                "template_id": None,
            },
            opts,
        )


# ---------------------------------------------------------------------------
# Step 3: Deploy the Serverless endpoint
# ---------------------------------------------------------------------------

serverless_endpoint = RunPodServerlessResource(
    "autonomous-sre-serverless",
    image_name=final_image_name,
    endpoint_name="autonomous-sre-serverless",
    idle_timeout=30,
    min_workers=0,
    max_workers=3,
    gpu_ids="AMPERE_16",
    opts=pulumi.ResourceOptions(depends_on=[vllm_serverless_image]) if not _image_tag else None,
)

# ---------------------------------------------------------------------------
# Stack outputs
# ---------------------------------------------------------------------------

pulumi.export("endpoint_id", serverless_endpoint.endpoint_id)
pulumi.export("endpoint_url", serverless_endpoint.endpoint_url)
pulumi.export("image_name", final_image_name)
pulumi.export(
    "env_vars",
    serverless_endpoint.endpoint_id.apply(
        lambda eid: {
            "RUNPOD_SERVERLESS_ENABLED": "true",
            "RUNPOD_SERVERLESS_ENDPOINT_ID": eid,
            "LOCAL_MODEL_ENABLED": "true",
        }
    ),
)
