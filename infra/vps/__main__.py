"""Pulumi program: build the FastAPI image and provision a Hetzner VPS.

Resources created:
  - docker.Image        Build + push the FastAPI app image to your registry
  - hcloud.Server       (cpx11: 2 vCPU / 4 GB RAM, ~$6/month)
  - hcloud.Firewall     (allow port 8000 from Slack IP ranges + Prometheus)
  - hcloud.FirewallAttachment

Required Pulumi config / environment variables:
  hcloud:token        Hetzner Cloud API token  (pulumi config set --secret hcloud:token <value>)
  DOCKER_REGISTRY     Registry prefix  e.g. "docker.io/youruser" or "ghcr.io/yourorg"
  DOCKER_USERNAME     Registry username
  DOCKER_PASSWORD     Registry password / token
  ENV_FILE_CONTENT    Newline-separated KEY=VALUE pairs injected into the server

Optional:
  PROMETHEUS_IP       IP of your Prometheus server (adds firewall rule)
  server_type         Hetzner server type (default: cpx11)
  location            Hetzner datacenter   (default: nbg1)
  image_tag           Docker image tag     (default: latest)
"""
from __future__ import annotations

import os

import pulumi
import pulumi_docker as docker
import pulumi_hcloud as hcloud

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

config = pulumi.Config()

DOCKER_REGISTRY: str = config.get("dockerRegistry") or os.environ.get("DOCKER_REGISTRY", "")
DOCKER_USERNAME: str = config.get("dockerUsername") or os.environ.get("DOCKER_USERNAME", "")
DOCKER_PASSWORD: str = config.get("dockerPassword") or os.environ.get("DOCKER_PASSWORD", "")
ENV_FILE_CONTENT: str = config.get("envFileContent") or os.environ.get("ENV_FILE_CONTENT", "")
PROMETHEUS_IP: str = config.get("prometheusIp") or os.environ.get("PROMETHEUS_IP", "")
SERVER_TYPE: str = config.get("serverType") or "cx23"
LOCATION: str = config.get("location") or "nbg1"
# Restrict SSH to your IP: pulumi config set allowedSshCidr "1.2.3.4/32"
ALLOWED_SSH_CIDR: str = config.get("allowedSshCidr") or os.environ.get("ALLOWED_SSH_CIDR", "")

_image_tag = config.get("imageTag") or "latest"
IMAGE_NAME = (
    f"{DOCKER_REGISTRY}/autonomous-sre:{_image_tag}"
    if DOCKER_REGISTRY
    else f"autonomous-sre:{_image_tag}"
)

# Slack outbound IP ranges (update periodically via https://api.slack.com/reference/ip-ranges)
SLACK_IP_RANGES: list[str] = [
    "18.165.0.0/16",
    "52.10.62.0/24",
    "54.86.52.0/24",
    "54.88.98.0/24",
    "34.196.188.0/22",
    "34.226.153.0/24",
]

# ---------------------------------------------------------------------------
# Step 1: Build and push the FastAPI Docker image
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

# Build context is the repo root (where the Dockerfile lives)
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

app_image = docker.Image(
    "autonomous-sre-app-image",
    image_name=IMAGE_NAME,
    build=docker.DockerBuildArgs(
        context=_repo_root,
        dockerfile=os.path.join(_repo_root, "Dockerfile"),
        platform="linux/amd64",
    ),
    registry=registry_info,
    skip_push=not bool(DOCKER_REGISTRY),
)

# ---------------------------------------------------------------------------
# Step 2: Cloud-Init user data
# ---------------------------------------------------------------------------

def _render_cloud_init(image_name: str) -> str:
    template_path = os.path.join(os.path.dirname(__file__), "cloud-init.yaml")
    with open(template_path) as f:
        template = f.read()
    # Indent each env line by 6 spaces to sit inside the write_files content block scalar.
    indented_env = "\n".join(f"      {line}" for line in ENV_FILE_CONTENT.splitlines())
    return (
        template
        .replace("${image_name}", image_name)
        .replace("      ${env_file_content}", indented_env)
    )


# Render cloud-init once the image name is known (it's a plain string here,
# but we depend on app_image so Pulumi waits for the push to finish first).
user_data = app_image.image_name.apply(_render_cloud_init)

# ---------------------------------------------------------------------------
# Step 3: Firewall
# ---------------------------------------------------------------------------

inbound_rules: list[hcloud.FirewallRuleArgs] = []

for cidr in SLACK_IP_RANGES:
    inbound_rules.append(
        hcloud.FirewallRuleArgs(
            direction="in",
            protocol="tcp",
            port="8000",
            source_ips=[cidr],
            description=f"Slack {cidr}",
        )
    )

if PROMETHEUS_IP:
    inbound_rules.append(
        hcloud.FirewallRuleArgs(
            direction="in",
            protocol="tcp",
            port="8000",
            source_ips=[f"{PROMETHEUS_IP}/32"],
            description="Prometheus scrape",
        )
    )

if not ALLOWED_SSH_CIDR:
    pulumi.log.warn(
        "allowedSshCidr is not set — SSH port 22 is open to 0.0.0.0/0. "
        "Set it with: pulumi config set allowedSshCidr YOUR.IP/32"
    )
_ssh_source_ips = [ALLOWED_SSH_CIDR] if ALLOWED_SSH_CIDR else ["0.0.0.0/0", "::/0"]
inbound_rules.append(
    hcloud.FirewallRuleArgs(
        direction="in",
        protocol="tcp",
        port="22",
        source_ips=_ssh_source_ips,
        description="SSH",
    )
)

firewall = hcloud.Firewall("autonomous-sre-fw", rules=inbound_rules)

# ---------------------------------------------------------------------------
# Step 4: Server (depends on app_image so it starts only after push)
# ---------------------------------------------------------------------------

server = hcloud.Server(
    "autonomous-sre-server",
    server_type=SERVER_TYPE,
    image="debian-12",
    location=LOCATION,
    user_data=user_data,
    labels={"app": "autonomous-sre", "managed-by": "pulumi"},
    opts=pulumi.ResourceOptions(depends_on=[app_image]),
)

hcloud.FirewallAttachment(
    "autonomous-sre-fw-attachment",
    firewall_id=firewall.id,
    server_ids=[server.id],
)

# ---------------------------------------------------------------------------
# Stack outputs
# ---------------------------------------------------------------------------

pulumi.export("image_name", app_image.image_name)
pulumi.export("server_id", server.id)
pulumi.export("server_ip", server.ipv4_address)
pulumi.export("firewall_id", firewall.id)
pulumi.export(
    "webhook_url",
    server.ipv4_address.apply(lambda ip: f"http://{ip}:8000/webhook"),
)
