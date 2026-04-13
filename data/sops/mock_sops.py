"""Mock SOP (Standard Operating Procedure) catalogue for the Autonomous SRE system.

Phase 1 implementation — in-memory keyword search.
Phase 2 will replace this with pgvector similarity search.
"""
from __future__ import annotations

SOPS: list[dict] = [
    {
        "title": "CrashLoopBackOff Recovery",
        "content": (
            "1. Check pod logs: kubectl logs <pod> --previous\n"
            "2. Describe the pod for events: kubectl describe pod <pod>\n"
            "3. If OOM, increase memory limits in deployment spec.\n"
            "4. If config error, fix ConfigMap/Secret and restart pod.\n"
            "5. Restart the service: kubectl rollout restart deployment/<name>\n"
            "6. Monitor pod status for 5 minutes after restart.\n"
            "7. Escalate if pod continues crashing after 3 restarts."
        ),
        "tags": ["CrashLoopBackOff", "crashloop", "crash", "pod", "restart"],
        "recommended_tool": "restart_service",
    },
    {
        "title": "OOMKilled — Out Of Memory Recovery",
        "content": (
            "1. Identify the OOM container: kubectl describe pod <pod> | grep OOMKilled\n"
            "2. Review memory usage: kubectl top pod <pod>\n"
            "3. Increase memory requests/limits in the deployment spec.\n"
            "4. Check for memory leaks in application logs.\n"
            "5. Restart the affected pod to restore service.\n"
            "6. Set up memory alerts at 80% threshold.\n"
            "7. If recurring, escalate to engineering team for profiling."
        ),
        "tags": ["OOMKilled", "oom", "memory", "out-of-memory", "pod"],
        "recommended_tool": "restart_service",
    },
    {
        "title": "ImagePullBackOff Resolution",
        "content": (
            "1. Verify image name and tag: kubectl describe pod <pod>\n"
            "2. Check image exists in registry: docker manifest inspect <image>\n"
            "3. Verify imagePullSecret is present and valid.\n"
            "4. Re-create imagePullSecret if credentials rotated.\n"
            "5. Update deployment with corrected image reference.\n"
            "6. Trigger rollout: kubectl rollout restart deployment/<name>\n"
            "7. Confirm pods reach Running state."
        ),
        "tags": ["ImagePullBackOff", "imagepull", "image", "registry", "docker"],
        "recommended_tool": "execute_rollback",
    },
    {
        "title": "HighErrorRate — Service Degradation",
        "content": (
            "1. Check recent deployments: kubectl rollout history deployment/<name>\n"
            "2. Review error logs for root cause.\n"
            "3. If a recent deployment caused the error spike, rollback immediately.\n"
            "4. Execute rollback: kubectl rollout undo deployment/<name>\n"
            "5. Verify error rate returns to baseline within 5 minutes.\n"
            "6. If rollback does not help, check downstream dependencies.\n"
            "7. Enable maintenance mode if error rate exceeds 50%."
        ),
        "tags": ["HighErrorRate", "high-error", "error-rate", "5xx", "degradation"],
        "recommended_tool": "execute_rollback",
    },
    {
        "title": "HighLatency — Slow Response Times",
        "content": (
            "1. Check current resource usage: kubectl top pods -n <namespace>\n"
            "2. Review recent traffic patterns in APM.\n"
            "3. Scale up deployment replicas if CPU-bound.\n"
            "4. Check database query times — look for slow queries.\n"
            "5. Verify downstream service latencies.\n"
            "6. Enable horizontal pod autoscaling if not present.\n"
            "7. Rollback if latency spike correlates with recent deploy."
        ),
        "tags": ["HighLatency", "latency", "slow", "timeout", "performance"],
        "recommended_tool": "execute_rollback",
    },
    {
        "title": "ConnectionRefused — Network Connectivity",
        "content": (
            "1. Check if target service pods are running: kubectl get pods -n <namespace>\n"
            "2. Verify Service and Endpoints: kubectl describe svc <service>\n"
            "3. Test network connectivity from within the cluster.\n"
            "4. Check NetworkPolicy rules for blocking traffic.\n"
            "5. Verify port numbers match in Service spec and container.\n"
            "6. Restart the target service if pods are in bad state.\n"
            "7. Escalate to networking team if issue persists."
        ),
        "tags": ["ConnectionRefused", "connection-refused", "network", "connectivity", "port"],
        "recommended_tool": "restart_service",
    },
    {
        "title": "DiskPressure — Node Storage Critical",
        "content": (
            "1. Identify affected node: kubectl get nodes\n"
            "2. Check disk usage: kubectl describe node <node> | grep DiskPressure\n"
            "3. Clean up unused images: docker system prune -a\n"
            "4. Remove old log files: find /var/log -name '*.log' -mtime +7 -delete\n"
            "5. Evict non-critical pods from the node.\n"
            "6. Cordon the node to prevent new scheduling.\n"
            "7. Request node volume expansion or add additional nodes."
        ),
        "tags": ["DiskPressure", "disk", "storage", "disk-pressure", "node"],
        "recommended_tool": "restart_service",
    },
    {
        "title": "CertificateExpired — TLS Certificate Renewal",
        "content": (
            "1. Identify expiring certs: kubectl get certificates -A\n"
            "2. Check cert-manager logs for renewal failures.\n"
            "3. Manually trigger renewal: kubectl annotate certificate <name> cert-manager.io/issue-once=true\n"
            "4. If cert-manager is failing, check ACME challenge DNS records.\n"
            "5. As emergency measure, create self-signed cert and update Secret.\n"
            "6. Restart ingress controller after cert update.\n"
            "7. Verify TLS handshake succeeds after renewal."
        ),
        "tags": ["CertificateExpired", "certificate", "tls", "ssl", "cert", "expired"],
        "recommended_tool": "restart_service",
    },
]


def search_sops(query: str) -> list[dict]:
    """Search SOPs by keyword matching against tags and title.

    Returns up to 2 best-matching SOPs, ordered by match count descending.
    Returns empty list if no matches found.
    """
    query_lower = query.lower()
    query_tokens = query_lower.split()

    scored: list[tuple[int, dict]] = []
    for sop in SOPS:
        score = _score_sop(sop, query_lower, query_tokens)
        if score > 0:
            scored.append((score, sop))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [sop for _, sop in scored[:2]]


def _score_sop(sop: dict, query_lower: str, query_tokens: list[str]) -> int:
    """Compute a match score for a single SOP against the query.

    Scoring rules (strict keyword matching to avoid false positives):
    - Exact full-query substring in title: +5
    - Each query token that exactly equals a tag: +3
    - Each query token that exactly equals part of a hyphenated tag: +2
    """
    score = 0
    title_lower = sop["title"].lower()
    tags_lower = [t.lower() for t in sop["tags"]]

    # Strongest signal: query appears verbatim in title
    if query_lower in title_lower:
        score += 5

    for token in query_tokens:
        # Skip short or overly common words to avoid false positives
        if len(token) < 6:
            continue
        for tag in tags_lower:
            if token == tag:
                score += 3
            else:
                # Check whole-word within hyphenated tags
                tag_parts = tag.split("-")
                if token in tag_parts:
                    score += 2

    return score
