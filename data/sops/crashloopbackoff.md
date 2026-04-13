---
title: CrashLoopBackOff Recovery
tags: [CrashLoopBackOff, crashloop, crash, pod, restart]
recommended_tool: restart_service
---

# CrashLoopBackOff Recovery

A pod that enters CrashLoopBackOff is restarting repeatedly because the container keeps failing on startup.

## Steps

1. Check pod logs: `kubectl logs <pod> --previous`
2. Describe the pod for events: `kubectl describe pod <pod>`
3. If OOM, increase memory limits in deployment spec.
4. If config error, fix ConfigMap/Secret and restart pod.
5. Restart the service: `kubectl rollout restart deployment/<name>`
6. Monitor pod status for 5 minutes after restart.
7. Escalate if pod continues crashing after 3 restarts.

## Common Causes

- Application startup failure (missing env var, bad config)
- OOMKilled immediately after start
- Liveness probe misconfiguration causing premature termination
- Missing secrets or ConfigMaps
