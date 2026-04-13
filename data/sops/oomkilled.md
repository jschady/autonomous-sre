---
title: OOMKilled — Out Of Memory Recovery
tags: [OOMKilled, oom, memory, out-of-memory, pod]
recommended_tool: restart_service
---

# OOMKilled — Out Of Memory Recovery

The container was killed by the Linux OOM killer because it exceeded its memory limit.

## Steps

1. Identify the OOM container: `kubectl describe pod <pod> | grep OOMKilled`
2. Review memory usage: `kubectl top pod <pod>`
3. Increase memory requests/limits in the deployment spec.
4. Check for memory leaks in application logs.
5. Restart the affected pod to restore service.
6. Set up memory alerts at 80% threshold.
7. If recurring, escalate to engineering team for profiling.

## Common Causes

- Memory leak in application code
- Insufficient memory limits set in deployment
- Large in-memory data processing without streaming
- Cache unbounded growth
