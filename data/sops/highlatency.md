---
title: HighLatency — Slow Response Times
tags: [HighLatency, latency, slow, timeout, performance]
recommended_tool: execute_rollback
---

# HighLatency — Slow Response Times

Service response times are abnormally high, degrading user experience and downstream SLAs.

## Steps

1. Check current resource usage: `kubectl top pods -n <namespace>`
2. Review recent traffic patterns in APM.
3. Scale up deployment replicas if CPU-bound.
4. Check database query times — look for slow queries.
5. Verify downstream service latencies.
6. Enable horizontal pod autoscaling if not present.
7. Rollback if latency spike correlates with recent deploy.

## Common Causes

- Sudden traffic increase exceeding capacity
- Slow database queries or missing indexes
- Memory pressure causing GC pauses
- Bad deployment introducing slow code path
- Downstream service latency cascading upstream
