---
title: HighErrorRate — Service Degradation
tags: [HighErrorRate, high-error, error-rate, 5xx, degradation]
recommended_tool: execute_rollback
---

# HighErrorRate — Service Degradation

Service is returning a high rate of 5xx errors, indicating degraded service health.

## Steps

1. Check recent deployments: `kubectl rollout history deployment/<name>`
2. Review error logs for root cause.
3. If a recent deployment caused the error spike, rollback immediately.
4. Execute rollback: `kubectl rollout undo deployment/<name>`
5. Verify error rate returns to baseline within 5 minutes.
6. If rollback does not help, check downstream dependencies.
7. Enable maintenance mode if error rate exceeds 50%.

## Common Causes

- Bad deployment introducing a regression
- Downstream dependency failure (database, external API)
- Configuration change causing errors
- Resource exhaustion (connection pool, thread pool)
