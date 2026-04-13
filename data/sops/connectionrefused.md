---
title: ConnectionRefused — Network Connectivity
tags: [ConnectionRefused, connection-refused, network, connectivity, port]
recommended_tool: restart_service
---

# ConnectionRefused — Network Connectivity

Service is unable to establish connections, indicating network or service availability issues.

## Steps

1. Check if target service pods are running: `kubectl get pods -n <namespace>`
2. Verify Service and Endpoints: `kubectl describe svc <service>`
3. Test network connectivity from within the cluster.
4. Check NetworkPolicy rules for blocking traffic.
5. Verify port numbers match in Service spec and container.
6. Restart the target service if pods are in bad state.
7. Escalate to networking team if issue persists.

## Common Causes

- Target pod not running or not ready
- Service selector mismatch
- NetworkPolicy blocking traffic
- Port misconfiguration in Service spec
- DNS resolution failure within cluster
