---
title: DiskPressure — Node Storage Critical
tags: [DiskPressure, disk, storage, disk-pressure, node]
recommended_tool: restart_service
---

# DiskPressure — Node Storage Critical

A Kubernetes node is reporting DiskPressure, which can cause pod evictions and scheduling failures.

## Steps

1. Identify affected node: `kubectl get nodes`
2. Check disk usage: `kubectl describe node <node> | grep DiskPressure`
3. Clean up unused images: `docker system prune -a`
4. Remove old log files: `find /var/log -name '*.log' -mtime +7 -delete`
5. Evict non-critical pods from the node.
6. Cordon the node to prevent new scheduling.
7. Request node volume expansion or add additional nodes.

## Common Causes

- Accumulated container images filling disk
- Large application log files not rotated
- Persistent volume claims consuming node-local storage
- Temp files accumulating from jobs or builds
