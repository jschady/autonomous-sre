---
title: ImagePullBackOff Resolution
tags: [ImagePullBackOff, imagepull, image, registry, docker]
recommended_tool: execute_rollback
---

# ImagePullBackOff Resolution

Kubernetes cannot pull the container image, causing the pod to fail to start.

## Steps

1. Verify image name and tag: `kubectl describe pod <pod>`
2. Check image exists in registry: `docker manifest inspect <image>`
3. Verify imagePullSecret is present and valid.
4. Re-create imagePullSecret if credentials rotated.
5. Update deployment with corrected image reference.
6. Trigger rollout: `kubectl rollout restart deployment/<name>`
7. Confirm pods reach Running state.

## Common Causes

- Non-existent image tag (typo or deleted tag)
- Missing or expired imagePullSecret
- Registry authentication failure
- Private registry unreachable from the cluster
