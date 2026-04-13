---
title: CertificateExpired — TLS Certificate Renewal
tags: [CertificateExpired, certificate, tls, ssl, cert, expired]
recommended_tool: restart_service
---

# CertificateExpired — TLS Certificate Renewal

A TLS certificate has expired or is about to expire, causing HTTPS failures.

## Steps

1. Identify expiring certs: `kubectl get certificates -A`
2. Check cert-manager logs for renewal failures.
3. Manually trigger renewal: `kubectl annotate certificate <name> cert-manager.io/issue-once=true`
4. If cert-manager is failing, check ACME challenge DNS records.
5. As emergency measure, create self-signed cert and update Secret.
6. Restart ingress controller after cert update.
7. Verify TLS handshake succeeds after renewal.

## Common Causes

- cert-manager misconfigured or not running
- ACME DNS challenge failing (DNS propagation issues)
- Rate limit hit on Let's Encrypt
- Certificate Secret deleted manually
- Ingress annotation missing cert-manager.io/cluster-issuer
