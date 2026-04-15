# Phase 4: Production Deployment Guide

Target cost: **~$8–12/month** (Hetzner VPS ~$6 + Supabase free + RunPod pay-per-second).

---

## Architecture

```
Slack (operator)
    ↕  Block Kit buttons
VPS (Hetzner, ~$6/mo)
  └─ FastAPI container  ──→  Supabase Postgres ($0, free tier)
  └─ /webhook               ├─ resolved_incidents (pgvector)
  └─ /status/{id}           ├─ alert_costs
  └─ /slack/interactive     └─ LangGraph checkpoints (durable state)
         ↓ (on approval)
  RunPod Serverless (~$0.0002/s)
  └─ vLLM + Llama 3.1-8B (baked in image, ~20s cold start)
```

---

## Phase 4A: Supabase (Data Layer)

### 1. Create Supabase project

1. Go to [supabase.com](https://supabase.com) → New project
2. In **Database** → **Extensions**, enable **vector** (pgvector)
3. In the **SQL Editor**, paste and run `infra/supabase/bootstrap.sql`

### 2. Get the connection string

Settings → Database → **Connection string** → **Transaction mode** (port 6543):

```
postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
```

Set in your `.env`:

```env
POSTGRES_DSN=postgresql://postgres.[ref]:[pw]@aws-0-[region].pooler.supabase.com:6543/postgres
```

### Notes

- **Port 6543** = Transaction Mode pooler (required for connection pooling)
- **Port 5432** = Session Mode (needed if using `LISTEN/NOTIFY` — not required here)
- SSL is enforced by Supabase; the DSN does not need `?sslmode=require` explicitly
- The `AsyncPostgresSaver.setup()` call in the FastAPI lifespan creates LangGraph
  checkpoint tables automatically; the bootstrap SQL includes them for reference

---

## Phase 4B: Hetzner VPS (Compute Layer)

### 1. Set Pulumi config

```bash
cd infra/vps
pulumi stack init prod
pulumi config set hcloudToken      <your-hetzner-api-token>
pulumi config set dockerRegistry   docker.io/<youruser>
pulumi config set dockerUsername   <dockerhub-username>
pulumi config set dockerPassword   <dockerhub-token>
pulumi config set prometheusIp     <your-prometheus-server-ip>   # optional
pulumi config set envFileContent   "$(cat .env | grep -v '^#' | grep -v '^$')"
```

### 2. Deploy

```bash
pip install -r requirements.txt
pulumi up
```

Pulumi exports:
- `server_ip` — your VPS IP address
- `webhook_url` — `http://<ip>:8000/webhook`

### 4. Create a Slack App

1. [api.slack.com/apps](https://api.slack.com/apps) → Create New App → From scratch
2. **OAuth & Permissions** → Scopes: `chat:write`, `channels:read`
3. Install to workspace → copy **Bot User OAuth Token** (`xoxb-...`)
4. **Basic Information** → **Signing Secret** → copy
5. **Interactivity & Shortcuts** → enable → set Request URL:
   `http://<server_ip>:8000/slack/interactive`

Set in `.env`:
```env
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
SLACK_CHANNEL_ID=C01234567
```

### Firewall notes

The Pulumi program allows inbound TCP 8000 from Slack IP ranges and your
Prometheus server only. Update `SLACK_IP_RANGES` in `infra/vps/__main__.py`
if Slack's published ranges change (check [api.slack.com/reference/ip-ranges](https://api.slack.com/reference/ip-ranges)).

---

## Phase 4C: RunPod Serverless (LLM Brain)

### 1. Build the baked-weights image (~15 GB)

```bash
cd infra/runpod
docker build \
  --build-arg HF_TOKEN=hf_<your_token> \
  -f Dockerfile.serverless \
  -t docker.io/<youruser>/autonomous-sre-vllm-serverless:latest \
  .
docker push docker.io/<youruser>/autonomous-sre-vllm-serverless:latest
```

> **Note:** The push will be large (~15 GB). Use a registry in the same
> region as your RunPod deployment to minimise cold-start pull time.

### 2. Deploy the Serverless endpoint

```bash
cd infra/runpod
pulumi stack init prod
pulumi config set runpodApiKey     <your-runpod-api-key>
pulumi config set dockerRegistry   docker.io/<youruser>
pulumi config set dockerUsername   <dockerhub-username>
pulumi config set dockerPassword   <dockerhub-token>
pulumi config set hfToken          hf_<your-token>   # optional
pulumi up
```

Copy the exported `endpoint_id` and set in `.env`:

```env
RUNPOD_SERVERLESS_ENABLED=true
RUNPOD_SERVERLESS_ENDPOINT_ID=<endpoint_id>
LOCAL_MODEL_ENABLED=true
RUNPOD_COLD_START_TIMEOUT=60
```

### Cold-start behaviour

- First request after idle: ~20 second GPU wake-up
- Slack receives an immediate "Processing..." acknowledgement
- Final result is posted via Slack's `response_url` once inference completes
- Subsequent requests within the `idle_timeout` window (30s) start immediately

### Cost calculation

| Scenario | Cost |
|----------|------|
| 10 alerts/day × 30s inference | 300s × $0.0002 = **$0.06/day** |
| 100 alerts/day × 30s inference | 3000s × $0.0002 = **$0.60/day** |
| Idle (most of the time) | **$0.00** |

---

## Phase 4D: Slack Bot Approval Flow

When an alert reaches the `human_gate` node, the bot sends:

```
⚠️  Alert: PodCrashLooping

Severity:  CRITICAL        Alert ID:  alert-abc-123

─────────────────────────────────────────────

🔍 Analysis
Logs show ConnectionRefusedError to the database on startup.

🔧 Recommended Action
kubectl rollout restart deployment/checkout-api -n checkout

─────────────────────────────────────────────

[Approve ✓]   [Reject ✗]
```

Clicking **Approve** or **Reject** POSTs to `/slack/interactive`, which:
1. Verifies the HMAC signature using `SLACK_SIGNING_SECRET`
2. Parses the button's `action_id` (`approve_<alert_id>` / `reject_<alert_id>`)
3. Resumes the LangGraph workflow from the Supabase checkpoint
4. Updates the Slack message with the outcome

---

## Success Criteria Verification

### Durable Resumption

```bash
# Submit an alert
curl -X POST http://<vps>:8000/webhook \
  -H "Content-Type: application/json" \
  -d '{"alertname": "TestDurable", "status": "firing", "labels": {}}'

# Note the alert_id. Wait 10+ hours. Then approve via Slack or:
curl -X POST http://<vps>:8000/slack/interactive \
  -H "Content-Type: application/json" \
  -d '{"alert_id": "<alert_id>", "approved": true}'
```

### Security (HMAC rejection)

```bash
# Should return 403
curl -X POST http://<vps>:8000/slack/interactive \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d 'payload={"type":"block_actions","actions":[]}' \
  # (no X-Slack-Signature header)
```

### Cost verification

After a week of usage, check RunPod dashboard → Serverless → Usage.
Compare against the ~$6 Hetzner invoice and $0 Supabase bill.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Graph always uses MemorySaver | `POSTGRES_DSN` empty or wrong port | Use port 6543, check URL format |
| Slack 403 on interactive | Wrong `SLACK_SIGNING_SECRET` | Re-copy from Slack App → Basic Information |
| RunPod job never completes | GPU cold start > 60s | Increase `RUNPOD_COLD_START_TIMEOUT` |
| Slack timeout after approval | Response takes > 3s | Already handled: bot responds immediately, posts result via `response_url` |
| VPS webhook not reachable | Firewall blocked | Check your source IP is in Slack's published IP ranges |
