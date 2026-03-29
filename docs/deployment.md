# MnemoCUDA Deployment Guide

## Quick Start (Development)

```bash
./build/mnemo_server /path/to/model --http 8095 --warmup light
```

This binds to **127.0.0.1** (localhost only) by default.

## Production Deployment

MnemoCUDA is designed to run behind a reverse proxy for production use.

### Architecture

```
Internet → Nginx/Envoy (TLS, auth, rate limit) → MnemoCUDA (localhost:8095)
```

### 1. Start MnemoCUDA

```bash
# Set auth token via environment variable (avoids exposure in ps)
export MNEMO_AUTH_TOKEN=$(openssl rand -hex 32)

./build/mnemo_server /path/to/model \
    --http 8095 \
    --auth $MNEMO_AUTH_TOKEN \
    --context 8192 \
    --warmup full
```

**Do NOT use `--bind 0.0.0.0`** in production unless behind a firewall. The default `127.0.0.1` ensures the server is only accessible via the reverse proxy.

### 2. Nginx Configuration

```nginx
upstream mnemo {
    server 127.0.0.1:8095;
}

server {
    listen 443 ssl;
    server_name inference.example.com;

    ssl_certificate /etc/ssl/certs/inference.pem;
    ssl_certificate_key /etc/ssl/private/inference.key;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=inference:10m rate=10r/m;

    location /v1/completions {
        limit_req zone=inference burst=5;
        proxy_pass http://mnemo;
        proxy_set_header Host $host;
        proxy_read_timeout 300s;  # MoE inference can be slow on first request
    }

    # Health checks (no rate limit)
    location /live   { proxy_pass http://mnemo; }
    location /ready  { proxy_pass http://mnemo; }
    location /status { proxy_pass http://mnemo; }
    location /health { proxy_pass http://mnemo; }
    location /heat   { proxy_pass http://mnemo; }

    # SSE streaming needs special proxy config
    location /v1/completions {
        proxy_pass http://mnemo;
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }
}
```

### 3. Health Checks

| Endpoint | Purpose | Auth Required |
|----------|---------|:---:|
| `GET /live` | Process alive | Yes* |
| `GET /ready` | Model loaded, ready/busy status | Yes* |
| `GET /status` | Detailed: VRAM, cache, heat, last TTFT | Yes* |
| `GET /health` | Alias for /ready (backward compat) | Yes* |

*Auth is required on all endpoints when `--auth` is set. For Kubernetes probes, either exempt health endpoints at the proxy level or use a separate unauthenticated internal port.

### 4. Monitoring

The `/status` endpoint returns operational metrics:

```json
{
    "status": "ready",
    "model": "MnemoCUDA: 235B MoE (K=8), 2 GPUs, 2048 MB resident/GPU",
    "gpus": 2,
    "vram_mb": 45000,
    "resident_mb": 5120,
    "heat_tokens": 150000,
    "pinning_active": true,
    "active_experts": 8500,
    "cache_slots": 3690,
    "cache_used": 3500,
    "last_ttft": 9.3,
    "last_tok_s": 5.2
}
```

Per-request metrics are logged to stderr with structured format:

```
2026-03-29 14:30:00.123 [INFO] req=42 prompt=128 gen=256 ttft=9.300s tok/s=5.2 total=58.5s
```

### 5. Systemd Service

```ini
[Unit]
Description=MnemoCUDA Inference Server
After=network.target

[Service]
Type=simple
User=mnemo
Environment=MNEMO_AUTH_TOKEN=your_secret_here
ExecStart=/opt/mnemo/build/mnemo_server /data/models/qwen3-235b --http 8095 --warmup full
Restart=on-failure
RestartSec=30
LimitMEMLOCK=infinity
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3060 12GB | RTX 4090 24GB + RTX 5090 32GB |
| NVMe SSD | PCIe 3.0, 2 GB/s | PCIe 4.0+, 5+ GB/s |
| RAM | 32 GB | 64 GB (for page cache) |
| CPU | 4 cores | 8+ cores (I/O pool) |

## Tuning

| Parameter | Effect |
|-----------|--------|
| `--context 4096` | Less KV VRAM = more expert cache = higher hit rate |
| `--context 32768` | More context but lower cache hit rate |
| `--warmup full` | Best first-request performance, slower startup |
| `--warmup off` | Fastest startup, cold first requests |

## Security Checklist

- [ ] Bind to 127.0.0.1 (default)
- [ ] Set `MNEMO_AUTH_TOKEN` or `--auth`
- [ ] TLS via reverse proxy
- [ ] Rate limiting at proxy
- [ ] Firewall rules for GPU host
- [ ] Log monitoring for errors
