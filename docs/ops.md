# NervNews operations guide

This guide provides practical steps for provisioning hardware, preparing local
models, and operating the ingestion stack in production-like environments.

## Environment preparation

1. **Python runtime** – Install Python 3.11. When packaging inside Docker the
   provided `Dockerfile` already selects `python:3.11-slim`.
2. **Ollama runtime** – Install [Ollama](https://ollama.com) on the host and
   ensure `ollama serve` is running. Pull the desired model (for example
   `ollama pull qwen3:30b`) before starting NervNews.
3. **Virtual environment** – Use `python -m venv .venv` and install both
   `requirements.txt` (runtime) and `requirements-dev.txt` (testing & linting).
4. **Database directory** – Create `data/` before first launch so SQLite files
   persist and can be bind-mounted into containers.

## GPU requirements

- NervNews can operate on CPU-only machines, but for production workloads a GPU
  with at least 12 GB of VRAM is recommended for 7B+ parameter models served via
  Ollama.
- Install the latest NVIDIA drivers and ensure the Ollama daemon detects your
  GPU (see `ollama help run` for environment variables such as `OLLAMA_USE_GPU`).
- When running Ollama on another host, expose it with `OLLAMA_HOST=0.0.0.0` and
  update `llm.base_url` in `config/settings.yaml` so NervNews can reach it.

## Ollama model management

1. Pull the desired models with `ollama pull <model>` (for example,
   `ollama pull qwen3:30b`). Ollama caches the weights under its own
   configuration directory.
2. Verify the runtime is healthy with `ollama ps` and test interactively using
   `ollama run <model> "ping"`.
3. Update `llm.model` and `llm.base_url` in `config/settings.yaml` (or the admin
   UI) so NervNews targets the correct model name and host.
4. The scheduler polls for configuration changes every 60 seconds; no restart is
   required after updating the admin form.
5. Enable `llm.debug_payloads` for short periods when troubleshooting failed
   generations. NervNews will log redacted request/response payloads at DEBUG
   level to help operators diagnose backend issues.

## Container orchestration

- `docker compose up --build` launches three containers (`service`, `scheduler`,
  `web`). The default ports are 8000 (API), 8080 (dashboard), and 9000/9001 for
  metrics.
- Bind mount locations:
  - `./config` → `/app/config`
  - `./data` → `/app/data`
  - `./models` → `/app/models`
- Override runtime configuration via environment variables in `docker-compose.yml`.
  For example, set `NERVNEWS_LOG_LEVEL=DEBUG` or change metrics ports.

## Monitoring & observability

- Structured logging is enabled by default in Docker (`NERVNEWS_LOG_FORMAT=json`).
- Prometheus metrics are exposed on `/metrics` for each component that sets
  `NERVNEWS_METRICS_PORT`. Scrape the endpoints from your monitoring stack.
- Key metrics include ingestion throughput, enrichment batch timing, and summary
  cycle outcomes. See `src/telemetry/metrics.py` for counter names.

## Scaling and future work

To handle bursts of articles and scale beyond a single node:

1. **Queue-based ingestion**
   - Replace direct enrichment calls with a Redis or RabbitMQ queue.
   - Scheduler publishes article IDs to the queue; multiple worker processes pull
     tasks and execute enrichment concurrently.

2. **Multi-process enrichment workers**
   - Spawn several enrichment workers per host, each sharing an Ollama backend
     pinned to available CPU/GPU resources.
   - Use a coordination layer (e.g., Celery, Dramatiq) to balance load and retry
     failed tasks without blocking the scheduler.

3. **Summary orchestration separation**
   - Move summarisation cycles to a dedicated worker pool triggered via message
     queue. This allows summarisation to scale independently of ingestion
     throughput.

4. **Back-pressure & rate limiting**
   - Track queue depth and dynamically adjust feed polling intervals when the
     system is saturated.
   - Emit alerts when enrichment latency exceeds SLA thresholds so additional
     workers can be provisioned.

5. **Horizontal sharding**
   - Partition feeds by topic or geography and assign them to different scheduler
     instances backed by a shared PostgreSQL database when SQLite becomes a
     bottleneck.

These steps provide a roadmap toward a resilient, burst-tolerant architecture
that can ingest and summarise large news volumes.
