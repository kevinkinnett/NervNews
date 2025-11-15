# NervNews operations guide

This guide provides practical steps for provisioning hardware, preparing local
models, and operating the ingestion stack in production-like environments.

## Environment preparation

1. **Python runtime** – Install Python 3.11. When packaging inside Docker the
   provided `Dockerfile` already selects `python:3.11-slim`.
2. **System dependencies** – Ensure `libstdc++`, `libgomp`, and `build-essential`
   are installed on bare-metal hosts if you plan to compile llama.cpp kernels.
3. **Virtual environment** – Use `python -m venv .venv` and install both
   `requirements.txt` (runtime) and `requirements-dev.txt` (testing & linting).
4. **Database directory** – Create `data/` before first launch so SQLite files
   persist and can be bind-mounted into containers.

## GPU requirements

- NervNews can operate on CPU-only machines, but for production workloads a GPU
  with at least 12 GB of VRAM is recommended for 7B+ parameter models.
- Install the latest NVIDIA drivers and CUDA toolkit that match the
  `llama-cpp-python` wheel. For container deployments ensure the host runtime
  provides `nvidia-container-toolkit` so CUDA devices are available to Docker.
- Configure llama.cpp thread counts via `llm.threads` in `config/settings.yaml`;
  when running on GPU set this to the number of physical CPU cores used for
  offloading tasks (e.g., 8).

## Model quantisation workflow

1. Download a base GGUF checkpoint (e.g., `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf`).
2. Place the file in `models/` and update `llm.model_path` in
   `config/settings.yaml`.
3. To perform custom quantisation:
   - Clone `https://github.com/ggerganov/llama.cpp` and build the quantisation
     tools (`make quantize`).
   - Convert your original checkpoint to GGUF if required (`convert-hf-to-gguf`).
   - Run `./quantize original.gguf custom.Q4_K_M.gguf Q4_K_M` and copy the output
     into `models/`.
4. Restart the scheduler to reload settings; new quantised models are loaded on
   demand when enrichment or summarisation jobs run.

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
   - Spawn several enrichment workers per host, each with its own llama.cpp
     instance pinned to available CPU/GPU resources.
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
