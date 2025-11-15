# NervNews

NervNews provides an RSS ingestion pipeline that polls configured feeds, extracts
full article content, enriches articles with LLM generated metadata, and stores
the results in SQLite for downstream processing.

## Environment setup

- **Python version** – Python 3.11 is recommended. Create an isolated
  environment before installing dependencies:

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  pip install -r requirements-dev.txt  # optional: testing utilities
  ```

- **GPU & LLM acceleration** – NervNews runs on CPU-only hosts but benefits from
  a CUDA capable GPU when serving larger Ollama models. Ensure the Ollama daemon
  is running with your chosen model (for example `ollama pull qwen3:30b`) and
  see [`docs/ops.md`](docs/ops.md) for runtime guidance.

- **Configuration** – Edit `config/settings.yaml` (or point
  `NERVNEWS_SETTINGS` at an alternate file). The default configuration stores
  data in `data/nervnews.db` so that the SQLite database can be persisted or
  mounted inside containers.

## Local workflow

1. **Configure feeds** – Add RSS/Atom feeds in `config/settings.yaml` under the
   `feeds` section.
2. **Seed defaults** – Optionally copy `config/seed.example.yaml` and adjust it
   before the first run; the scheduler automatically loads it when the database
   is empty.
3. **Run the scheduler**

   ```bash
   python -m src.main
   ```

   The APScheduler loop polls each feed, records ingestion logs, and triggers
   LLM enrichment and summarisation cycles.

4. **Launch the dashboard**

   ```bash
   uvicorn src.web.main:app --reload
   ```

   Navigate to `http://127.0.0.1:8000/` for newsroom summaries and
   `http://127.0.0.1:8000/admin` for configuration. Admin changes are stored in
   SQLite and reloaded by the scheduler every 60 seconds.

## Testing

Run the test suite with:

```bash
pytest
```

The tests cover ingestion, enrichment, and summarisation pipelines and use an
in-memory SQLite database.

## Telemetry

- **Structured logging** – set `NERVNEWS_LOG_FORMAT=json` (or `structured`) to
  emit JSON logs. Adjust `NERVNEWS_LOG_LEVEL` to control verbosity.
- **Prometheus metrics** – expose counters and histograms by defining
  `NERVNEWS_METRICS_PORT`. Metrics endpoints start automatically on the chosen
  port (for example `http://localhost:9000/metrics`).

## Docker & Compose

The provided `Dockerfile` packages the scheduler, FastAPI service, and dashboard
into a single image. `docker-compose.yml` wires three containers together:

- `service` – FastAPI backend + metrics exporter on port 8000.
- `scheduler` – background ingestion and summarisation runner.
- `web` – dashboard-focused instance listening on port 8080.

All services mount `./config` and `./data` so that configuration and the SQLite
database persist between restarts. Ollama stores model weights separately in its
own data directory.

```bash
docker compose up --build
```

Refer to [`docs/ops.md`](docs/ops.md) for GPU provisioning, model management,
and operational runbooks.

## Scaling roadmap

The project is designed to scale via queued work distribution and horizontal
workers. A high-level plan is included in [`docs/ops.md`](docs/ops.md#scaling-and-future-work),
covering Redis-based ingestion queues, multi-process enrichment workers, and
back-pressure controls for bursty feed updates.

## Seed configuration

`config/seed.example.yaml` contains a ready-to-use template that mirrors the
expected YAML structure. Copy it to a new environment and adjust feed URLs,
model names, and the user profile text before the first run.

## Relevance grading and user profile

Summaries now capture a user profile that steers the relevance critic. The
profile can be edited from the admin UI (minimum 40 characters). Once a summary
is generated, the system produces:

- An overall relevance and criticality rating for the briefing.
- Per-key-point badges highlighting importance, recommended action, and the
  supporting rationale.

Ratings are stored alongside each summary and surfaced on the dashboard and the
detail page. When no profile is present, the critic defaults to a general
newsroom audience.
