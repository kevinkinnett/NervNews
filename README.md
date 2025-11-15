# NervNews

NervNews provides an RSS ingestion pipeline that polls configured feeds, extracts
full article content, and stores the results in SQLite for downstream
processing.

## Getting Started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure feeds**

   Edit `config/settings.yaml` (or provide an alternative path via the
   `NERVNEWS_SETTINGS` environment variable) to list the feeds you want to poll,
   control their schedule, and define metadata.

3. **Run the scheduler**

   ```bash
   python -m src.main
   ```

   The scheduler uses APScheduler to poll each enabled feed at its configured
   interval. Newly ingested article IDs are recorded in `article_ingestion_logs`
   for downstream processing.

4. **Launch the dashboard (optional)**

   ```bash
   uvicorn src.web.main:app --reload
   ```

   The FastAPI dashboard exposes `http://127.0.0.1:8000/` for the newsroom
   summary view and `http://127.0.0.1:8000/admin` for configuration. All changes
   made through the admin forms are persisted to the database and picked up by
   the running scheduler within 60 seconds without downtime.

## Seed configuration

`config/seed.example.yaml` contains a ready-to-use template that mirrors the
expected YAML structure. Copy it to a new environment and adjust feed URLs,
model paths, and the user profile text before the first run.

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
