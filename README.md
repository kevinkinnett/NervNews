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
