"""Scheduling entrypoint for NervNews ingestion."""
from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from src.config.settings import AppSettings, FeedSettings, load_settings
from src.db.session import create_engine_from_url, create_session_factory, init_db
from src.ingestion.extractor import ArticleExtractor
from src.ingestion.rss import RSSIngestionService

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _register_jobs(
    scheduler: BackgroundScheduler,
    settings: AppSettings,
    ingestion_service: RSSIngestionService,
) -> None:
    for feed in settings.feeds:
        if not feed.enabled:
            logger.info("Skipping disabled feed %s", feed.name)
            continue

        scheduler.add_job(
            _run_ingestion_job,
            "interval",
            seconds=feed.schedule_seconds,
            args=[feed, ingestion_service],
            id=f"feed-{feed.name}",
            replace_existing=True,
        )
        logger.info(
            "Scheduled feed %s (%s) every %ss", feed.name, feed.url, feed.schedule_seconds
        )


def _run_ingestion_job(feed: FeedSettings, ingestion_service: RSSIngestionService) -> None:
    new_articles = ingestion_service.ingest(feed)
    if new_articles:
        logger.info(
            "Feed %s produced %d new articles. IDs=%s",
            feed.name,
            len(new_articles),
            new_articles,
        )


def run_scheduler(settings: AppSettings | None = None) -> BackgroundScheduler:
    """Start the APScheduler-based ingestion pipeline."""
    _configure_logging()
    if settings is None:
        settings = load_settings()

    engine = create_engine_from_url(settings.database_url)
    init_db(engine)
    session_factory = create_session_factory(engine)

    extractor = ArticleExtractor(timeout=settings.request_timeout, user_agent=settings.user_agent)
    ingestion_service = RSSIngestionService(session_factory=session_factory, extractor=extractor)

    scheduler = BackgroundScheduler()
    _register_jobs(scheduler, settings, ingestion_service)
    scheduler.start()
    logger.info("Scheduler started with %d jobs", len(scheduler.get_jobs()))
    return scheduler


__all__ = ["run_scheduler"]
