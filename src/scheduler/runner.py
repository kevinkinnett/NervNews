"""Scheduling entrypoint for NervNews ingestion with hot-reloadable configuration."""
from __future__ import annotations

import logging
from typing import Iterable, List

from apscheduler.schedulers.background import BackgroundScheduler

from src.config.settings import AppSettings, FeedSettings, load_settings
from src.config.store import (
    CONFIG_LLM_MODEL_PATH,
    CONFIG_LLM_PROVIDER,
    CONFIG_SUMMARIZATION_INTERVAL,
    build_llm_settings,
    build_summarization_settings,
    ensure_seed_data,
    get_config,
    load_feed_settings,
)
from src.db.session import (
    create_engine_from_url,
    create_session_factory,
    init_db,
    session_scope,
)
from src.ingestion.extractor import ArticleExtractor
from src.ingestion.rss import RSSIngestionService
from src.llm import (
    ArticleEnrichmentService,
    LLMClient,
    LLMClientError,
    SummaryOrchestrationService,
)

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _feed_job_id(feed: FeedSettings) -> str:
    token = feed.id if feed.id is not None else feed.name
    return f"feed-{token}"


def _feeds_signature(feeds: Iterable[FeedSettings]) -> List[tuple]:
    signature: List[tuple] = []
    for feed in feeds:
        signature.append(
            (
                feed.id or feed.name,
                feed.url,
                feed.schedule_seconds,
                feed.enabled,
            )
        )
    signature.sort()
    return signature


def _sync_feed_jobs(
    scheduler: BackgroundScheduler,
    feeds: Iterable[FeedSettings],
    ingestion_service: RSSIngestionService,
    enrichment_service: ArticleEnrichmentService,
) -> None:
    desired: dict[str, FeedSettings] = {}
    for feed in feeds:
        job_id = _feed_job_id(feed)
        if not feed.enabled:
            continue
        desired[job_id] = feed

    existing = {
        job.id for job in scheduler.get_jobs() if job.id and job.id.startswith("feed-")
    }
    for job_id in existing - desired.keys():
        scheduler.remove_job(job_id)
        logger.info("Removed feed schedule %s", job_id)

    for job_id, feed in desired.items():
        scheduler.add_job(
            _run_ingestion_job,
            "interval",
            seconds=max(feed.schedule_seconds, 60),
            args=[feed, ingestion_service, enrichment_service],
            id=job_id,
            replace_existing=True,
        )
        logger.info(
            "Scheduled feed %s (%s) every %ss",
            feed.name,
            feed.url,
            feed.schedule_seconds,
        )


def _register_jobs(
    scheduler: BackgroundScheduler,
    feeds: Iterable[FeedSettings],
    summarization_interval: int,
    ingestion_service: RSSIngestionService,
    enrichment_service: ArticleEnrichmentService,
    summarization_service: SummaryOrchestrationService,
) -> None:
    _sync_feed_jobs(scheduler, feeds, ingestion_service, enrichment_service)

    scheduler.add_job(
        summarization_service.run_cycle,
        "interval",
        seconds=max(summarization_interval, 300),
        id="summaries-hourly",
        replace_existing=True,
    )
    logger.info(
        "Scheduled summarisation job every %ss",
        summarization_interval,
    )


def _run_ingestion_job(
    feed: FeedSettings,
    ingestion_service: RSSIngestionService,
    enrichment_service: ArticleEnrichmentService,
) -> None:
    new_articles = ingestion_service.ingest(feed)
    if new_articles:
        logger.info(
            "Feed %s produced %d new articles. IDs=%s",
            feed.name,
            len(new_articles),
            new_articles,
        )
        try:
            enrichment_service.enrich_articles(new_articles)
        except LLMClientError:
            logger.exception("LLM enrichment failed for feed %s", feed.name)


def run_scheduler(settings: AppSettings | None = None) -> BackgroundScheduler:
    """Start the APScheduler-based ingestion pipeline."""
    _configure_logging()
    if settings is None:
        settings = load_settings()

    engine = create_engine_from_url(settings.database_url)
    init_db(engine)
    session_factory = create_session_factory(engine)

    ensure_seed_data(session_factory, settings)

    extractor = ArticleExtractor(timeout=settings.request_timeout, user_agent=settings.user_agent)
    llm_client = LLMClient(settings.llm)
    enrichment_service = ArticleEnrichmentService(
        session_factory=session_factory,
        llm_client=llm_client,
    )
    summarization_service = SummaryOrchestrationService(
        session_factory=session_factory,
        llm_client=llm_client,
        settings=settings.summarization,
    )
    ingestion_service = RSSIngestionService(session_factory=session_factory, extractor=extractor)

    with session_scope(session_factory) as session:
        feed_configs = load_feed_settings(session)
        if not feed_configs:
            feed_configs = settings.feeds

        interval_override = int(
            get_config(
                session,
                CONFIG_SUMMARIZATION_INTERVAL,
                settings.summarization.interval_seconds,
            )
        )
        summarization_settings = build_summarization_settings(
            settings.summarization,
            interval_override,
        )
        summarization_service.update_settings(summarization_settings)

        provider_override = get_config(session, CONFIG_LLM_PROVIDER, settings.llm.provider)
        model_override = get_config(session, CONFIG_LLM_MODEL_PATH, settings.llm.model_path)
        llm_settings = build_llm_settings(settings.llm, provider_override, model_override)
        llm_client.update_settings(llm_settings)

    scheduler = BackgroundScheduler()
    _register_jobs(
        scheduler,
        feed_configs,
        summarization_service.settings.interval_seconds,
        ingestion_service,
        enrichment_service,
        summarization_service,
    )

    state = {
        "feed_signature": _feeds_signature(feed_configs),
        "summary_interval": summarization_service.settings.interval_seconds,
        "llm_settings": llm_client.settings,
    }

    def _reload_configuration() -> None:
        with session_scope(session_factory) as session:
            feeds = load_feed_settings(session)
            if not feeds:
                feeds = settings.feeds

            signature = _feeds_signature(feeds)
            if signature != state["feed_signature"]:
                _sync_feed_jobs(scheduler, feeds, ingestion_service, enrichment_service)
                state["feed_signature"] = signature

            interval_value = int(
                get_config(
                    session,
                    CONFIG_SUMMARIZATION_INTERVAL,
                    state["summary_interval"],
                )
            )
            if interval_value != state["summary_interval"]:
                summarization_settings_new = build_summarization_settings(
                    settings.summarization,
                    interval_value,
                )
                summarization_service.update_settings(summarization_settings_new)
                scheduler.reschedule_job(
                    "summaries-hourly",
                    trigger="interval",
                    seconds=max(summarization_settings_new.interval_seconds, 300),
                )
                state["summary_interval"] = summarization_settings_new.interval_seconds
                logger.info(
                    "Rescheduled summarisation interval to %ss",
                    summarization_settings_new.interval_seconds,
                )

            provider = get_config(session, CONFIG_LLM_PROVIDER, settings.llm.provider)
            model_path = get_config(session, CONFIG_LLM_MODEL_PATH, settings.llm.model_path)
            llm_settings_new = build_llm_settings(settings.llm, provider, model_path)
            if llm_settings_new != state["llm_settings"]:
                llm_client.update_settings(llm_settings_new)
                state["llm_settings"] = llm_settings_new

    scheduler.add_job(
        _reload_configuration,
        "interval",
        seconds=60,
        id="config-reloader",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Scheduler started with %d jobs", len(scheduler.get_jobs()))
    return scheduler


__all__ = ["run_scheduler"]
