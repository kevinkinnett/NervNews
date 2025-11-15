"""Prometheus-friendly metrics collection with graceful fallbacks."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency path
    from prometheus_client import Counter, Histogram, start_http_server
except Exception:  # pragma: no cover - when prometheus-client is unavailable
    Counter = None  # type: ignore
    Histogram = None  # type: ignore

    def start_http_server(*_args, **_kwargs):  # type: ignore[override]
        raise RuntimeError("prometheus-client is not installed")


@dataclass
class IngestionEvent:
    feed: str
    article_count: int
    duration_seconds: float
    status: str


@dataclass
class EnrichmentEvent:
    attempted: int
    successes: int
    failures: int
    duration_seconds: float


@dataclass
class SummaryEvent:
    article_count: int
    duration_seconds: float
    status: str


class MetricsCollector:
    """Centralised metrics registry for the ingestion pipeline."""

    def __init__(self) -> None:
        self._prometheus_enabled = Counter is not None and Histogram is not None
        self._exporter_started = False

        if self._prometheus_enabled:
            self._ingestion_articles = Counter(
                "nervnews_ingestion_articles_total",
                "Number of articles discovered per feed",
                labelnames=("feed", "status"),
            )
            self._ingestion_duration = Histogram(
                "nervnews_ingestion_duration_seconds",
                "Duration of feed polling in seconds",
                labelnames=("feed", "status"),
                buckets=(0.5, 1, 2, 5, 10, 30, 60, 120),
            )
            self._enrichment_articles = Counter(
                "nervnews_enrichment_articles_total",
                "Number of articles processed by the enrichment stage",
                labelnames=("result",),
            )
            self._enrichment_batch_duration = Histogram(
                "nervnews_enrichment_batch_duration_seconds",
                "Batch enrichment duration in seconds",
                buckets=(0.5, 1, 2, 5, 10, 30, 60),
            )
            self._summary_cycles = Counter(
                "nervnews_summary_cycles_total",
                "Summarisation cycles executed",
                labelnames=("status",),
            )
            self._summary_duration = Histogram(
                "nervnews_summary_cycle_duration_seconds",
                "Duration of summarisation cycles in seconds",
                labelnames=("status",),
                buckets=(1, 2, 5, 10, 30, 60, 120, 300),
            )
            self._summary_articles = Counter(
                "nervnews_summary_articles_total",
                "Articles evaluated during summarisation",
                labelnames=("status",),
            )
        else:  # pragma: no cover - exercised implicitly when dependency missing
            self._ingestion_articles = None
            self._ingestion_duration = None
            self._enrichment_articles = None
            self._enrichment_batch_duration = None
            self._summary_cycles = None
            self._summary_duration = None
            self._summary_articles = None

        self.last_ingestion: Optional[IngestionEvent] = None
        self.last_enrichment: Optional[EnrichmentEvent] = None
        self.last_summary_cycle: Optional[SummaryEvent] = None

    def enable_exporter(self, port: int) -> bool:
        """Start the Prometheus HTTP exporter if dependencies are present."""

        if not self._prometheus_enabled:
            logger.warning("Prometheus metrics requested but prometheus-client is not installed")
            return False
        if self._exporter_started:
            return True
        start_http_server(port)
        self._exporter_started = True
        logger.info("Prometheus metrics exporter started", extra={"event": "metrics.started", "port": port})
        return True

    def record_ingestion(self, feed: str, article_count: int, duration_seconds: float, status: str) -> None:
        self.last_ingestion = IngestionEvent(feed, article_count, duration_seconds, status)
        if not self._prometheus_enabled:
            return
        self._ingestion_articles.labels(feed=feed, status=status).inc(article_count)
        self._ingestion_duration.labels(feed=feed, status=status).observe(duration_seconds)

    def record_enrichment_batch(self, *, attempted: int, successes: int, failures: int, duration_seconds: float) -> None:
        self.last_enrichment = EnrichmentEvent(attempted, successes, failures, duration_seconds)
        if not self._prometheus_enabled:
            return
        if successes:
            self._enrichment_articles.labels(result="success").inc(successes)
        if failures:
            self._enrichment_articles.labels(result="failure").inc(failures)
        self._enrichment_batch_duration.observe(duration_seconds)

    def record_summary_cycle(self, *, article_count: int, duration_seconds: float, status: str) -> None:
        self.last_summary_cycle = SummaryEvent(article_count, duration_seconds, status)
        if not self._prometheus_enabled:
            return
        self._summary_cycles.labels(status=status).inc()
        self._summary_duration.labels(status=status).observe(duration_seconds)
        if article_count:
            self._summary_articles.labels(status=status).inc(article_count)

    def reset(self) -> None:
        """Reset cached inspection state (primarily for tests)."""

        self.last_ingestion = None
        self.last_enrichment = None
        self.last_summary_cycle = None


metrics = MetricsCollector()


def configure_metrics_from_env() -> None:
    """Start metrics exporter when ``NERVNEWS_METRICS_PORT`` is defined."""

    port_value = os.getenv("NERVNEWS_METRICS_PORT")
    if not port_value:
        return
    try:
        port = int(port_value)
    except ValueError:
        logger.warning(
            "Invalid NERVNEWS_METRICS_PORT value; expected integer", extra={"event": "metrics.invalid_port", "value": port_value}
        )
        return
    metrics.enable_exporter(port)


__all__ = ["configure_metrics_from_env", "metrics", "MetricsCollector"]
