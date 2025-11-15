"""RSS polling and ingestion services."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import List, Optional

import html
import json
import re

import feedparser
from sqlalchemy import or_
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import FeedSettings
from src.db.models import Article, ArticleIngestionLog, Feed
from src.ingestion.extractor import ArticleExtractor, ExtractedArticle
from src.telemetry import metrics

logger = logging.getLogger(__name__)

TAG_RE = re.compile(r"<[^>]+>")


def _sanitize_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    # Remove simple HTML tags while keeping the textual content.
    stripped = TAG_RE.sub("", html.unescape(text))
    stripped = stripped.strip()
    return stripped or None


def _parse_datetime(entry: feedparser.FeedParserDict) -> Optional[datetime]:
    parsed = getattr(entry, "published_parsed", None) or entry.get("published_parsed")
    if not parsed:
        parsed = getattr(entry, "updated_parsed", None) or entry.get("updated_parsed")
    if not parsed:
        return None
    try:
        return datetime.fromtimestamp(time.mktime(parsed), tz=timezone.utc)
    except Exception:  # pragma: no cover - fallback when timestamp conversion fails
        return None


class RSSIngestionService:
    """Service responsible for polling feeds and persisting new articles."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        extractor: ArticleExtractor,
    ) -> None:
        self._session_factory = session_factory
        self._extractor = extractor

    def ingest(self, feed_config: FeedSettings) -> List[int]:
        """Poll the feed and return IDs of newly created articles."""
        start_time = time.perf_counter()
        article_count = 0
        status = "success"
        logger.info(
            "Polling feed %s",
            feed_config.url,
            extra={"event": "feed.poll", "feed": feed_config.name, "url": feed_config.url},
        )
        parsed = feedparser.parse(feed_config.url)
        if parsed.bozo:
            logger.warning(
                "Feed parsing issues encountered for %s: %s",
                feed_config.url,
                parsed.bozo_exception,
                extra={"event": "feed.parse_warning", "feed": feed_config.name},
            )

        entries = getattr(parsed, "entries", []) or []
        if not entries:
            logger.info(
                "No entries found for %s",
                feed_config.url,
                extra={"event": "feed.empty", "feed": feed_config.name},
            )
            status = "empty"
            return []

        new_article_ids: List[int] = []
        session: Session = self._session_factory()
        try:
            feed = self._get_or_create_feed(session, feed_config)
            for entry in entries:
                guid = entry.get("id") or entry.get("guid") or entry.get("link")
                link = entry.get("link")
                if not link:
                    continue

                if self._article_exists(session, feed.id, guid, link):
                    continue

                extracted = self._attempt_extraction(link)
                title = entry.get("title") or (extracted.title if extracted else None)
                summary = _sanitize_text(entry.get("summary"))
                content = extracted.text if extracted else None
                if not content and extracted:
                    content = extracted.summary

                article = Article(
                    feed_id=feed.id,
                    guid=guid,
                    url=link,
                    title=title,
                    summary=summary,
                    content=content,
                    published_at=_parse_datetime(entry),
                )
                session.add(article)
                session.flush()  # Obtain primary key
                log_entry = ArticleIngestionLog(article_id=article.id, feed_id=feed.id)
                session.add(log_entry)
                new_article_ids.append(article.id)

            feed.last_polled_at = datetime.utcnow()
            session.commit()
            article_count = len(new_article_ids)
            status = "success" if article_count else "empty"
        except Exception:
            session.rollback()
            status = "error"
            logger.exception(
                "Failed to ingest feed %s",
                feed_config.url,
                extra={"event": "feed.ingest_error", "feed": feed_config.name},
            )
            raise
        finally:
            session.close()
            metrics.record_ingestion(
                feed_config.name,
                article_count,
                time.perf_counter() - start_time,
                status,
            )

        if new_article_ids:
            logger.info(
                "Created %d new articles for %s",
                len(new_article_ids),
                feed_config.name,
                extra={"event": "feed.articles_created", "feed": feed_config.name, "count": len(new_article_ids)},
            )
        return new_article_ids

    def _get_or_create_feed(self, session: Session, feed_config: FeedSettings) -> Feed:
        feed = session.query(Feed).filter(Feed.url == feed_config.url).one_or_none()
        if feed:
            feed.enabled = feed_config.enabled
            feed.schedule_seconds = feed_config.schedule_seconds
            feed.name = feed_config.name
            feed.metadata_json = None
            if feed_config.metadata:
                feed.metadata_json = json.dumps(feed_config.metadata)
            return feed

        feed = Feed(
            name=feed_config.name,
            url=feed_config.url,
            schedule_seconds=feed_config.schedule_seconds,
            enabled=feed_config.enabled,
            metadata_json=json.dumps(feed_config.metadata) if feed_config.metadata else None,
        )
        session.add(feed)
        session.flush()
        return feed

    def _article_exists(self, session: Session, feed_id: int, guid: Optional[str], url: str) -> bool:
        query = session.query(Article.id).filter(Article.feed_id == feed_id)
        if guid:
            query = query.filter(or_(Article.guid == guid, Article.url == url))
        else:
            query = query.filter(Article.url == url)
        return query.first() is not None

    def _attempt_extraction(self, url: str) -> Optional[ExtractedArticle]:
        try:
            return self._extractor.extract(url)
        except Exception:
            logger.info("Extraction failed for %s", url)
            return None


__all__ = ["RSSIngestionService"]
