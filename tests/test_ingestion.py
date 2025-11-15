from __future__ import annotations

import time
from types import SimpleNamespace

import feedparser

from src.config.settings import FeedSettings
from src.db.models import Article, ArticleIngestionLog, Feed
from src.db.session import session_scope
from src.ingestion.extractor import ExtractedArticle
from src.ingestion.rss import RSSIngestionService
from src.telemetry import metrics


class StubExtractor:
    def extract(self, url: str) -> ExtractedArticle:
        return ExtractedArticle(title="Stub Title", text="Full article text", summary="<p>HTML body</p>")


def test_rss_ingestion_creates_articles(session_factory, monkeypatch) -> None:
    feed_config = FeedSettings(name="Example Feed", url="http://example.com/rss", schedule_seconds=60)
    service = RSSIngestionService(session_factory=session_factory, extractor=StubExtractor())

    published = time.gmtime()
    parsed = SimpleNamespace(
        bozo=False,
        entries=[
            {
                "id": "guid-1",
                "link": "http://example.com/article-1",
                "title": "Example headline",
                "summary": "<p>Summary paragraph</p>",
                "published_parsed": published,
            }
        ],
    )

    monkeypatch.setattr(feedparser, "parse", lambda url: parsed)

    article_ids = service.ingest(feed_config)
    assert article_ids

    with session_scope(session_factory) as session:
        stored_feed = session.query(Feed).one()
        assert stored_feed.url == feed_config.url

        article = session.query(Article).one()
        assert article.id == article_ids[0]
        assert article.title == "Example headline"
        assert article.summary == "Summary paragraph"
        assert article.content == "Full article text"

        logs = session.query(ArticleIngestionLog).all()
        assert len(logs) == 1
        assert logs[0].article_id == article.id

    assert metrics.last_ingestion is not None
    assert metrics.last_ingestion.feed == "Example Feed"
    assert metrics.last_ingestion.article_count == 1
    assert metrics.last_ingestion.status == "success"
