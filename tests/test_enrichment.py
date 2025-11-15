from __future__ import annotations

from datetime import datetime

from src.db.models import Article, Feed
from src.db.session import session_scope
from src.llm.enrichment import ArticleEnrichmentService
from src.llm.prompts import (
    ARTICLE_BRIEF_PROMPT,
    CATEGORY_CLASSIFICATION_PROMPT,
    LOCATION_EXTRACTION_PROMPT,
    TOPIC_IDENTIFICATION_PROMPT,
)
from src.telemetry import metrics


class StubLLMClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_structured(self, template, variables, max_retries=None):  # type: ignore[override]
        self.calls.append(template.name)
        if template.name == LOCATION_EXTRACTION_PROMPT.name:
            return {
                "location_name": "London",
                "country": "United Kingdom",
                "confidence": 0.9,
                "justification": "Mentioned as dateline",
            }
        if template.name == TOPIC_IDENTIFICATION_PROMPT.name:
            return {
                "topic": "Politics",
                "confidence": 0.85,
                "supporting_points": "Discusses parliamentary vote",
            }
        if template.name == CATEGORY_CLASSIFICATION_PROMPT.name:
            return {
                "category": "Politics",
                "subcategory": "Elections",
                "confidence": 0.8,
                "rationale": "Covers campaign trail",
            }
        if template.name == ARTICLE_BRIEF_PROMPT.name:
            return {"brief": "Concise capsule"}
        raise AssertionError(f"Unexpected prompt {template.name}")


def test_article_enrichment_updates_fields(session_factory) -> None:
    with session_scope(session_factory) as session:
        feed = Feed(name="Feed", url="http://example.com/rss", schedule_seconds=120, enabled=True)
        session.add(feed)
        session.flush()
        article = Article(
            feed_id=feed.id,
            url="http://example.com/story",
            title="Campaign update",
            summary="Summary text",
            content="Full text",
            published_at=datetime.utcnow(),
        )
        session.add(article)

    service = ArticleEnrichmentService(session_factory=session_factory, llm_client=StubLLMClient())
    service.enrich_articles([article.id])

    with session_scope(session_factory) as session:
        enriched = session.get(Article, article.id)
        assert enriched is not None
        assert enriched.enriched_at is not None
        assert enriched.location_name == "London"
        assert enriched.topic == "Politics"
        assert enriched.category == "Politics"
        assert enriched.brief_summary == "Concise capsule"

    assert metrics.last_enrichment is not None
    assert metrics.last_enrichment.attempted == 1
    assert metrics.last_enrichment.successes == 1
    assert metrics.last_enrichment.failures == 0
