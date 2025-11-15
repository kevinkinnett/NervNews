from __future__ import annotations

from datetime import datetime

from src.config.settings import LLMSettings
from src.db.models import Article, Feed
from src.db.session import session_scope
from src.llm.client import LLMClientError
from src.llm.enrichment import ArticleEnrichmentService
from src.llm.prompts import (
    ARTICLE_BRIEF_PROMPT,
    CATEGORY_CLASSIFICATION_PROMPT,
    JsonPromptTemplate,
    LOCATION_EXTRACTION_PROMPT,
    TOPIC_IDENTIFICATION_PROMPT,
)
from src.telemetry import metrics


class StubLLMClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.settings = LLMSettings()

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


class RecordingLLMClient:
    def __init__(self, *, context_window: int, fail_first: bool = False) -> None:
        self.settings = LLMSettings(context_window=context_window)
        self.calls: list[dict[str, object]] = []
        self.fail_first = fail_first
        self._attempts = 0

    def generate_structured(self, template, variables, max_retries=None):  # type: ignore[override]
        self._attempts += 1
        self.calls.append(dict(variables))
        if self.fail_first and self._attempts == 1:
            raise LLMClientError("boom")
        return {"result": "ok"}


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


def test_safe_invoke_uses_summary_when_content_too_large(caplog) -> None:
    llm = RecordingLLMClient(context_window=30)
    service = ArticleEnrichmentService(session_factory=None, llm_client=llm)  # type: ignore[arg-type]

    template = JsonPromptTemplate(
        name="summary-test",
        system_prompt="System instructions.",
        user_template="Summarise the following content:\n{content}",
        response_schema={
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        },
    )

    variables = {
        "title": "Title",
        "summary": "short summary",
        "content": "x" * 500,
    }

    caplog.set_level("WARNING")

    result = service._safe_invoke(template, variables)

    assert result == {"result": "ok"}
    assert llm.calls, "expected LLM to be invoked"
    used_variables = llm.calls[-1]
    assert used_variables["content"] == "short summary"
    assert any("Using article summary" in record.message for record in caplog.records)
    assert variables["content"] == "x" * 500, "original variables should remain untouched"


def test_safe_invoke_truncates_when_no_summary(caplog) -> None:
    llm = RecordingLLMClient(context_window=24)
    service = ArticleEnrichmentService(session_factory=None, llm_client=llm)  # type: ignore[arg-type]

    template = JsonPromptTemplate(
        name="truncate-test",
        system_prompt="System instructions.",
        user_template="Provide insights for:\n{content}",
        response_schema={
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        },
    )

    variables = {
        "title": "Important News",
        "content": "y" * 1000,
    }

    caplog.set_level("WARNING")

    result = service._safe_invoke(template, variables)

    assert result == {"result": "ok"}
    assert llm.calls, "expected LLM to be invoked"
    used_variables = llm.calls[-1]
    assert len(used_variables["content"]) < len(variables["content"])
    assert len(used_variables["content"]) <= llm.settings.context_window * 4
    assert any("Truncated article content" in record.message for record in caplog.records)


def test_safe_invoke_compacts_fallback_after_failure(caplog) -> None:
    llm = RecordingLLMClient(context_window=24, fail_first=True)
    service = ArticleEnrichmentService(session_factory=None, llm_client=llm)  # type: ignore[arg-type]

    template = JsonPromptTemplate(
        name="fallback-test",
        system_prompt="System instructions.",
        user_template="Analyse:\n{content}",
        response_schema={
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        },
    )

    variables = {
        "title": "Fallback Title",
        "content": "z" * 800,
    }

    caplog.set_level("WARNING")

    result = service._safe_invoke(template, variables)

    assert result == {"result": "ok"}
    assert len(llm.calls) == 2
    fallback_variables = llm.calls[-1]
    assert len(fallback_variables["content"]) <= llm.settings.context_window * 4
    assert any("Primary enrichment call failed" in record.message for record in caplog.records)
