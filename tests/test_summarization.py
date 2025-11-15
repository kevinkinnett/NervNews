from __future__ import annotations

from datetime import datetime, timedelta

from src.config.settings import SummarizationSettings
from src.db.models import Article, Feed, Summary, SummaryEvaluation, UserProfile
from src.db.session import session_scope
from src.llm.summarization import SummaryOrchestrationService
from src.llm.prompts import (
    CRITIC_REVIEW_PROMPT,
    REPORTER_SUMMARY_PROMPT,
    SUMMARY_RELEVANCE_PROMPT,
)
from src.telemetry import metrics


class StubSummarizationLLM:
    def __init__(self) -> None:
        self.reporter_calls = 0
        self.critic_calls = 0

    def generate_structured(self, template, variables, max_retries=None):  # type: ignore[override]
        if template.name == REPORTER_SUMMARY_PROMPT.name:
            self.reporter_calls += 1
            if self.reporter_calls == 1:
                return {
                    "headline": "Initial draft",
                    "summary": "Preliminary summary",
                    "key_points": ["Point A", "Point B"],
                }
            return {
                "headline": "Final headline",
                "summary": "Refined newsroom copy",
                "key_points": ["Point A", "Point B"],
            }
        if template.name == CRITIC_REVIEW_PROMPT.name:
            self.critic_calls += 1
            if self.critic_calls == 1:
                return {
                    "should_revise": True,
                    "strengths": "Good structure",
                    "issues": "Needs more detail",
                    "revision_guidance": "Add context on impact",
                }
            return {
                "should_revise": False,
                "strengths": "Balanced coverage",
                "issues": "",
                "revision_guidance": "",
            }
        if template.name == SUMMARY_RELEVANCE_PROMPT.name:
            return {
                "overall_relevance": {"score": 4, "label": "High", "explanation": "Matches profile"},
                "overall_criticality": {"score": 3, "label": "Medium", "explanation": "Important but stable"},
                "items": [
                    {
                        "key_point": "Point A",
                        "relevance": {"score": 4, "label": "High"},
                        "criticality": {"score": 3, "label": "Medium"},
                        "explanation": "Key update",
                        "escalation": "monitor",
                    },
                    {
                        "key_point": "Point B",
                        "relevance": {"score": 3, "label": "Medium"},
                        "criticality": {"score": 2, "label": "Low"},
                        "explanation": "Secondary detail",
                        "escalation": "inform",
                    },
                ],
            }
        raise AssertionError(f"Unexpected template: {template.name}")


def test_summarization_cycle_creates_summary(session_factory) -> None:
    now = datetime.utcnow()
    enriched_at = now - timedelta(minutes=5)

    with session_scope(session_factory) as session:
        feed = Feed(name="Feed", url="http://example.com/rss", schedule_seconds=300, enabled=True)
        session.add(feed)
        session.flush()
        article = Article(
            feed_id=feed.id,
            url="http://example.com/story",
            title="Newsworthy event",
            summary="Summary",
            brief_summary="News brief",
            topic="Politics",
            enriched_at=enriched_at,
            fetched_at=enriched_at,
            created_at=enriched_at,
        )
        session.add(article)
        profile = UserProfile(title="Analyst", content="Focus on politics", is_active=True)
        session.add(profile)

    settings = SummarizationSettings(
        interval_seconds=3600,
        context_window_chars=2000,
        max_iterations=3,
        historical_days=0,
        max_recent_articles=10,
        max_historical_per_topic=3,
    )
    service = SummaryOrchestrationService(
        session_factory=session_factory,
        llm_client=StubSummarizationLLM(),
        settings=settings,
    )

    service.run_cycle()

    with session_scope(session_factory) as session:
        summary = session.query(Summary).one()
        assert summary.status == "completed"
        assert summary.final_json is not None
        assert summary.iteration_count == 2

        evaluation = session.query(SummaryEvaluation).one()
        assert evaluation.summary_id == summary.id
        assert "overall_relevance" in evaluation.ratings_json

    assert metrics.last_summary_cycle is not None
    assert metrics.last_summary_cycle.status == "completed"
    assert metrics.last_summary_cycle.article_count == 1
