"""Reporter-style summarisation orchestrator."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import List, Sequence

from sqlalchemy import or_
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import SummarizationSettings
from src.db.models import Article, Summary
from src.db.session import session_scope
from src.llm.client import LLMClient, LLMClientError
from src.llm.prompts import (
    CRITIC_REVIEW_PROMPT,
    REPORTER_SUMMARY_PROMPT,
)

logger = logging.getLogger(__name__)


class SummaryOrchestrationService:
    """Generate newsroom-style summaries with critic review."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        llm_client: LLMClient,
        settings: SummarizationSettings,
    ) -> None:
        self._session_factory = session_factory
        self._llm_client = llm_client
        self._settings = settings

    def run_cycle(self) -> None:
        """Run a single summarisation cycle."""

        now = datetime.utcnow()
        window_start = self._compute_window_start(now)
        window_end = now

        with session_scope(self._session_factory) as session:
            recent_articles = self._fetch_recent_articles(session, window_start, window_end)
            if not recent_articles:
                logger.info(
                    "No new articles ready for summarisation between %s and %s",
                    window_start,
                    window_end,
                )
                return

            summary_record = Summary(
                window_start=window_start,
                window_end=window_end,
                article_ids_json=json.dumps([article.id for article in recent_articles]),
                status="draft",
            )
            session.add(summary_record)
            session.flush()

            historical_articles = self._fetch_historical_context(session, window_start, recent_articles)
            recent_context = self._render_context(recent_articles, self._recent_limit)
            historical_context = self._render_context(historical_articles, self._historical_limit)

            context_digest = self._truncate(
                "\n".join(filter(None, [recent_context, historical_context])),
                self._settings.context_window_chars,
            )

            feedback_note = ""
            iteration = 0
            final_payload = None

            try:
                while iteration < self._settings.max_iterations:
                    iteration += 1
                    reporter_payload = self._invoke_reporter(
                        time_window=(window_start, window_end),
                        recent_context=recent_context,
                        historical_context=historical_context,
                        feedback=feedback_note,
                    )

                    summary_record.draft_json = json.dumps(reporter_payload)
                    summary_record.iteration_count = iteration

                    critic_payload = self._invoke_critic(
                        time_window=(window_start, window_end),
                        reporter_payload=reporter_payload,
                        context_digest=context_digest,
                    )

                    summary_record.critic_feedback_json = json.dumps(critic_payload)

                    if not critic_payload.get("should_revise") or iteration >= self._settings.max_iterations:
                        final_payload = reporter_payload
                        break

                    feedback_note = self._build_feedback_note(critic_payload)

                if final_payload is None:
                    final_payload = reporter_payload

                summary_record.final_json = json.dumps(final_payload)
                summary_record.status = "completed"
                logger.info(
                    "Summary %s completed for %d articles after %d iterations",
                    summary_record.id,
                    len(recent_articles),
                    summary_record.iteration_count,
                )
            except LLMClientError:
                logger.exception("Summarisation cycle failed due to LLM error")
                summary_record.status = "error"
            except Exception:
                logger.exception("Unexpected error during summarisation cycle")
                summary_record.status = "error"
                raise

    def _compute_window_start(self, now: datetime) -> datetime:
        interval = timedelta(seconds=self._settings.interval_seconds)
        default_start = now - interval

        with session_scope(self._session_factory) as session:
            last_summary = (
                session.query(Summary)
                .order_by(Summary.window_end.desc())
                .first()
            )
        if last_summary is None:
            return default_start
        return max(last_summary.window_end, default_start)

    def _fetch_recent_articles(
        self,
        session: Session,
        window_start: datetime,
        window_end: datetime,
    ) -> List[Article]:
        articles = (
            session.query(Article)
            .filter(Article.enriched_at.isnot(None))
            .filter(Article.enriched_at >= window_start)
            .filter(Article.enriched_at <= window_end)
            .order_by(Article.enriched_at.asc())
            .all()
        )

        scored: List[tuple[datetime, Article]] = []
        for article in articles:
            timestamp = article.published_at or article.enriched_at or article.fetched_at or article.created_at
            if timestamp is None:
                timestamp = window_end
            scored.append((timestamp, article))

        scored.sort(key=lambda item: item[0], reverse=True)
        limited = [item[1] for item in scored[: self._settings.max_recent_articles]]
        return limited

    def _fetch_historical_context(
        self,
        session: Session,
        window_start: datetime,
        recent_articles: Sequence[Article],
    ) -> List[Article]:
        topics = {article.topic for article in recent_articles if article.topic}
        if not topics or self._settings.historical_days <= 0:
            return []

        cutoff = window_start - timedelta(days=self._settings.historical_days)
        recent_ids = {article.id for article in recent_articles}

        candidate_articles = (
            session.query(Article)
            .filter(Article.topic.in_(topics))
            .filter(Article.enriched_at.isnot(None))
            .filter(Article.enriched_at < window_start)
            .filter(
                or_(
                    Article.published_at >= cutoff,
                    Article.fetched_at >= cutoff,
                    Article.created_at >= cutoff,
                )
            )
            .order_by(Article.enriched_at.desc())
            .all()
        )

        grouped: dict[str, List[Article]] = {topic: [] for topic in topics}
        for article in candidate_articles:
            if article.id in recent_ids or not article.topic:
                continue
            if article.topic in grouped and len(grouped[article.topic]) >= self._settings.max_historical_per_topic:
                continue
            grouped.setdefault(article.topic, []).append(article)

        historical: List[Article] = []
        for bucket in grouped.values():
            historical.extend(bucket)
        return historical

    def _invoke_reporter(
        self,
        *,
        time_window: tuple[datetime, datetime],
        recent_context: str,
        historical_context: str,
        feedback: str,
    ) -> dict:
        start, end = time_window
        payload = self._llm_client.generate_structured(
            REPORTER_SUMMARY_PROMPT,
            {
                "time_window": f"{start.isoformat()}Z – {end.isoformat()}Z",
                "recent_context": recent_context or "(no new coverage)",
                "historical_context": historical_context or "(no notable history)",
                "feedback": feedback or "(none)",
            },
        )
        return payload

    def _invoke_critic(
        self,
        *,
        time_window: tuple[datetime, datetime],
        reporter_payload: dict,
        context_digest: str,
    ) -> dict:
        start, end = time_window
        points = reporter_payload.get("key_points") or []
        key_points_rendered = "\n".join(f"- {point}" for point in points)
        payload = self._llm_client.generate_structured(
            CRITIC_REVIEW_PROMPT,
            {
                "time_window": f"{start.isoformat()}Z – {end.isoformat()}Z",
                "draft_headline": reporter_payload.get("headline", ""),
                "draft_summary": reporter_payload.get("summary", ""),
                "draft_key_points": key_points_rendered or "(no key points)",
                "context_digest": context_digest or "(no context)",
            },
        )
        return payload

    def _render_context(self, articles: Sequence[Article], char_limit: int) -> str:
        snippets: List[str] = []
        remaining = max(char_limit, 0)
        for article in articles:
            snippet = self._format_article_snippet(article)
            if not snippet:
                continue
            if len(snippet) + 1 > remaining and snippets:
                break
            snippets.append(snippet)
            remaining -= len(snippet) + 1
        return "\n".join(snippets)

    def _format_article_snippet(self, article: Article) -> str:
        timestamp = article.published_at or article.enriched_at or article.fetched_at
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M UTC") if timestamp else "Unknown time"
        topic = article.topic or "General"
        title = article.title or article.url
        summary = article.brief_summary or article.summary or ""
        summary = self._truncate(summary, 320)
        return f"- [{timestamp_str}] ({topic}) {title}: {summary}"

    def _build_feedback_note(self, critic_payload: dict) -> str:
        issues = critic_payload.get("issues", "").strip()
        guidance = critic_payload.get("revision_guidance", "").strip()
        return "\n".join(filter(None, [issues, guidance]))

    def _truncate(self, text: str, limit: int) -> str:
        if not text or len(text) <= limit:
            return text
        return text[: max(limit - 3, 0)].rstrip() + "..."

    @property
    def _recent_limit(self) -> int:
        return int(self._settings.context_window_chars * 0.6)

    @property
    def _historical_limit(self) -> int:
        return self._settings.context_window_chars - self._recent_limit


__all__ = ["SummaryOrchestrationService"]
