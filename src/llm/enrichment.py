"""Sequential enrichment pipeline that annotates newly ingested articles."""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, Sequence

from sqlalchemy.orm import Session, sessionmaker

from src.db.models import Article
from src.db.session import session_scope
from src.llm.client import LLMClient, LLMClientError
from src.llm.prompts import (
    CATEGORY_CLASSIFICATION_PROMPT,
    JsonPromptTemplate,
    LOCATION_EXTRACTION_PROMPT,
    TOPIC_IDENTIFICATION_PROMPT,
)

logger = logging.getLogger(__name__)


class ArticleEnrichmentService:
    """Run prompt-based enrichment for a batch of article identifiers."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        llm_client: LLMClient,
    ) -> None:
        self._session_factory = session_factory
        self._llm_client = llm_client

    def enrich_articles(self, article_ids: Sequence[int]) -> None:
        ids = list(article_ids)
        if not ids:
            return

        batch_start = time.perf_counter()
        successes = 0
        failures = 0
        with session_scope(self._session_factory) as session:
            articles = (
                session.query(Article)
                .filter(Article.id.in_(ids))
                .all()
            )
            for article in articles:
                if article.enriched_at:
                    logger.debug("Skipping article %s already enriched", article.id)
                    continue
                try:
                    self._enrich_single(session, article)
                    successes += 1
                except Exception as exc:
                    logger.exception("Failed to enrich article %s: %s", article.id, exc)
                    failures += 1

        duration = time.perf_counter() - batch_start
        if successes:
            avg = duration / max(successes, 1)
            logger.info(
                "Enriched %d/%d articles in %.2fs (avg %.2fs per article, failures=%d)",
                successes,
                len(ids),
                duration,
                avg,
                failures,
            )
        else:
            logger.warning(
                "No articles were successfully enriched for ids: %s",
                ids,
            )

    def _collect_variables(self, article: Article) -> Dict[str, str | None]:
        return {
            "title": article.title,
            "summary": article.summary,
            "content": article.content,
        }

    def _enrich_single(self, session: Session, article: Article) -> None:
        timer_start = time.perf_counter()
        variables = self._collect_variables(article)

        if not any(variables.values()):
            raise ValueError("Article has no textual content to enrich")

        location = self._safe_invoke(LOCATION_EXTRACTION_PROMPT, variables)
        topic = self._safe_invoke(TOPIC_IDENTIFICATION_PROMPT, variables)
        classification = self._safe_invoke(CATEGORY_CLASSIFICATION_PROMPT, variables)

        article.location_name = location.get("location_name")
        article.location_country = location.get("country")
        article.location_confidence = self._to_float(location.get("confidence"))
        article.location_justification = location.get("justification")

        article.topic = topic.get("topic")
        article.topic_confidence = self._to_float(topic.get("confidence"))
        article.topic_supporting_points = topic.get("supporting_points")

        article.category = classification.get("category")
        article.subcategory = classification.get("subcategory")
        article.category_confidence = self._to_float(classification.get("confidence"))
        article.category_rationale = classification.get("rationale")

        article.enriched_at = datetime.utcnow()

        duration = time.perf_counter() - timer_start
        logger.info(
            "Article %s enriched in %.2fs (topic=%s[%.2f], category=%s/%s[%.2f], location_conf=%.2f)",
            article.id,
            duration,
            article.topic,
            article.topic_confidence or 0.0,
            article.category,
            article.subcategory,
            article.category_confidence or 0.0,
            article.location_confidence or 0.0,
        )
        session.add(article)

    def _safe_invoke(
        self,
        prompt: JsonPromptTemplate,
        variables: Dict[str, str | None],
    ) -> Dict[str, object]:
        try:
            return self._llm_client.generate_structured(prompt, variables)
        except LLMClientError as exc:
            logger.warning("Primary enrichment call failed: %s", exc)
            fallback_variables = dict(variables)
            fallback_variables["content"] = variables.get("summary") or variables.get("title")
            return self._llm_client.generate_structured(
                prompt,
                fallback_variables,
                max_retries=2,
            )

    @staticmethod
    def _to_float(value) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None


__all__ = ["ArticleEnrichmentService"]
